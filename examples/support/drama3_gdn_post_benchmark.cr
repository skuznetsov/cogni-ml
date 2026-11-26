# Diagnostic-only capture, device receipts, and matched replay support for the
# post-GDN RMS -> SiLU -> multiply fusion benchmark.
require "json"
require "set"
require "../../src/ml/audio/drama3/generator_resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  module ML::Audio::Drama3
    class WeightStore
      def gdn_post_norm_tensor_names : Array(String)
        @tensors.keys.select do |name|
          name.starts_with?("backbone.layers.") && name.ends_with?(".linear_attn.norm.weight")
        end.sort
      end
    end
  end

  module Drama3GDNPostBenchmarkSupport
    include ML::Audio::Drama3
    alias Context = Resident::Context
    alias Buffer = Resident::Buffer
    alias Signature = Tuple(Int32, Int32, Int32, Float64)

    SELECTED_ROWS      = [0, 29, 59] of Int32
    LAYER_COUNT        = 30
    EXPECTED_LAYER_IDS = (0...40).select { |layer| (layer + 1) % 4 != 0 }

    RECEIPT_SOURCE = Resident::SOURCE + <<-METAL
      inline uint dr3_gdn_bench_class(float value) {
          if (isnan(value)) return 1u;
          if (isinf(value)) return (as_type<uint>(value) >> 31) ? 3u : 2u;
          return 0u;
      }
      inline void dr3_gdn_bench_record(device atomic_uint* receipt, uint base,
                                       float expected, float actual) {
          uint ec = dr3_gdn_bench_class(expected), ac = dr3_gdn_bench_class(actual);
          if (ec == 0u && ac == 0u && as_type<uint>(expected) != as_type<uint>(actual))
              atomic_fetch_add_explicit(&receipt[base], 1u, memory_order_relaxed);
          if (ec != ac) atomic_fetch_add_explicit(&receipt[base + 1], 1u, memory_order_relaxed);
          if (ec == 0u) atomic_fetch_add_explicit(&receipt[base + 2], 1u, memory_order_relaxed);
          if (ac == 0u) atomic_fetch_add_explicit(&receipt[base + 3], 1u, memory_order_relaxed);
          if (ec == 1u) atomic_fetch_add_explicit(&receipt[base + 4], 1u, memory_order_relaxed);
          if (ac == 1u) atomic_fetch_add_explicit(&receipt[base + 5], 1u, memory_order_relaxed);
          if (ec == 2u) atomic_fetch_add_explicit(&receipt[base + 6], 1u, memory_order_relaxed);
          if (ac == 2u) atomic_fetch_add_explicit(&receipt[base + 7], 1u, memory_order_relaxed);
          if (ec == 3u) atomic_fetch_add_explicit(&receipt[base + 8], 1u, memory_order_relaxed);
          if (ac == 3u) atomic_fetch_add_explicit(&receipt[base + 9], 1u, memory_order_relaxed);
      }
      kernel void dr3_gdn_bench_compare(
          device const float* expected [[buffer(0)]], device const float* actual [[buffer(1)]],
          device atomic_uint* receipt [[buffer(2)]], constant ulong* params [[buffer(3)]],
          uint index [[thread_position_in_grid]]) {
          if (ulong(index) >= params[0]) return;
          dr3_gdn_bench_record(receipt, uint(params[1]), expected[index], actual[index]);
      }
      METAL

    record InputVariation, name : String, kind : String, from_row : Int32, to_row : Int32,
      expected : Buffer, actual : Buffer

    class PostCall
      getter frame : Int32
      getter position : Int32
      getter name : String
      getter layer : Int32
      getter x : Buffer
      getter z : Buffer
      getter w : Buffer
      getter candidate : Buffer
      getter eps : Float64
      getter heads : Int32

      def initialize(@frame, @position, @name, @layer, @x, @z, @w, @candidate, @eps, @heads)
      end

      def signature : Signature
        {@heads, @x.count, @w.count, @eps}
      end

      def metadata : JSON::Any
        JSON.parse({frame: @frame, position: @position, layer: @layer, norm_weight: @name, heads: @heads,
                    eps: @eps, x_values: @x.count, z_values: @z.count, weight_values: @w.count,
                    candidate_values: @candidate.count, signature: signature.to_a}.to_json)
      end
    end

    class Capture
      property frame = 0
      property active = false
      getter records = [] of PostCall
      getter expected_names : Array(String)
      getter selected_candidate_dispatches = 0_i64
      getter prompt_rows : Int32
      @armed_name : String?
      @armed_weight : Buffer?
      @context : Context
      @initial_dispatches : Int64
      @observed_position : Int32?
      @observed_layers : Array(Int32)

      def initialize(@context : Context, store : WeightStore, @prompt_rows : Int32)
        @expected_names = store.gdn_post_norm_tensor_names
        raise "expected exactly #{LAYER_COUNT} linear-attention GDN norm tensors, found #{@expected_names.size}" unless @expected_names.size == LAYER_COUNT
        expected_layers = @expected_names.map { |name| parse_layer(name) }.sort
        raise "GDN norm tensor names do not match the expected layer schedule" unless expected_layers == EXPECTED_LAYER_IDS
        raise "capture requires the frozen candidate selector" unless @context.gdn_post_fusion_enabled
        @initial_dispatches = @context.gdn_post_fused_dispatches
        @armed_name = nil
        @armed_weight = nil
        @observed_position = nil
        @observed_layers = [] of Int32
      end

      private def parse_layer(name : String) : Int32
        parts = name.split('.')
        raise "unexpected GDN norm tensor name #{name}" unless parts.size == 6 && parts[0] == "backbone" &&
                                                               parts[1] == "layers" && parts[3] == "linear_attn" && parts[4] == "norm" && parts[5] == "weight"
        parts[2].to_i? || raise "invalid GDN layer in #{name}"
      end

      def begin_layer(position : Int32, layer : Int32, prefix : String) : Nil
        raise "GDN position is outside the configured 40-layer backbone: #{position}" if position < 0
        raise "unexpected linear-attention prefix #{prefix} for layer #{layer}" unless prefix == "backbone.layers.#{layer}.linear_attn"
        raise "unexpected GDN layer #{layer}" unless EXPECTED_LAYER_IDS.includes?(layer)

        if previous_position = @observed_position
          if previous_position != position
            raise "GDN layer sequence at position #{previous_position} is incomplete" unless @observed_layers == EXPECTED_LAYER_IDS
            raise "GDN position skipped or repeated: #{previous_position} -> #{position}" unless position == previous_position + 1
            @observed_layers.clear
          end
        else
          raise "GDN position sequence must start at zero, got #{position}" unless position == 0
        end

        if @observed_position != position
          @observed_position = position
          @frame = position - @prompt_rows
        end

        expected_next = EXPECTED_LAYER_IDS[@observed_layers.size]?
        raise "GDN layer order mismatch at position #{position}: got #{layer}, expected #{expected_next}" unless expected_next == layer
        @observed_layers << layer
        @active = SELECTED_ROWS.includes?(@frame)
      end

      def end_layer(position : Int32, layer : Int32) : Nil
        raise "GDN layer completion does not match observed position/layer" unless @observed_position == position && @observed_layers.last? == layer
        @active = false
        @armed_name = nil
        @armed_weight = nil
      end

      def finish_generation! : Nil
        @active = false
        raise "generation ended before all GDN layers ran at the last position" unless @observed_layers == EXPECTED_LAYER_IDS
        raise "generation did not reach every selected GDN row" unless SELECTED_ROWS.all? do |row|
                                                                         @records.count { |record| record.frame == row } == LAYER_COUNT
                                                                       end
      end

      def arm_weight(name : String, value : Buffer) : Nil
        return unless @active
        return unless name.starts_with?("backbone.layers.") && name.ends_with?(".linear_attn.norm.weight")
        raise "GDN norm weight armed twice before post call" if @armed_name
        @armed_name = name
        @armed_weight = value
      end

      private def same_buffer?(left : Buffer, right : Buffer) : Bool
        left.buffer.handle.address == right.buffer.handle.address && left.offset == right.offset && left.count == right.count
      end

      private def snapshot(value : Buffer) : Buffer
        saved = @context.alloc(value.count)
        raise "GDN capture snapshot aliases input" if saved.buffer.handle.address == value.buffer.handle.address
        @context.copy(value, saved)
        saved
      end

      def intercept(x : Buffer, z : Buffer, w : Buffer, eps : Float64, heads : Int32,
                    & : -> Buffer) : Buffer
        return yield unless @active
        name = @armed_name || raise "selected GDN post call has no scoped norm weight name"
        armed = @armed_weight || raise "selected GDN post call has no scoped norm weight buffer"
        raise "GDN post weight/name mapping changed for #{name}" unless same_buffer?(armed, w)
        raise "selected GDN post call row mismatch" unless SELECTED_ROWS.includes?(@frame)
        raise "GDN capture shape mismatch for #{name}" unless x.count == z.count && heads > 0 && x.count % heads == 0 && w.count == x.count // heads
        parts = name.split('.')
        raise "unexpected GDN norm tensor name #{name}" unless parts.size == 6 && parts[0] == "backbone" &&
                                                               parts[1] == "layers" && parts[3] == "linear_attn" && parts[4] == "norm" && parts[5] == "weight"
        layer = parts[2].to_i? || raise "invalid GDN layer in #{name}"
        before_graph, before_reads = @context.graph_stats.dup, @context.readbacks.dup
        before_dispatches = @context.gdn_post_fused_dispatches

        # Snapshot every candidate input before the fused command can consume
        # it, then snapshot its output as the first operation after that call.
        saved_x, saved_z, saved_w = snapshot(x), snapshot(z), snapshot(w)
        @active = false
        output = yield
        saved_output = snapshot(output)
        raise "GDN candidate silently fell back for #{name}" unless @context.gdn_post_fused_dispatches == before_dispatches + 1
        after_graph = @context.graph_stats
        raise "GDN capture submitted or waited before its terminal receipt" unless after_graph["submitted"] == before_graph["submitted"] &&
                                                                                   after_graph["completion_waits"] == before_graph["completion_waits"] &&
                                                                                   @context.readbacks == before_reads
        @records << PostCall.new(@frame, @observed_position || raise("GDN post has no observed backbone position"),
          name, layer, saved_x, saved_z, saved_w, saved_output, eps, heads)
        @selected_candidate_dispatches += 1_i64
        output
      ensure
        @armed_name = nil
        @armed_weight = nil
        @active = false
      end

      def assert_inventory! : Nil
        raise "capture rows differ from requested #{SELECTED_ROWS}" unless @records.map(&.frame).uniq.sort == SELECTED_ROWS
        SELECTED_ROWS.each do |row|
          observed = @records.select { |record| record.frame == row }.map(&.name).sort
          raise "row #{row} captured #{observed.size} GDN calls, expected #{LAYER_COUNT}" unless observed.size == LAYER_COUNT
          raise "row #{row} GDN norm name inventory differs" unless observed == @expected_names
          expected_position = @prompt_rows + row
          raise "row #{row} maps to an unexpected backbone position" unless @records.select { |record| record.frame == row }.all? { |record| record.position == expected_position }
          observed_layers = @records.select { |record| record.frame == row }.map(&.layer).sort
          raise "row #{row} layer IDs differ from the expected schedule" unless observed_layers == EXPECTED_LAYER_IDS
        end
        raise "capture total is #{@records.size}, expected #{SELECTED_ROWS.size * LAYER_COUNT}" unless @records.size == SELECTED_ROWS.size * LAYER_COUNT
        raise "selected candidate dispatches differ from captured calls" unless @selected_candidate_dispatches == @records.size
      end

      def input_variations : Array(InputVariation)
        grouped = {} of Tuple(Int32, String) => PostCall
        @records.each { |record| grouped[{record.frame, record.name}] = record }
        variations = [] of InputVariation
        [{0, 29}, {29, 59}].each do |from, to|
          @expected_names.each do |name|
            left = grouped[{from, name}]? || raise "missing GDN input #{name} at row #{from}"
            right = grouped[{to, name}]? || raise "missing GDN input #{name} at row #{to}"
            raise "GDN input signature changed for #{name}" unless left.signature == right.signature
            variations << InputVariation.new(name, "x", from, to, left.x, right.x)
            variations << InputVariation.new(name, "z", from, to, left.z, right.z)
          end
        end
        variations
      end

      def inventory : JSON::Any
        assert_inventory!
        JSON.parse({selected_rows: SELECTED_ROWS, total_calls: @records.size,
                    calls_per_row: SELECTED_ROWS.to_h { |row| {row.to_s, @records.count { |record| record.frame == row }} },
                    expected_norm_weight_names: @expected_names,
                    signatures: @records.map(&.signature).uniq.sort_by(&.to_a.join(":")),
                    selected_captured_candidate_dispatches: @selected_candidate_dispatches,
                    context_dispatch_delta_including_generation_and_local_replay: @context.gdn_post_fused_dispatches - @initial_dispatches}.to_json)
      end
    end

    record Arm, name : String, command_index : Int64, outputs : Array(Buffer),
      host_encode_seconds : Float64, fused_dispatches : Int64

    def self.clear(context : Context, receipt : Buffer) : Nil
      context.dispatch("dr3_fill", Resident::SOURCE, [receipt], [receipt.count.to_u64, 0_u64], receipt.count)
    end

    def self.compare(context : Context, expected : Buffer, actual : Buffer,
                     receipt : Buffer, pair_index : Int32) : Nil
      raise "GDN comparison shape mismatch" unless expected.count == actual.count
      context.dispatch("dr3_gdn_bench_compare", RECEIPT_SOURCE, [expected, actual, receipt],
        [expected.count.to_u64, (pair_index * 10).to_u64], expected.count)
    end

    def self.read_receipt(context : Context, receipt : Buffer) : Array(UInt32)
      context.read(receipt, "terminal").map(&.unsafe_as(UInt32))
    end

    def self.accepts?(counts : Array(UInt32), base : Int32) : Bool
      counts[base] == 0_u32 && counts[base + 1] == 0_u32 &&
        counts[base + 2] == counts[base + 3] && counts[base + 4] == counts[base + 5] &&
        counts[base + 6] == counts[base + 7] && counts[base + 8] == counts[base + 9]
    end

    def self.finite_exact?(counts : Array(UInt32), base : Int32, values : Int32) : Bool
      accepts?(counts, base) && counts[base + 2] == values.to_u32 && counts[base + 4] == 0_u32 &&
        counts[base + 6] == 0_u32 && counts[base + 8] == 0_u32
    end

    def self.verify_capture(context : Context, capture : Capture) : JSON::Any
      capture.assert_inventory!
      before_graph, before_reads = context.graph_stats.dup, context.readbacks.dup
      references = capture.records.map do |record|
        context.mul(context.rms(record.x, record.w, record.eps, heads: record.heads), context.silu(record.z))
      end
      raise "explicit GDN references submitted or read back" unless context.graph_stats["submitted"] == before_graph["submitted"] &&
                                                                    context.graph_stats["completion_waits"] == before_graph["completion_waits"] && context.readbacks == before_reads

      variations = capture.input_variations
      comparison_count = capture.records.size + variations.size + 1
      receipt = context.alloc(comparison_count * 10)
      clear(context, receipt)
      capture.records.each_with_index do |record, index|
        compare(context, references[index], record.candidate, receipt, index)
      end
      variations.each_with_index do |variation, index|
        compare(context, variation.expected, variation.actual, receipt, capture.records.size + index)
      end

      one = 1_f32
      one_ulp_up = (one.unsafe_as(UInt32) + 1_u32).unsafe_as(Float32)
      negative_expected = context.upload([one, Float32::NAN, Float32::INFINITY, -Float32::INFINITY])
      negative_actual = context.upload([one_ulp_up, 0_f32, -Float32::INFINITY, Float32::INFINITY])
      negative_index = comparison_count - 1
      compare(context, negative_expected, negative_actual, receipt, negative_index)
      raise "GDN capture validation submitted or waited before terminal read" unless context.graph_stats["submitted"] == before_graph["submitted"] &&
                                                                                     context.graph_stats["completion_waits"] == before_graph["completion_waits"] && context.readbacks == before_reads

      counts = read_receipt(context, receipt)
      capture.records.each_with_index do |record, index|
        base = index * 10
        raise "explicit/candidate GDN mismatch for #{record.name} row #{record.frame}: #{counts[base, 10]}" unless finite_exact?(counts, base, record.candidate.count)
      end
      variation_reports = variations.each_with_index.map do |variation, index|
        base = (capture.records.size + index) * 10
        local = counts[base, 10]
        raise "captured #{variation.kind} did not vary across rows for #{variation.name}" unless local[0] > 0_u32 && local[1] == 0_u32 && local[2] == local[3] && local[4] == 0_u32 && local[6] == 0_u32 && local[8] == 0_u32
        {name: variation.name, kind: variation.kind, from_row: variation.from_row,
         to_row: variation.to_row, bit_mismatches: local[0], finite_values: local[2]}
      end
      control = negative_index * 10
      expected_control = [1_u32, 3_u32, 1_u32, 2_u32, 1_u32, 0_u32,
                          1_u32, 1_u32, 1_u32, 1_u32]
      raise "seeded GDN comparator control was not rejected: #{counts[control, 10]}" unless counts[control, 10] == expected_control
      after_graph = context.graph_stats.dup
      raise "capture comparator did not drain" unless after_graph["pending"] == 0 && after_graph["retained_bytes"] == 0
      JSON.parse({candidate_reference_pairs: capture.records.size, bit_mismatches: 0,
                  nonfinite_values: 0, input_variations: variation_reports,
                  seeded_negative_control: counts[control, 10], graph_before: before_graph,
                  graph_after: after_graph, readbacks_before: before_reads,
                  readbacks_after: context.readbacks, nan_payloads_compared: false}.to_json)
    end

    def self.encode_arm(context : Context, records : Array(PostCall), name : String, repeats : Int32) : Arm
      command_index = context.graph_stats["submitted"]
      before_dispatches = context.gdn_post_fused_dispatches
      started = Time.instant
      outputs = [] of Buffer
      repeats.times do
        records.each do |record|
          output = if name == "candidate"
                     context.gdn_post(record.x, record.z, record.w, record.eps, heads: record.heads)
                   else
                     context.mul(context.rms(record.x, record.w, record.eps, heads: record.heads), context.silu(record.z))
                   end
          outputs << output
        end
      end
      elapsed = (Time.instant - started).total_seconds
      dispatches = context.gdn_post_fused_dispatches - before_dispatches
      expected = name == "candidate" ? (records.size * repeats).to_i64 : 0_i64
      raise "#{name} GDN fused dispatches #{dispatches}, expected #{expected}" unless dispatches == expected
      context.submit
      Arm.new(name, command_index, outputs, elapsed, dispatches)
    end

    def self.pair(context : Context, records : Array(PostCall), pair_index : Int32,
                  warmups : Int32, repeats : Int32, negative : Bool) : JSON::Any
      before, reads_before = context.graph_stats.dup, context.readbacks.dup
      raise "GDN pair requires a drained queue" unless before["pending"] == 0 && before["retained_bytes"] == 0
      order = pair_index % 4 == 0 || pair_index % 4 == 3 ? ["unfused", "candidate"] : ["candidate", "unfused"]
      arms = {} of String => Arm
      order.each do |name|
        arm = encode_arm(context, records, name, repeats)
        raise "GDN pair armed an early read/wait" unless context.readbacks == reads_before &&
                                                         context.graph_stats["completion_waits"] == before["completion_waits"]
        arms[name] = arm
      end
      queued = context.graph_stats
      raise "GDN pair did not retain two in-flight arm commands" unless queued["pending"] == 2 &&
                                                                        queued["submitted"] == before["submitted"] + 2

      unfused, candidate = arms["unfused"], arms["candidate"]
      raise "GDN arm output count mismatch" unless unfused.outputs.size == candidate.outputs.size
      values = unfused.outputs.sum(0_i64, &.count.to_i64)
      control_words = negative ? 10 : 0
      receipt = context.alloc(unfused.outputs.size * 10 + control_words)
      clear(context, receipt)
      unfused.outputs.zip(candidate.outputs).each_with_index do |(expected, actual), index|
        compare(context, expected, actual, receipt, index)
      end
      if negative
        one = 1_f32
        bad = (one.unsafe_as(UInt32) + 1_u32).unsafe_as(Float32)
        expected = context.upload([one, Float32::NAN, Float32::INFINITY, -Float32::INFINITY])
        actual = context.upload([bad, 0_f32, -Float32::INFINITY, Float32::INFINITY])
        compare(context, expected, actual, receipt, unfused.outputs.size)
      end
      context.finish
      counts = read_receipt(context, receipt)
      outputs = unfused.outputs.size
      outputs.times do |index|
        raise "paired GDN mismatch: #{counts[index * 10, 10]}" unless finite_exact?(counts, index * 10, unfused.outputs[index].count)
      end
      control = negative ? outputs * 10 : nil
      if negative
        expected_control = [1_u32, 3_u32, 1_u32, 2_u32, 1_u32, 0_u32,
                            1_u32, 1_u32, 1_u32, 1_u32]
        raise "paired seeded defect was accepted: #{counts[control.not_nil!, 10]}" unless counts[control.not_nil!, 10] == expected_control
      end
      after = context.graph_stats.dup
      raise "GDN pair failed to drain" unless after["pending"] == 0 && after["retained_bytes"] == 0
      expected_bytes = receipt.count.to_i64 * 4_i64
      raise "GDN pair receipt byte accounting changed" unless context.readbacks.fetch("terminal", 0_i64) == reads_before.fetch("terminal", 0_i64) + expected_bytes
      report = {event: "drama3_gdn_post_pair", signature: records.first.signature.to_a,
                pair: pair_index, warmup: pair_index < warmups, repeats_per_arm: repeats,
                order: order, command_indices: arms.transform_values(&.command_index),
                host_encode_seconds: arms.transform_values(&.host_encode_seconds),
                candidate_fused_dispatches: arms.transform_values(&.fused_dispatches),
                records: records.size, values_per_arm: values, bit_mismatches: 0,
                nonfinite_values: 0, negative_control: control.try { counts[control.not_nil!, 10] },
                graph_before: before, graph_after: after, readbacks_before: reads_before,
                readbacks_after: context.readbacks,
                gpu_timing_source: "existing drama3_gpu_command stderr events; correlate index with command_indices"}
      JSON.parse(report.to_json)
    end

    def self.run_pairs(context : Context, capture : Capture, warmups : Int32, pairs : Int32,
                       repeats : Int32) : JSON::Any
      capture.active = false
      groups = capture.records.group_by(&.signature).to_a.sort_by { |entry| entry[0].to_a.join(":") }
      raise "no homogeneous GDN record group" if groups.empty?
      # Upload the controls before timeline measurement; the first pair in each
      # signature group must prove that the receipt rejects a seeded defect.
      groups_json = [] of JSON::Any
      total_values = 0_i64
      context.finish
      context.enable_gdn_post_timeline!
      groups.each do |signature, records|
        raise "GDN homogeneous replay signature changed" unless records.all? { |record| record.signature == signature }
        start_graph, start_reads = context.graph_stats.dup, context.readbacks.dup
        group_pairs = [] of JSON::Any
        (warmups + pairs).times do |index|
          group_pairs << pair(context, records, index, warmups, repeats, index == 0)
          total_values += records.sum(0_i64) { |record| record.candidate.count.to_i64 } * repeats
        end
        groups_json << JSON.parse({signature: signature.to_a, records: records.size,
                                   warmups: warmups, measured_pairs: pairs, repeats_per_arm: repeats,
                                   pairs: group_pairs, graph_before: start_graph, graph_after: context.graph_stats,
                                   readbacks_before: start_reads, readbacks_after: context.readbacks}.to_json)
      end
      JSON.parse({groups: groups_json, verified_values_per_arm: total_values,
                  pair_order_cycle: [["unfused", "candidate"], ["candidate", "unfused"],
                                     ["candidate", "unfused"], ["unfused", "candidate"]],
                  warmups: warmups, measured_pairs: pairs, repeats_per_arm: repeats,
                  graph: context.graph_stats, readbacks: context.readbacks, activation_readbacks: 0}.to_json)
    end
  end

  module ML::Audio::Drama3::Resident
    class Context
      property gdn_post_benchmark_capture : Drama3GDNPostBenchmarkSupport::Capture?

      def enable_gdn_post_timeline! : Nil
        raise "GDN timing transition requires a drained queue" unless graph_stats["pending"] == 0
        @timeline = true
      end

      def gdn_post(x : Buffer, z : Buffer, w : Buffer, eps : Float64, heads : Int32 = 1) : Buffer
        capture = @gdn_post_benchmark_capture
        return previous_def unless capture && capture.active
        capture.intercept(x, z, w, eps, heads) { previous_def(x, z, w, eps, heads) }
      end
    end
  end

  module ML::Audio::Drama3
    class BackboneResident
      property gdn_post_benchmark_capture : Drama3GDNPostBenchmarkSupport::Capture?

      private def linear_attention_ready(qkv : Resident::Buffer, z : Resident::Buffer, b : Resident::Buffer,
                                         a : Resident::Buffer, prefix : String, layer : Int32, position : Int32) : Resident::Buffer
        capture = @gdn_post_benchmark_capture
        started = false
        if capture
          capture.begin_layer(position, layer, prefix)
          started = true
        end
        previous_def(qkv, z, b, a, prefix, layer, position)
      ensure
        capture.try(&.end_layer(position, layer)) if started
      end

      private def weight(name : String) : Resident::Buffer
        value = previous_def(name)
        @gdn_post_benchmark_capture.try(&.arm_weight(name, value))
        value
      end
    end
  end
{% end %}
