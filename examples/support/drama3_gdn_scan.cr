# Standalone diagnostic: nothing in production requires this file.
require "../../src/ml/audio/drama3/backbone_prefill"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  module Drama3GDNScan
    alias Context = ML::Audio::Drama3::Resident::Context
    alias Buffer = ML::Audio::Drama3::Resident::Buffer
    SOURCE = {{ read_file("#{__DIR__}/drama3_gdn_scan.metal") }}
    COMPARE_SOURCE = {{ read_file("#{__DIR__}/drama3_prefill_compare.metal") }}
    MODES = ["sequential", "serial_tile", "affine_scan"]

    record Inputs, q : Buffer, k : Buffer, v : Buffer, a : Buffer, b : Buffer,
      alog : Buffer, dt : Buffer, initial : Buffer, rows : Int32,
      value_heads : Int32, value_dim : Int32, key_heads : Int32, key_dim : Int32 do
      def key_width : Int32
        key_heads * key_dim
      end

      def value_width : Int32
        value_heads * value_dim
      end

      def state_count : Int32
        value_heads * key_dim * value_dim
      end

      def buffers : Array(Buffer)
        [q, k, v, a, b, alog, dt, initial]
      end

      def validate!(context : Context) : Nil
        raise ArgumentError.new("GDN probe dimensions") unless rows.in?(1..32) && key_dim.in?(1..128) &&
          value_dim.in?(1..128) && key_heads.in?(1..128) && value_heads.in?(1..128) && value_heads % key_heads == 0
        shapes = [rows * key_width, rows * key_width, rows * value_width,
                  rows * value_heads, rows * value_heads, value_heads, value_heads, state_count]
        raise ArgumentError.new("GDN probe shape") unless buffers.map(&.count) == shapes
        # Validate against the actual context owner without encoding or allocation.
        raise ArgumentError.new("GDN probe foreign buffer") unless buffers.all? { |buffer| context.gdn_scan_owns?(buffer) }
      end
    end

    class Plan
      getter inputs : Inputs, block_size : Int32, blocks : Int32, prefix_states : Buffer
      @outputs = {} of String => Buffer
      @states = {} of String => Buffer
      @summary_a : Buffer
      @summary_b : Buffer

      def initialize(@context : Context, @inputs : Inputs, @block_size : Int32)
        @inputs.validate!(@context)
        raise ArgumentError.new("GDN probe block size") unless @block_size.in?(1..@inputs.rows)
        @blocks = (@inputs.rows + @block_size - 1) // @block_size
        MODES.each do |mode|
          @outputs[mode] = @context.alloc(@inputs.rows * @inputs.value_width)
          @states[mode] = @context.alloc(@inputs.state_count)
        end
        summary_count = @blocks * @inputs.value_heads * @inputs.key_dim * (@inputs.key_dim + @inputs.value_dim)
        @summary_a = @context.alloc(summary_count)
        @summary_b = @context.alloc(summary_count)
        @prefix_states = @context.alloc(@blocks * @inputs.state_count)
      end

      def output(mode : String) : Buffer
        @outputs[mode]
      end

      def state(mode : String) : Buffer
        @states[mode]
      end

      def scratch_bytes : Int64
        (@summary_a.count.to_i64 + @summary_b.count + @prefix_states.count) * 4
      end

      def encode(mode : String) : Nil
        raise ArgumentError.new("GDN probe mode") unless MODES.includes?(mode)
        i = @inputs
        if mode == "sequential"
          @context.copy(i.initial, state(mode))
          i.rows.times do |row|
            @context.dispatch("dr3_delta_step_transposed", ML::Audio::Drama3::BackboneResident::SOURCE,
              [i.q.slice(row * i.key_width, i.key_width), i.k.slice(row * i.key_width, i.key_width),
               i.v.slice(row * i.value_width, i.value_width), i.a.slice(row * i.value_heads, i.value_heads),
               i.b.slice(row * i.value_heads, i.value_heads), i.alog, i.dt, state(mode),
               output(mode).slice(row * i.value_width, i.value_width)],
              [i.value_heads.to_u64, i.value_dim.to_u64, i.key_heads.to_u64, i.key_dim.to_u64], i.value_width)
          end
          return
        end

        if mode == "serial_tile"
          params = parameters(i.rows, 1)
          replay(i.initial, mode, params, 1)
          return
        end

        params = parameters(@block_size, @blocks)
        @context.dispatch("dgs_summary", SOURCE, [i.k, i.v, i.a, i.b, i.alog, i.dt, @summary_a],
          params, @blocks * i.value_heads * (i.key_dim + i.value_dim))
        source, target = @summary_a, @summary_b
        distance = 1
        while distance < @blocks
          @context.dispatch("dgs_compose", SOURCE, [source, target], params + [distance.to_u64], source.count)
          source, target = target, source
          distance *= 2
        end
        @context.dispatch("dgs_prefix_state", SOURCE, [i.initial, source, @prefix_states], params, @prefix_states.count)
        replay(@prefix_states, mode, params, @blocks)
      end

      private def parameters(block_size : Int32, blocks : Int32) : Array(UInt64)
        i = @inputs
        [i.rows.to_u64, i.value_heads.to_u64, i.value_dim.to_u64, i.key_heads.to_u64,
         i.key_dim.to_u64, block_size.to_u64, blocks.to_u64]
      end

      private def replay(initial : Buffer, mode : String, params : Array(UInt64), blocks : Int32) : Nil
        i = @inputs
        @context.dispatch("dgs_replay", SOURCE,
          [i.q, i.k, i.v, i.a, i.b, i.alog, i.dt, initial, output(mode), state(mode)],
          params, blocks * i.value_width)
      end
    end

    # Counts stay below 2^24 for the admitted shapes; the existing float receipt
    # therefore represents every integer count exactly. Nonfinite is fail-closed.
    def self.compare(context : Context, actual : Buffer, reference : Buffer) : Buffer
      raise ArgumentError.new("GDN probe comparison shape") unless actual.count == reference.count
      tiles = (actual.count + 1023) // 1024
      parts, result = context.alloc(tiles * 5), context.alloc(5)
      context.dispatch("dr3_prefill_compare", COMPARE_SOURCE, [actual, reference, parts],
        [actual.count.to_u64, 6e-5_f32.unsafe_as(UInt32).to_u64, 3e-4_f32.unsafe_as(UInt32).to_u64], tiles)
      context.dispatch("dr3_prefill_compare_reduce", COMPARE_SOURCE, [parts, result], [tiles.to_u64], 1, 1)
      result
    end
  end

  module ML::Audio::Drama3::Resident
    class Context
      def gdn_scan_owns?(value : Buffer) : Bool
        value.owner.same?(@owner)
      end

      def gdn_scan_timeline! : Nil
        raise "GDN probe timing requires drained queue" unless graph_stats["pending"] == 0
        @timeline = true
      end
    end
  end
{% end %}
