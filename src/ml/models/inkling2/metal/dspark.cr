require "./dspark_weights"
require "./dspark_transformer"
require "./dspark_markov"
require "./math"

module ML::GGUF
  # Holds target features until the verifier has committed or rewound their
  # rows. The drafter's persistent context only receives committed features.
  class Inkling2DSpark
    WINDOW    = 1023
    WIDTH     = 2304
    private TAP_COUNT =    4

    private alias MathOps = Inkling2ResidentMath

    record Capture, start : Int32, rows : Int32, taps : StaticArray(Inkling2View, 4) do
      def copy(encoder : ML::Metal::ComputeEncoder, tap : Int32, source : Inkling2View,
               tile_start : Int32) : Nil
        MathOps.copy(encoder, source.slice((start - tile_start) * WIDTH), taps[tap], rows * WIDTH)
      end
    end

    @tap_buffer : ML::MetalBuffer? = nil
    @context_buffer : ML::MetalBuffer? = nil
    @ids_buffer : ML::MetalBuffer? = nil
    @scratch = Inkling2ResidentScratch.new
    @pending : Capture? = nil
    @context_start = 0
    @context_end = 0
    getter? active : Bool = true

    def self.open(path : String, target : Inkling2Weights, linear : Inkling2ResidentLinear, max_seq : Int32) : self
      weights = Inkling2DSparkWeights.new(path, File.join(File.dirname(path), "config.json"), target.config)
      transformer = Inkling2DSparkTransformer.new(weights)
      markov = Inkling2DSparkMarkov.new(weights, target, linear)
      new(weights, transformer, markov, Math.min(max_seq, WINDOW))
    rescue ex
      transformer.try(&.close)
      weights.try(&.close)
      raise ex
    end

    private def initialize(@weights : Inkling2DSparkWeights, @transformer : Inkling2DSparkTransformer,
                           @markov : Inkling2DSparkMarkov, @tap_capacity : Int32)
    end

    def reset(@active : Bool) : Nil
      @pending = nil
      @context_start = 0
      @context_end = 0
    end

    def close : Nil
      @tap_buffer.try(&.release)
      @context_buffer.try(&.release)
      @ids_buffer.try(&.release)
      @scratch.close
      @transformer.close
      @weights.close
    end

    def capture(encoder : ML::Metal::ComputeEncoder, tile_start : Int32, tile_rows : Int32,
                visible_from : Int32) : Capture?
      return unless @active

      # A long prefill can skip all rows before the final context window.
      start = Math.max(tile_start, visible_from)
      rows = tile_start + tile_rows - start
      if pending = @pending
        if pending.start + pending.rows <= start - WINDOW || pending.start + pending.rows <= visible_from
          @pending = nil
        else
          absorb_pending(encoder)
        end
      end
      return unless rows > 0

      stride = @tap_capacity.to_i64 * WIDTH * sizeof(Float32)
      buffer = @tap_buffer ||= ML::MetalBuffer.new(TAP_COUNT * stride, ML::StorageMode::Private)
      taps = StaticArray(Inkling2View, 4).new { |tap| Inkling2View.new(buffer, tap * stride) }
      @pending = Capture.new(start, rows, taps)
    end

    def rewind(position : Int32) : Nil
      return unless pending = @pending

      rows = position - pending.start
      unless rows.in?(0..pending.rows)
        raise ArgumentError.new("Inkling 2 DSpark rewind is outside the staged target rows")
      end
      @pending = rows > 0 ? Capture.new(pending.start, rows, pending.taps) : nil
    end

    def encode(encoder : ML::Metal::ComputeEncoder, anchor : Int32, embedding : Inkling2View,
               position : Int32, count : Int32) : Nil
      raise ArgumentError.new("Inkling 2 DSpark is inactive for this sequence") unless @active
      absorb_pending(encoder)
      if position <= 0 || @context_end != position || @context_start > Math.max(position - WINDOW, 0)
        raise "Inkling 2 DSpark context does not cover the committed target window"
      end

      @scratch.reset
      hidden = @transformer.encode_hidden(encoder, embedding, position, @scratch)
      ids = @ids_buffer ||= ML::MetalBuffer.new(3_i64 * sizeof(Int32))
      @markov.encode(encoder, anchor, hidden, context, Inkling2View.new(ids), count, @scratch)
    end

    def candidates(count : Int32) : Array(Int32)
      data = @ids_buffer.not_nil!.contents.as(Int32*)
      Array(Int32).new(count) { |row| data[row] }
    end

    private def context : Inkling2View
      buffer = @context_buffer ||= ML::MetalBuffer.new(WIDTH.to_i64 * sizeof(Float32), ML::StorageMode::Private)
      Inkling2View.new(buffer)
    end

    private def absorb_pending(encoder : ML::Metal::ComputeEncoder) : Nil
      return unless pending = @pending

      @scratch.reset
      @transformer.encode_absorb(encoder, pending.taps, pending.start, pending.rows, @scratch)
      @transformer.encode_mix(encoder, pending.taps, pending.rows - 1, 3, context)
      @context_start = pending.start if pending.start != @context_end
      @context_start = Math.max(@context_start, pending.start + pending.rows - WINDOW)
      @context_end = pending.start + pending.rows
      @pending = nil
    end
  end
end
