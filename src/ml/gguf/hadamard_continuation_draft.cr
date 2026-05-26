module ML::GGUF
  # Draft-only near-repeat continuation search.
  #
  # This is intentionally not an exact n-gram matcher. It sketches a fixed-size
  # token window with a deterministic feature-hash + FWHT sign sketch, then
  # proposes the continuation after the nearest prior window. The target model
  # must still verify every proposed token exactly.
  module HadamardContinuationDraft
    extend self

    record CandidateSpan,
      ids : Array(Int32),
      source_start : Int32,
      window_size : Int32,
      hamming : Int32

    class IndexedHistory
      getter history : Array(Int32)
      getter window_size : Int32
      getter sketch_bits : Int32
      getter vector_dim : Int32

      def initialize(history : Array(Int32),
                     @window_size : Int32 = 8,
                     @sketch_bits : Int32 = 64,
                     @vector_dim : Int32 = 64,
                     @seed : UInt64 = 0x9e3779b97f4a7c15_u64)
        raise ArgumentError.new("window_size must be positive") unless @window_size > 0
        raise ArgumentError.new("sketch_bits must be positive") unless @sketch_bits > 0
        raise ArgumentError.new("sketch_bits must be <= 64") unless @sketch_bits <= 64
        raise ArgumentError.new("vector_dim must be a positive power of two") unless HadamardContinuationDraft.positive_power_of_two?(@vector_dim)

        @history = [] of Int32
        @sketches = [] of UInt64
        append(history)
      end

      def append(ids : Array(Int32)) : Nil
        ids.each { |id| append(id) }
      end

      def append(id : Int32) : Nil
        @history << id
        rebuild_sketches
      end

      def candidate_span(gamma : Int32,
                         min_candidates : Int32 = 0,
                         max_hamming : Int32 = 16) : CandidateSpan?
        raise ArgumentError.new("gamma must be positive") unless gamma > 0
        raise ArgumentError.new("min_candidates must be non-negative") unless min_candidates >= 0
        raise ArgumentError.new("max_hamming must be non-negative") unless max_hamming >= 0
        return nil if @history.size < @window_size * 2 + 1

        current_start = @history.size - @window_size
        current = sketch_at(current_start)

        best_start = -1
        best_dist = @sketch_bits + 1
        # Latest compatible prior window wins ties; this mirrors cache locality.
        (current_start - 1).downto(0) do |start|
          continuation_start = start + @window_size
          next if continuation_start >= @history.size

          dist = HadamardContinuationDraft.hamming_distance(current, sketch_at(start))
          next if dist > max_hamming
          next if dist > best_dist

          best_dist = dist
          best_start = start
        end

        return nil if best_start < 0

        continuation_start = best_start + @window_size
        ids = [] of Int32
        i = continuation_start
        while i < @history.size && ids.size < gamma
          ids << @history[i]
          i += 1
        end
        return nil if min_candidates > 0 && ids.size < min_candidates
        return nil if ids.empty?

        CandidateSpan.new(ids, best_start, @window_size, best_dist)
      end

      def candidates(gamma : Int32,
                     min_candidates : Int32 = 0,
                     max_hamming : Int32 = 16) : Array(Int32)
        candidate_span(gamma, min_candidates: min_candidates, max_hamming: max_hamming).try(&.ids) || [] of Int32
      end

      private def rebuild_sketches : Nil
        @sketches.clear
        return if @history.size < @window_size

        last_start = @history.size - @window_size
        0.upto(last_start) do |start|
          @sketches << build_sketch(start)
        end
      end

      private def sketch_at(start : Int32) : UInt64
        @sketches[start]
      end

      private def build_sketch(start : Int32) : UInt64
        v = Array(Float64).new(@vector_dim, 0.0)
        @window_size.times do |offset|
          token = @history[start + offset]
          h = HadamardContinuationDraft.mix(token.to_i64.to_u64 &+ @seed &+ offset.to_u64 &* 0x9e3779b97f4a7c15_u64)
          idx = (h % @vector_dim.to_u64).to_i
          sign = ((h >> 63) & 1_u64) == 0_u64 ? 1.0 : -1.0
          # Position weighting makes delimiter/order matches survive value drift.
          v[idx] += sign * (1.0 + (offset % 3).to_f64 * 0.25)
        end

        HadamardContinuationDraft.fwht!(v)

        bits = 0_u64
        @sketch_bits.times do |i|
          bits |= (1_u64 << i) if v[i % @vector_dim] >= 0.0
        end
        bits
      end
    end

    def candidates(history : Array(Int32),
                   gamma : Int32,
                   window_size : Int32 = 8,
                   sketch_bits : Int32 = 64,
                   vector_dim : Int32 = 64,
                   min_candidates : Int32 = 0,
                   max_hamming : Int32 = 16) : Array(Int32)
      IndexedHistory.new(history, window_size: window_size, sketch_bits: sketch_bits, vector_dim: vector_dim)
        .candidates(gamma, min_candidates: min_candidates, max_hamming: max_hamming)
    end

    protected def positive_power_of_two?(n : Int32) : Bool
      n > 0 && (n & (n - 1)) == 0
    end

    protected def fwht!(values : Array(Float64)) : Nil
      h = 1
      n = values.size
      while h < n
        step = h * 2
        i = 0
        while i < n
          h.times do |j|
            x = values[i + j]
            y = values[i + j + h]
            values[i + j] = x + y
            values[i + j + h] = x - y
          end
          i += step
        end
        h = step
      end
    end

    protected def mix(x : UInt64) : UInt64
      z = x
      z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9_u64
      z = (z ^ (z >> 27)) &* 0x94d049bb133111eb_u64
      z ^ (z >> 31)
    end

    protected def hamming_distance(a : UInt64, b : UInt64) : Int32
      x = a ^ b
      count = 0
      while x != 0_u64
        x &= x - 1
        count += 1
      end
      count
    end
  end
end
