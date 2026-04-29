require "set"

module ML::GGUF
  module NgramDraft
    extend self

    def candidates(history : Array(Int32),
                   gamma : Int32,
                   max_ngram : Int32,
                   min_ngram : Int32,
                   recursive : Bool = false) : Array(Int32)
      raise ArgumentError.new("gamma must be positive") unless gamma > 0
      raise ArgumentError.new("min_ngram must be positive") unless min_ngram > 0
      raise ArgumentError.new("max_ngram must be >= min_ngram") unless max_ngram >= min_ngram
      return [] of Int32 if history.empty?

      return candidates_once(history, gamma, max_ngram, min_ngram) unless recursive

      first = candidates_once(history, gamma, max_ngram, min_ngram)
      return first if first.empty? || first.size >= gamma

      scratch = history.dup
      scratch.concat(first)
      result = first
      while result.size < gamma
        chunk = candidates_once(scratch, gamma - result.size, max_ngram, min_ngram)
        break if chunk.empty?
        result.concat(chunk)
        scratch.concat(chunk)
      end
      result
    end

    def risky_candidate_shape?(ids : Array(Int32), min_size : Int32 = 16) : Bool
      return false if ids.size < min_size

      period = exact_period(ids, 8)
      return true if period == 8

      pair_unique_ratio(ids) > 0.90 && lag_ratio(ids, 4) < 0.05 && lag_ratio(ids, 8) < 0.20
    end

    def exact_period(ids : Array(Int32), max_period : Int32) : Int32
      return 0 if ids.empty?

      1.upto(Math.min(max_period, ids.size)) do |period|
        exact = true
        period.upto(ids.size - 1) do |i|
          if ids[i] != ids[i % period]
            exact = false
            break
          end
        end
        return period if exact
      end
      0
    end

    def lag_ratio(ids : Array(Int32), lag : Int32) : Float64
      return 0.0 if ids.size <= lag

      matches = 0
      lag.upto(ids.size - 1) do |i|
        matches += 1 if ids[i] == ids[i - lag]
      end
      matches.to_f / (ids.size - lag)
    end

    def pair_unique_ratio(ids : Array(Int32)) : Float64
      return 0.0 if ids.size < 2

      pairs = Set(Tuple(Int32, Int32)).new
      0.upto(ids.size - 2) { |i| pairs << {ids[i], ids[i + 1]} }
      pairs.size.to_f / (ids.size - 1)
    end

    private def candidates_once(history : Array(Int32),
                                gamma : Int32,
                                max_ngram : Int32,
                                min_ngram : Int32) : Array(Int32)
      max_len = Math.min(max_ngram, history.size)
      max_len.downto(min_ngram) do |n|
        suffix_start = history.size - n
        i = history.size - n - 1
        while i >= 0
          matched = true
          n.times do |j|
            if history[i + j] != history[suffix_start + j]
              matched = false
              break
            end
          end

          if matched && i + n < history.size
            result = [] of Int32
            k = i + n
            while k < history.size && result.size < gamma
              result << history[k]
              k += 1
            end
            return result unless result.empty?
          end
          i -= 1
        end
      end

      [] of Int32
    end
  end
end
