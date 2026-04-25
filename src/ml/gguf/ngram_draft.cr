module ML::GGUF
  module NgramDraft
    extend self

    def candidates(history : Array(Int32),
                   gamma : Int32,
                   max_ngram : Int32,
                   min_ngram : Int32) : Array(Int32)
      raise ArgumentError.new("gamma must be positive") unless gamma > 0
      raise ArgumentError.new("min_ngram must be positive") unless min_ngram > 0
      raise ArgumentError.new("max_ngram must be >= min_ngram") unless max_ngram >= min_ngram
      return [] of Int32 if history.empty?

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
