require "set"

module ML::GGUF
  module NgramDraft
    extend self

    record CandidateSpan,
      ids : Array(Int32),
      match_len : Int32,
      source_start : Int32

    record ReplayScheduleResult,
      chunks : Array(Int32),
      full_accept_chunks : Int32,
      verified_tokens : Int32,
      committed_tokens : Int32,
      discarded_accept_prefix : Int32,
      reject_index : Int32,
      full_accept : Bool

    class IndexedHistory
      getter history : Array(Int32)
      getter min_ngram : Int32
      getter max_ngram : Int32

      def initialize(history : Array(Int32),
                     @max_ngram : Int32,
                     @min_ngram : Int32)
        raise ArgumentError.new("min_ngram must be positive") unless @min_ngram > 0
        raise ArgumentError.new("max_ngram must be >= min_ngram") unless @max_ngram >= @min_ngram

        @history = [] of Int32
        @positions = Hash(String, Array(Int32)).new { |hash, key| hash[key] = [] of Int32 }
        append(history)
      end

      def append(ids : Array(Int32)) : Nil
        ids.each { |id| append(id) }
      end

      def append(id : Int32) : Nil
        @history << id
        index_suffixes_ending_at(@history.size - 1)
      end

      def candidates(gamma : Int32,
                     recursive : Bool = false,
                     min_candidates : Int32 = 0) : Array(Int32)
        raise ArgumentError.new("gamma must be positive") unless gamma > 0
        raise ArgumentError.new("min_candidates must be non-negative") unless min_candidates >= 0
        return [] of Int32 if @history.empty?

        result = if recursive
                   first = candidates_once(gamma)
                   if first.empty? || first.size >= gamma
                     first
                   else
                     scratch = fork
                     scratch.append(first)
                     expanded = first
                     while expanded.size < gamma
                       chunk = scratch.candidates_once(gamma - expanded.size)
                       break if chunk.empty?

                       expanded.concat(chunk)
                       scratch.append(chunk)
                     end
                     expanded
                   end
                 else
                   candidates_once(gamma)
                 end

        if min_candidates > 0 && result.size < min_candidates
          [] of Int32
        else
          result
        end
      end

      def candidate_span(gamma : Int32,
                         recursive : Bool = false,
                         min_candidates : Int32 = 0) : CandidateSpan?
        raise ArgumentError.new("gamma must be positive") unless gamma > 0
        raise ArgumentError.new("min_candidates must be non-negative") unless min_candidates >= 0
        return nil if @history.empty?

        first = candidates_once_span(gamma)
        return nil unless first

        ids = first.ids
        if recursive && ids.size < gamma
          scratch = fork
          scratch.append(ids)
          expanded = ids.dup
          while expanded.size < gamma
            chunk = scratch.candidates_once(gamma - expanded.size)
            break if chunk.empty?

            expanded.concat(chunk)
            scratch.append(chunk)
          end
          ids = expanded
        end

        return nil if min_candidates > 0 && ids.size < min_candidates

        CandidateSpan.new(ids, first.match_len, first.source_start)
      end

      def match_len : Int32
        return 0 if @history.empty?

        max_len = Math.min(@max_ngram, @history.size)
        max_len.downto(@min_ngram) do |n|
          return n if latest_prior_match_start(n)
        end
        0
      end

      def fork : IndexedHistory
        copy = IndexedHistory.allocate
        copy.initialize_copy(@history, @max_ngram, @min_ngram, @positions)
        copy
      end

      protected def initialize_copy(history : Array(Int32),
                                    @max_ngram : Int32,
                                    @min_ngram : Int32,
                                    positions : Hash(String, Array(Int32))) : Nil
        @history = history.dup
        @positions = Hash(String, Array(Int32)).new { |hash, key| hash[key] = [] of Int32 }
        positions.each do |key, values|
          @positions[key] = values.dup
        end
      end

      protected def candidates_once(gamma : Int32) : Array(Int32)
        candidates_once_span(gamma).try(&.ids) || [] of Int32
      end

      protected def candidates_once_span(gamma : Int32) : CandidateSpan?
        max_len = Math.min(@max_ngram, @history.size)
        max_len.downto(@min_ngram) do |n|
          if start = latest_prior_match_start(n)
            result = [] of Int32
            k = start + n
            while k < @history.size && result.size < gamma
              result << @history[k]
              k += 1
            end
            return CandidateSpan.new(result, n, start) unless result.empty?
          end
        end

        nil
      end

      private def latest_prior_match_start(n : Int32) : Int32?
        suffix_start = @history.size - n
        return nil if suffix_start <= 0

        positions = @positions[key_at(suffix_start, n)]?
        return nil unless positions

        positions.reverse_each do |start|
          return start if start < suffix_start && start + n < @history.size
        end
        nil
      end

      private def index_suffixes_ending_at(index : Int32) : Nil
        1.upto(@max_ngram) do |n|
          next if n < @min_ngram

          start = index - n + 1
          next if start < 0

          @positions[key_at(start, n)] << start
        end
      end

      private def key_at(start : Int32, n : Int32) : String
        io = IO::Memory.new(n * 4)
        n.times do |j|
          io.write_bytes(@history[start + j], IO::ByteFormat::LittleEndian)
        end
        io.to_s
      end
    end

    def candidates(history : Array(Int32),
                   gamma : Int32,
                   max_ngram : Int32,
                   min_ngram : Int32,
                   recursive : Bool = false,
                   min_candidates : Int32 = 0) : Array(Int32)
      raise ArgumentError.new("gamma must be positive") unless gamma > 0
      raise ArgumentError.new("min_ngram must be positive") unless min_ngram > 0
      raise ArgumentError.new("max_ngram must be >= min_ngram") unless max_ngram >= min_ngram
      raise ArgumentError.new("min_candidates must be non-negative") unless min_candidates >= 0
      return [] of Int32 if history.empty?

      result = if recursive
                 first = candidates_once(history, gamma, max_ngram, min_ngram)
                 if first.empty? || first.size >= gamma
                   first
                 else
                   scratch = history.dup
                   scratch.concat(first)
                   expanded = first
                   while expanded.size < gamma
                     chunk = candidates_once(scratch, gamma - expanded.size, max_ngram, min_ngram)
                     break if chunk.empty?
                     expanded.concat(chunk)
                     scratch.concat(chunk)
                   end
                   expanded
                 end
               else
                 candidates_once(history, gamma, max_ngram, min_ngram)
               end

      if min_candidates > 0 && result.size < min_candidates
        [] of Int32
      else
        result
      end
    end

    def fixed_split_acceptance(expected : Array(Int32),
                               actual : Array(Int32),
                               split_size : Int32) : ReplayScheduleResult
      raise ArgumentError.new("split_size must be positive") unless split_size > 0

      schedule_acceptance(expected, actual, [split_size])
    end

    def schedule_acceptance(expected : Array(Int32),
                            actual : Array(Int32),
                            schedule : Array(Int32)) : ReplayScheduleResult
      raise ArgumentError.new("schedule must not be empty") if schedule.empty?
      schedule.each { |size| raise ArgumentError.new("schedule chunk sizes must be positive") unless size > 0 }

      chunks = [] of Int32
      full_accept_chunks = 0
      verified_tokens = 0
      committed_tokens = 0
      discarded_accept_prefix = 0
      reject_index = -1
      pos = 0

      while pos < expected.size
        schedule_idx = Math.min(full_accept_chunks, schedule.size - 1)
        chunk_size = Math.min(schedule[schedule_idx], expected.size - pos)
        chunks << chunk_size
        verified_tokens += chunk_size

        local_accept = 0
        chunk_size.times do |j|
          row = pos + j
          if actual[row]? == expected[row]
            local_accept += 1
          else
            reject_index = row
            break
          end
        end

        if local_accept == chunk_size
          full_accept_chunks += 1
          committed_tokens += chunk_size
          pos += chunk_size
        else
          discarded_accept_prefix = local_accept
          break
        end
      end

      ReplayScheduleResult.new(
        chunks: chunks,
        full_accept_chunks: full_accept_chunks,
        verified_tokens: verified_tokens,
        committed_tokens: committed_tokens,
        discarded_accept_prefix: discarded_accept_prefix,
        reject_index: reject_index,
        full_accept: committed_tokens == expected.size)
    end

    def risky_candidate_shape?(ids : Array(Int32), min_size : Int32 = 16, match_len : Int32 = 0) : Bool
      return false if ids.size < Math.min(min_size, 8)

      period = exact_period(ids, 8)
      return true if ids.size >= min_size && period == 8
      if ids.size >= 8 && period == 8
        # Short period-8 tails are risky only when the repeated suffix is long
        # enough. This preserves compact table-like repeats where the candidate
        # is exact and cheap, while still catching YAML/JSON tails that overrun.
        return true if match_len >= 5 && pair_unique_ratio(ids) > 0.90 && unique_ratio(ids) < 0.95
      end

      # Medium chunks are economical enough to try only when their continuation
      # already repeats strongly across the verifier chunk. This keeps useful
      # prompt-echo/fact chunks while rejecting code/math/template tails that
      # otherwise pay an expensive bulk verify for an early reject.
      if ids.size > 8 && ids.size < min_size && match_len < 5
        return true if lag_ratio(ids, 8) < 0.75
      end

      if ids.size >= 12 && ids.size < min_size && match_len >= 8
        return true if pair_unique_ratio(ids) >= 0.95 &&
                       entropy_norm(ids) >= 0.80 &&
                       lag_ratio(ids, 4) < 0.10 &&
                       lag_ratio(ids, 8) < 0.10
      end

      prefix_run = prefix_period_run(ids, 4)
      return true if prefix_run >= 6 && prefix_run < ids.size && exact_period(ids, 4) == 0

      return false if ids.size < min_size

      pair_unique_ratio(ids) > 0.90 && lag_ratio(ids, 4) < 0.20 && lag_ratio(ids, 8) < 0.20
    end

    def match_len(history : Array(Int32), max_ngram : Int32, min_ngram : Int32) : Int32
      return 0 if history.empty?

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
          return n if matched && i + n < history.size
          i -= 1
        end
      end
      0
    end

    def unique_ratio(ids : Array(Int32)) : Float64
      return 0.0 if ids.empty?

      ids.to_set.size.to_f / ids.size
    end

    def entropy_norm(ids : Array(Int32)) : Float64
      return 0.0 if ids.empty?

      counts = Hash(Int32, Int32).new(0)
      ids.each { |id| counts[id] += 1 }

      entropy = 0.0
      counts.each_value do |count|
        p = count.to_f / ids.size
        entropy -= p * (Math.log(p) / Math.log(2.0))
      end

      max_entropy = ids.size > 1 ? Math.log(ids.size.to_f) / Math.log(2.0) : 1.0
      max_entropy > 0.0 ? entropy / max_entropy : 0.0
    end

    def corridor_candidate_shape?(ids : Array(Int32),
                                  match_len : Int32 = 0,
                                  min_size : Int32 = 4,
                                  match_len_min : Int32 = 0,
                                  lag4_min : Float64 = 0.25,
                                  lag8_min : Float64 = 0.5,
                                  entropy_max : Float64 = 0.6) : Bool
      return false if ids.size < min_size
      return true if match_len_min > 0 && match_len >= match_len_min

      lag_ratio(ids, 4) >= lag4_min ||
        lag_ratio(ids, 8) >= lag8_min ||
        entropy_norm(ids) <= entropy_max
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

    def prefix_period_run(ids : Array(Int32), max_period : Int32) : Int32
      return 0 if ids.empty?

      best = 0
      1.upto(Math.min(max_period, ids.size)) do |period|
        run = period
        period.upto(ids.size - 1) do |i|
          break unless ids[i] == ids[i - period]

          run += 1
        end
        best = Math.max(best, run) if run >= period * 2
      end
      best
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
