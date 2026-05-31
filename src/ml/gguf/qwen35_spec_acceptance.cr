module ML::GGUF
  # Pure token-level acceptance scan shared by exact speculative decoders.
  # It does not mutate model state; callers remain responsible for verifier
  # prefill/replay and draft resync after a rejection.
  module Qwen35SpecAcceptance
    extend self

    record Result,
      emitted : Array(Int32),
      accepted : Int32,
      rejected : Bool,
      next_expected : Int32,
      reject_index : Int32? do
      def full_accept? : Bool
        !@rejected
      end
    end

    def scan(candidates : Array(Int32),
             initial_expected : Int32,
             target_nexts : Array(Tuple(Int32, Float32)),
             max_output : Int32,
             eos_id : Int32? = nil) : Result
      emitted = [] of Int32
      accepted = 0
      expected = initial_expected
      reject_index = nil.as(Int32?)

      candidates.each_with_index do |cand, i|
        break if emitted.size >= max_output
        if cand == expected
          emitted << cand
          accepted += 1
          expected = target_nexts[i][0] if i < target_nexts.size
          break if eos_id && cand == eos_id
        else
          emitted << expected
          reject_index = i
          break
        end
      end

      Result.new(emitted, accepted, !reject_index.nil?, expected, reject_index)
    end
  end
end
