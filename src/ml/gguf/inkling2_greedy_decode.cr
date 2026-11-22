require "./inkling2_engine"
require "./ngram_draft"

module ML::GGUF
  module Inkling2GreedyDecode
    extend self

    record Result, ids : Array(Int32), blocks : Int32, accepted_rows : Int32

    def generate(engine : Inkling2Engine, prefix : Array(Int32), first_id : Int32, limit : Int32,
                 eos_id : Int32, *, ngram : Bool, depth_speculation : Bool = false) : Result
      if depth_speculation
        raise ArgumentError.new("Inkling 2 cannot combine depth speculation and n-gram verification") if ngram
        raise ArgumentError.new("Inkling 2 depth speculation was not enabled") unless engine.depth_speculation?
      end
      history = NgramDraft::IndexedHistory.new(prefix, max_ngram: 8, min_ngram: 6) if ngram
      ids = [] of Int32
      blocks = 0
      accepted_rows = 0
      pending = first_id

      while ids.size < limit
        ids << pending
        break if pending == eos_id || ids.size == limit

        # Keep space for the exact pending ID after the four evaluated rows.
        if depth_speculation && limit - ids.size >= 4 && engine.max_seq - engine.position >= 4
          block, accepted, pending = engine.verify_depth_block4(pending, eos_id: eos_id)
          blocks += 1
          accepted_rows += accepted
          (1...accepted).each { |row| ids << block[row] }
          next
        end

        if index = history
          index.append(pending)
          # Keep space for the exact next ID after the four evaluated rows.
          if limit - ids.size >= 4
            candidates = index.candidates(3, min_candidates: 3)
            unless candidates.empty? || candidates.includes?(eos_id)
              block = StaticArray[pending, candidates[0], candidates[1], candidates[2]]
              accepted, pending = engine.verify_greedy_block4(block)
              blocks += 1
              accepted_rows += accepted
              (1...accepted).each do |row|
                ids << block[row]
                index.append(block[row])
              end
              # The M2 Max timings put break-even above two committed rows.
              # Stop drafting for this completion after such a block.
              history = nil if accepted < 3
              next
            end
          end
        end

        pending = Inkling2Engine.greedy_token_id(engine.forward(pending).not_nil!)
      end

      Result.new(ids, blocks, accepted_rows)
    end
  end
end
