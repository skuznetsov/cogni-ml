require "./engine"
require "../../gguf/ngram_draft"

module ML::GGUF
  module Inkling2GreedyDecode
    extend self

    MIN_DSPARK_COMPLETION =   8
    private DSPARK_SCALAR_STEPS   =   2
    private DSPARK_PROBE_BUDGET   = 2.0
    private DSPARK_SAVINGS_LIMIT  = 2.0

    record Result, ids : Array(Int32), blocks : Int32, accepted_rows : Int32

    def generate(engine : Inkling2Engine, prefix : Array(Int32), first_id : Int32, limit : Int32,
                 eos_id : Int32, *, ngram : Bool, depth_speculation : Bool = false, dspark : Bool = false) : Result
      generate(engine, prefix, first_id, limit, eos_id, ngram: ngram, depth_speculation: depth_speculation, dspark: dspark) { true }
    end

    # Emit the target's accepted IDs in order. If the consumer stops mid-block,
    # the sequence may already hold the rest of that block. Reset before reuse.
    def generate(engine : Inkling2Engine, prefix : Array(Int32), first_id : Int32, limit : Int32,
                 eos_id : Int32, *, ngram : Bool, depth_speculation : Bool = false, dspark : Bool = false,
                 & : Int32 -> Bool) : Result
      if dspark
        if ngram || depth_speculation
          raise ArgumentError.new("Inkling 2 DSpark cannot be combined with another greedy drafter")
        end
        raise ArgumentError.new("Inkling 2 DSpark was not enabled") unless engine.dspark?
      end
      if depth_speculation
        raise ArgumentError.new("Inkling 2 cannot combine depth speculation and n-gram verification") if ngram
        raise ArgumentError.new("Inkling 2 depth speculation was not enabled") unless engine.depth_speculation?
      end
      history = NgramDraft::IndexedHistory.new(prefix, max_ngram: 8, min_ngram: 6) if ngram && limit > 4
      ids = [] of Int32
      blocks = 0
      accepted_rows = 0
      pending = first_id
      dspark_scalar_steps = 0
      dspark_scalar_ms = 0.0
      dspark_debt_ms = 0.0

      while ids.size < limit
        ids << pending
        break if !yield(pending) || pending == eos_id || ids.size == limit

        # Keep space for the exact pending ID after the four evaluated rows.
        if depth_speculation && limit - ids.size >= 4 && engine.max_seq - engine.position >= 4
          block, accepted, pending = engine.verify_depth_block4(pending, eos_id: eos_id)
          blocks += 1
          accepted_rows += accepted
          (1...accepted).each do |row|
            ids << block[row]
            return Result.new(ids, blocks, accepted_rows) unless yield(block[row])
          end
          next
        end

        if dspark
          width = Math.min(4, Math.min(limit - ids.size, engine.max_seq - engine.position))
          if width >= 3 && dspark_scalar_steps < DSPARK_SCALAR_STEPS
            # The first scalar step can absorb the prompt and warm pipelines.
            # Time the next step as the local cost of advancing one row.
            started = Time.instant
            pending = engine.forward_greedy(pending)
            dspark_scalar_steps += 1
            dspark_scalar_ms = (Time.instant - started).total_milliseconds
            next
          elsif width >= 3
            started = Time.instant
            block, accepted, pending = engine.verify_dspark(pending, width, eos_id: eos_id)
            elapsed = (Time.instant - started).total_milliseconds
            blocks += 1
            accepted_rows += accepted
            (1...accepted).each do |row|
              ids << block[row]
              return Result.new(ids, blocks, accepted_rows) unless yield(block[row])
            end
            # Charge discarded work and retain up to two scalar steps of
            # savings so a brief dip in acceptance can spend prior gains.
            dspark_debt_ms = Math.max(-DSPARK_SAVINGS_LIMIT * dspark_scalar_ms,
              dspark_debt_ms + elapsed - accepted * dspark_scalar_ms)
            if dspark_debt_ms > DSPARK_PROBE_BUDGET * dspark_scalar_ms
              dspark = false
              engine.disable_dspark
            end
            next
          end
          dspark = false
          engine.disable_dspark
        end

        if index = history
          index.append(pending)
          # Keep space for the exact next ID after the four evaluated rows.
          if limit - ids.size >= 4 && engine.max_seq - engine.position >= 4
            candidates = index.candidates(3, min_candidates: 3)
            unless candidates.empty? || candidates.includes?(eos_id)
              block = StaticArray[pending, candidates[0], candidates[1], candidates[2]]
              accepted, pending = engine.verify_greedy_block4(block)
              blocks += 1
              accepted_rows += accepted
              (1...accepted).each do |row|
                ids << block[row]
                return Result.new(ids, blocks, accepted_rows) unless yield(block[row])
                index.append(block[row])
              end
              # The M2 Max timings put break-even above two committed rows.
              # Stop drafting for this completion after such a block.
              history = nil if accepted < 3
              next
            end
          end
        end

        pending = engine.forward_greedy(pending)
      end

      Result.new(ids, blocks, accepted_rows)
    end
  end
end
