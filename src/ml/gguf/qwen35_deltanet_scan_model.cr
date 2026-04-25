module ML
  module GGUF
    module Qwen35DeltaNetScanModel
      record Estimate,
        prompt_tokens : Int32,
        block_size : Int32,
        compose_token_equiv : Float64,
        serial_depth : Float64,
        parallel_depth : Float64,
        speedup : Float64,
        scan_levels : Int32,
        max_compose_for_target : Float64

      record RankGrowthEstimate,
        prompt_tokens : Int32,
        block_size : Int32,
        state_size : Int32,
        rank_cap : Int32?,
        serial_depth : Float64,
        parallel_depth : Float64,
        compose_depth : Float64,
        speedup : Float64,
        scan_levels : Int32,
        max_rank_on_path : Int32

      # A serial DeltaNet chunk has a true dependency chain of one recurrent
      # update per token. A block-scan path keeps serial work inside each block,
      # then pays a prefix-scan over block summaries, then replays each block
      # from its scanned prefix state. This is a critical-path model, not a
      # total-work or memory-bandwidth proof.
      def self.estimate(prompt_tokens : Int32,
                        block_size : Int32,
                        compose_token_equiv : Float64,
                        target_speedup : Float64 = 1.25) : Estimate
        raise ArgumentError.new("prompt_tokens must be positive") unless prompt_tokens > 0
        raise ArgumentError.new("block_size must be positive") unless block_size > 0
        raise ArgumentError.new("compose_token_equiv must be non-negative") unless compose_token_equiv >= 0.0
        raise ArgumentError.new("target_speedup must be positive") unless target_speedup > 0.0

        n_blocks = ceil_div(prompt_tokens, block_size)
        levels = ceil_log2(n_blocks)
        serial = prompt_tokens.to_f64

        # Build summaries and replay blocks are both block-local serial scans.
        parallel = (2 * block_size).to_f64 + levels.to_f64 * compose_token_equiv
        max_compose =
          if levels == 0
            Float64::INFINITY
          else
            (serial / target_speedup - (2 * block_size).to_f64) / levels.to_f64
          end

        Estimate.new(
          prompt_tokens: prompt_tokens,
          block_size: block_size,
          compose_token_equiv: compose_token_equiv,
          serial_depth: serial,
          parallel_depth: parallel,
          speedup: serial / parallel,
          scan_levels: levels,
          max_compose_for_target: max_compose,
        )
      end

      # Naive dense composition of block summaries has matrix-matrix style cost.
      # Compared with one serial DeltaNet rowwise step (~3*s*s row work), a
      # dense A/B summary composition is O(s) token-steps. This is intentionally
      # a rough adversary estimate; it should reject dense summaries before Metal
      # implementation work starts.
      def self.naive_dense_compose_token_equiv(s : Int32 = 128) : Float64
        raise ArgumentError.new("s must be positive") unless s > 0

        4.0 * s.to_f64 / 3.0
      end

      # Critical-path model for a compact prefix scan where composing two
      # summaries of rank r1/r2 costs O(r1*r2*s). Without exact rank
      # compression, rank doubles at every tree level and this quickly
      # dominates the serial DeltaNet scan. With rank_cap=s, this is an
      # optimistic model for an exact basis-compression path; it does not
      # include the compression kernel cost.
      def self.rank_growth_estimate(prompt_tokens : Int32,
                                    block_size : Int32,
                                    state_size : Int32 = 128,
                                    rank_cap : Int32? = nil) : RankGrowthEstimate
        raise ArgumentError.new("prompt_tokens must be positive") unless prompt_tokens > 0
        raise ArgumentError.new("block_size must be positive") unless block_size > 0
        raise ArgumentError.new("state_size must be positive") unless state_size > 0
        raise ArgumentError.new("rank_cap must be positive") if rank_cap && rank_cap.not_nil! <= 0

        n_blocks = ceil_div(prompt_tokens, block_size)
        levels = ceil_log2(n_blocks)
        rank = block_size
        max_rank = rank
        compose = 0.0

        levels.times do
          effective_rank = rank_cap ? Math.min(rank, rank_cap.not_nil!) : rank
          # Two vector-dot/axpy style transformations dominate compact
          # transition + compact-B composition. Normalize against one serial
          # rowwise token step (~3*s*s lane work), matching the existing model.
          compose += 4.0 * effective_rank.to_f64 * effective_rank.to_f64 / (3.0 * state_size.to_f64)
          rank *= 2
          max_rank = rank if rank > max_rank
        end

        serial = prompt_tokens.to_f64
        parallel = (2 * block_size).to_f64 + compose
        RankGrowthEstimate.new(
          prompt_tokens: prompt_tokens,
          block_size: block_size,
          state_size: state_size,
          rank_cap: rank_cap,
          serial_depth: serial,
          parallel_depth: parallel,
          compose_depth: compose,
          speedup: serial / parallel,
          scan_levels: levels,
          max_rank_on_path: max_rank,
        )
      end

      def self.ceil_div(a : Int32, b : Int32) : Int32
        (a + b - 1) // b
      end

      def self.ceil_log2(n : Int32) : Int32
        raise ArgumentError.new("n must be positive") unless n > 0
        return 0 if n <= 1

        levels = 0
        value = 1
        while value < n
          value <<= 1
          levels += 1
        end
        levels
      end
    end
  end
end
