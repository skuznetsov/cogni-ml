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
