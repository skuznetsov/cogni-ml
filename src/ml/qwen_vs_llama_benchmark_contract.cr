module ML::QwenVsLlamaBenchmarkContract
  enum HeadMode
    DecoderBodyLowerBound
    FusedTop1
    FullLogits
  end

  enum ComparisonLevel
    Incomparable
    Diagnostic
    Strict
  end

  record Comparison,
    level : ComparisonLevel,
    scope : String,
    reason : String

  DEFAULT_PREFILL_HEAD = HeadMode::FullLogits
  DEFAULT_DECODE_HEAD  = HeadMode::FullLogits

  def self.prefill_label(mode : HeadMode) : String
    case mode
    in .decoder_body_lower_bound? then "prompt_decoder_body_lower_bound"
    in .fused_top1?               then "prompt_plus_final_top1"
    in .full_logits?              then "prompt_plus_final_full_logits"
    end
  end

  def self.decode_label(mode : HeadMode) : String
    case mode
    in .decoder_body_lower_bound? then "decoder_body_lower_bound"
    in .fused_top1?               then "product_greedy_top1"
    in .full_logits?              then "full_logits"
    end
  end

  def self.prefill_comparison(mode : HeadMode, *, cached : Bool) : Comparison
    if cached
      return Comparison.new(
        ComparisonLevel::Incomparable,
        "cache_restore",
        "native cache restore is not prompt processing",
      )
    end

    case mode
    in .full_logits?
      Comparison.new(
        ComparisonLevel::Diagnostic,
        "full_logits_diagnostic",
        "both paths expose full logits, but token streams, prompt batching/output-row count, and state-buffer lifecycle can differ",
      )
    in .decoder_body_lower_bound?
      Comparison.new(
        ComparisonLevel::Incomparable,
        "decoder_body_lower_bound",
        "native decoder-body lower bound skips the lm-head and logit transfer that unmodified llama-bench performs",
      )
    in .fused_top1?
      Comparison.new(
        ComparisonLevel::Incomparable,
        "product_final_top1",
        "native fused top1 does less output work than llama-bench full logits",
      )
    end
  end

  def self.decode_comparison(mode : HeadMode) : Comparison
    case mode
    in .full_logits?
      Comparison.new(
        ComparisonLevel::Diagnostic,
        "full_logits_diagnostic",
        "both paths expose full logits per token, but token streams, seeded states, and state-buffer lifecycle differ",
      )
    in .decoder_body_lower_bound?
      Comparison.new(
        ComparisonLevel::Incomparable,
        "decoder_body_lower_bound",
        "native decoder-body lower bound skips the lm-head and logit transfer that unmodified llama-bench performs",
      )
    in .fused_top1?
      Comparison.new(
        ComparisonLevel::Incomparable,
        "product_greedy_top1",
        "native greedy top1 feeds model outputs while llama-bench feeds synthetic tokens and copies full logits",
      )
    end
  end

  def self.llama_depth_args(depth : Int32) : Array(String)
    raise ArgumentError.new("decode depth must be non-negative") if depth < 0
    return [] of String if depth == 0

    ["-d", depth.to_s]
  end

  def self.validate_llama_result_shape!(row_count : Int32,
                                        actual_prompt : Int32,
                                        actual_gen : Int32,
                                        actual_depth : Int32,
                                        expected_prompt : Int32,
                                        expected_gen : Int32,
                                        expected_depth : Int32) : Nil
    unless row_count == 1 &&
           actual_prompt == expected_prompt &&
           actual_gen == expected_gen &&
           actual_depth == expected_depth
      raise ArgumentError.new(
        "llama-bench returned an unexpected benchmark shape: " \
        "rows=#{row_count} prompt=#{actual_prompt} gen=#{actual_gen} depth=#{actual_depth}; " \
        "expected rows=1 prompt=#{expected_prompt} gen=#{expected_gen} depth=#{expected_depth}"
      )
    end
  end

  # llama-bench avg_ts is the arithmetic mean of throughput for each sample.
  # Keep the native side on the same statistic instead of comparing a p50 with
  # llama-bench's mean or computing tokens / mean(sample duration).
  def self.mean_throughput(samples_ms : Array(Float64), n_tokens : Int32) : Float64
    raise ArgumentError.new("throughput requires at least one sample") if samples_ms.empty?
    raise ArgumentError.new("throughput token count must be positive") unless n_tokens > 0
    raise ArgumentError.new("throughput sample durations must be positive") if samples_ms.any? { |ms| ms <= 0.0 }

    samples_ms.sum(0.0) { |ms| n_tokens * 1000.0 / ms } / samples_ms.size
  end
end
