require "digest/sha256"

module ML::QwenVsLlamaBenchmarkContract
  enum HeadMode
    DecoderBodyLowerBound
    FusedTop1
    FullLogits
  end

  enum ComparisonLevel
    Incomparable
    Diagnostic
    SameToken
  end

  record Comparison,
    level : ComparisonLevel,
    scope : String,
    reason : String

  # This is a harness declaration, not an execution certificate. It describes
  # the externally observable work that the runner checks while executing.
  # Internal kernels, cache formats, and physical tiling may differ: those
  # differences are what the benchmark is intended to measure.
  record PrefillWorkloadDeclaration,
    tokens : Array(Int32),
    initial_depth : Int32,
    final_depth : Int32,
    logical_prompts : Int32,
    output_rows : Int32,
    output_width : Int32,
    full_logits : Bool,
    synchronized : Bool,
    state_reused : Bool,
    setup_outside_timing : Bool,
    host_copy_inside_timing : Bool,
    warmup_runs : Int32

  DEFAULT_PREFILL_HEAD = HeadMode::FullLogits
  DEFAULT_DECODE_HEAD  = HeadMode::FullLogits
  TOKEN_STREAM_DOMAIN  = "cogni-ml-qwen-vs-llama-token-stream-v1\0"

  def self.synthetic_prefill_tokens(count : Int32, vocab_size : Int32) : Array(Int32)
    raise ArgumentError.new("token count must be positive") unless count > 0
    raise ArgumentError.new("vocabulary size must be positive") unless vocab_size > 0

    modulus = Math.min(vocab_size, 1000)
    Array(Int32).new(count) { |i| ((i.to_i64 * 7 + 11) % modulus).to_i32 }
  end

  def self.token_stream_sha256(tokens : Array(Int32)) : String
    raise ArgumentError.new("token stream must not be empty") if tokens.empty?

    io = IO::Memory.new
    io << TOKEN_STREAM_DOMAIN
    io.write_bytes(tokens.size.to_i32, IO::ByteFormat::LittleEndian)
    tokens.each { |token| io.write_bytes(token, IO::ByteFormat::LittleEndian) }
    Digest::SHA256.hexdigest(io.to_slice)
  end

  def self.same_token_prefill_comparison(native : PrefillWorkloadDeclaration,
                                         llama : PrefillWorkloadDeclaration) : Comparison
    mismatch = same_token_prefill_mismatch(native, llama)
    if mismatch
      return Comparison.new(
        ComparisonLevel::Diagnostic,
        "prefill_workload_mismatch",
        mismatch,
      )
    end

    Comparison.new(
      ComparisonLevel::SameToken,
      "same_token_prefill_external_workload",
      "same declared token stream, empty logical state, one final full-logit row, reused-cleared state, and timer boundary; the runner supplies execution checks, while internal cache formats and kernel scheduling may differ",
    )
  end

  private def self.same_token_prefill_mismatch(native : PrefillWorkloadDeclaration,
                                               llama : PrefillWorkloadDeclaration) : String?
    unless native.tokens == llama.tokens
      return "native and llama token streams differ"
    end
    token_count = native.tokens.size
    return "same-token prefill requires a non-empty token stream" unless token_count > 0

    unless native.initial_depth == 0 && llama.initial_depth == 0 &&
           native.final_depth == token_count &&
           llama.final_depth == token_count
      return "prefill state depth does not match the consumed token stream"
    end

    unless native.logical_prompts == 1 && llama.logical_prompts == 1 &&
           native.output_rows == 1 && llama.output_rows == 1
      return "same-token prefill requires one logical prompt and one final output row"
    end

    unless native.output_width > 0 && native.output_width == llama.output_width &&
           native.full_logits && llama.full_logits
      return "same-token prefill requires the same full-logit output width"
    end

    unless native.synchronized && llama.synchronized &&
           native.state_reused && llama.state_reused &&
           native.setup_outside_timing && llama.setup_outside_timing &&
           native.host_copy_inside_timing && llama.host_copy_inside_timing &&
           native.warmup_runs == llama.warmup_runs
      return "native and llama timing or state-reuse boundaries differ"
    end

    nil
  end

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
