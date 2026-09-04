require "./spec_helper"
require "../src/ml/qwen_vs_llama_benchmark_contract"

describe ML::QwenVsLlamaBenchmarkContract do
  it "isolates comparison engines in sequential worker processes" do
    source = File.read(Path[__DIR__] / "../bin/benchmark_qwen_prefill_vs_llama_same_token.cr")
    source.includes?(%q{parser.on("--split-process"}).should be_true
    source.includes?(%q{parser.on("--worker=ENGINE"}).should be_true
    source.includes?(%q{SPLIT_PROCESS_ORDER  = ["native", "llama", "llama", "native"]}).should be_true
    source.index("if split_process").not_nil!.should be < source.index("native_weights : ML::GGUF::Qwen35Weights? = nil").not_nil!
    source.includes?(%q{Process.run(worker_executable}).should be_true
    source.includes?(%q{model_identity.same_file?(current_identity)}).should be_true
  end

  it "moves a fail-closed same-token and full-logit certificate across workers" do
    source = File.read(Path[__DIR__] / "../bin/benchmark_qwen_prefill_vs_llama_same_token.cr")
    source.includes?("SPLIT_WORKER_SCHEMA").should be_true
    source.includes?("token_sha256 : String").should be_true
    source.includes?("samples_ms : Array(Float64)").should be_true
    source.includes?("logits : Array(Float32)").should be_true
    source.includes?("validate_split_worker_result!").should be_true
    source.includes?("split worker token stream mismatch").should be_true
    source.includes?("split worker output width mismatch").should be_true
    source.includes?("effective_n_batch : Int32?").should be_true
    source.includes?("effective_n_ubatch : Int32?").should be_true
    source.includes?("split worker llama batch geometry missing").should be_true
    source.includes?("llama split worker batch geometry changed across processes").should be_true
  end

  it "matches llama-bench reset and performs only the getter synchronization" do
    source = File.read(Path[__DIR__] / "../bin/benchmark_qwen_prefill_vs_llama_same_token.cr")
    source.includes?("@context.kv_clear(data: false)").should be_true
    source.includes?("LlamaFFI.llama_synchronize(@context.handle)").should be_false
    source.includes?("native_cooldown_ms").should be_true
    source.includes?("unless native.terminal_last_used").should be_true
    source.includes?(%q{"n/a"}).should be_true
    source.includes?("model_capability: weights.output.q4_gemv_x16_capability").should be_true
    source.includes?("gguf_file_type: weights.gguf_file_type").should be_true
  end

  it "releases native Metal state between prompt sizes" do
    source = File.read(Path[__DIR__] / "../bin/benchmark_qwen_prefill_vs_llama_same_token.cr")
    source.includes?("Qwen35CPU.release_state_metal!(@state)").should be_true
    source.includes?("native_runner : NativePrefillRunner? = nil").should be_true
    in_process_main = source[source.index("native_runner : NativePrefillRunner? = nil").not_nil!..]
    in_process_main.index("native_runner = NativePrefillRunner.new").not_nil!.should be < in_process_main.index("warmup.times").not_nil!
    source.includes?("-> { native_runner.try(&.close) }").should be_true
    source.includes?("-> { native_weights.try(&.close) }").should be_true
  end

  it "covers partial benchmark construction with cleanup boundaries" do
    source = File.read(Path[__DIR__] / "../bin/benchmark_qwen_prefill_vs_llama_same_token.cr")
    source.includes?("rescue ex\n      close\n      raise ex").should be_true
    source.includes?("rescue ex\n      @context.free\n      raise ex").should be_true
    source.includes?("native_weights : ML::GGUF::Qwen35Weights? = nil").should be_true
    source.includes?("llama_model : ML::LLM::Model? = nil").should be_true
    source.includes?("ML::LLM.cleanup if llama_backend_initialized").should be_true
    source.includes?("def run_cleanups(cleanups : Enumerable(Proc(Nil))) : Nil").should be_true
    source.includes?("first_error ||= ex").should be_true
    source.index("-> { ML::LLM.cleanup if llama_backend_initialized }").not_nil!.should be < source.index("-> { llama_model.try(&.free) }").not_nil!
  end

  it "defaults to full logits without claiming strict apples-to-apples parity" do
    contract = ML::QwenVsLlamaBenchmarkContract
    ML::QwenVsLlamaBenchmarkContract::DEFAULT_PREFILL_HEAD.should eq(ML::QwenVsLlamaBenchmarkContract::HeadMode::FullLogits)
    ML::QwenVsLlamaBenchmarkContract::DEFAULT_DECODE_HEAD.should eq(ML::QwenVsLlamaBenchmarkContract::HeadMode::FullLogits)

    prefill = contract.prefill_comparison(ML::QwenVsLlamaBenchmarkContract::HeadMode::FullLogits, cached: false)
    decode = contract.decode_comparison(ML::QwenVsLlamaBenchmarkContract::HeadMode::FullLogits)
    prefill.level.diagnostic?.should be_true
    decode.level.diagnostic?.should be_true
    prefill.level.same_token?.should be_false
    decode.level.same_token?.should be_false
    prefill.reason.should contain("output-row count")
    decode.reason.should contain("state-buffer lifecycle")
  end

  it "keeps decoder-body and fused-top1 measurements out of llama-bench gaps" do
    contract = ML::QwenVsLlamaBenchmarkContract
    body = ML::QwenVsLlamaBenchmarkContract::HeadMode::DecoderBodyLowerBound
    top1 = ML::QwenVsLlamaBenchmarkContract::HeadMode::FusedTop1

    contract.prefill_comparison(body, cached: false).level.incomparable?.should be_true
    contract.decode_comparison(body).level.incomparable?.should be_true
    contract.prefill_comparison(top1, cached: false).level.incomparable?.should be_true
    contract.decode_comparison(top1).level.incomparable?.should be_true
    contract.decode_label(body).should eq("decoder_body_lower_bound")
    contract.decode_label(top1).should eq("product_greedy_top1")
  end

  it "marks prompt-cache restore as incomparable to prompt processing" do
    contract = ML::QwenVsLlamaBenchmarkContract
    comparison = contract.prefill_comparison(ML::QwenVsLlamaBenchmarkContract::HeadMode::FullLogits, cached: true)

    comparison.level.incomparable?.should be_true
    comparison.reason.should contain("cache restore")
  end

  it "adds llama-bench depth only for contextual decode" do
    contract = ML::QwenVsLlamaBenchmarkContract
    contract.llama_depth_args(0).should eq([] of String)
    contract.llama_depth_args(4096).should eq(["-d", "4096"])
    expect_raises(ArgumentError) { contract.llama_depth_args(-1) }
  end

  it "rejects llama-bench output whose extra arguments changed the measured shape" do
    contract = ML::QwenVsLlamaBenchmarkContract
    contract.validate_llama_result_shape!(1, 64, 0, 0, 64, 0, 0)

    expect_raises(ArgumentError, /unexpected benchmark shape/) do
      contract.validate_llama_result_shape!(2, 64, 0, 0, 64, 0, 0)
    end
    expect_raises(ArgumentError, /unexpected benchmark shape/) do
      contract.validate_llama_result_shape!(1, 64, 0, 4096, 64, 0, 0)
    end
  end

  it "compares mean per-repetition throughput with llama-bench avg_ts" do
    contract = ML::QwenVsLlamaBenchmarkContract
    # llama-bench avg_ts is mean(tokens / sample_time), not tokens / mean(sample_time).
    contract.mean_throughput([100.0, 200.0], 10).should be_close(75.0, 1e-9)
  end

  it "admits same-token prefill only for the same explicit stream and timing contract" do
    contract = ML::QwenVsLlamaBenchmarkContract
    tokens = contract.synthetic_prefill_tokens(4, 1000)
    tokens.should eq([11, 18, 25, 32])

    declaration = ML::QwenVsLlamaBenchmarkContract::PrefillWorkloadDeclaration.new(
      tokens: tokens,
      initial_depth: 0,
      final_depth: 4,
      logical_prompts: 1,
      output_rows: 1,
      output_width: 1000,
      full_logits: true,
      synchronized: true,
      state_reused: true,
      setup_outside_timing: true,
      host_copy_inside_timing: true,
      warmup_runs: 1,
    )

    comparison = contract.same_token_prefill_comparison(declaration, declaration)
    comparison.level.same_token?.should be_true
    comparison.scope.should eq("same_token_prefill_external_workload")
  end

  it "fails closed when one same-token prefill token changes" do
    contract = ML::QwenVsLlamaBenchmarkContract
    native_tokens = [11, 18, 25, 32]
    llama_tokens = native_tokens.dup
    llama_tokens[2] = 26

    native = ML::QwenVsLlamaBenchmarkContract::PrefillWorkloadDeclaration.new(
      tokens: native_tokens,
      initial_depth: 0,
      final_depth: 4,
      logical_prompts: 1,
      output_rows: 1,
      output_width: 1000,
      full_logits: true,
      synchronized: true,
      state_reused: true,
      setup_outside_timing: true,
      host_copy_inside_timing: true,
      warmup_runs: 1,
    )
    llama = native.copy_with(tokens: llama_tokens)

    comparison = contract.same_token_prefill_comparison(native, llama)
    comparison.level.diagnostic?.should be_true
    comparison.reason.should contain("token stream")
  end

  it "fails closed on extra output rows or a mismatched timer boundary" do
    contract = ML::QwenVsLlamaBenchmarkContract
    base = ML::QwenVsLlamaBenchmarkContract::PrefillWorkloadDeclaration.new(
      tokens: [11, 18, 25, 32],
      initial_depth: 0,
      final_depth: 4,
      logical_prompts: 1,
      output_rows: 1,
      output_width: 1000,
      full_logits: true,
      synchronized: true,
      state_reused: true,
      setup_outside_timing: true,
      host_copy_inside_timing: true,
      warmup_runs: 1,
    )

    contract.same_token_prefill_comparison(base, base.copy_with(logical_prompts: 2, output_rows: 2)).level.diagnostic?.should be_true
    contract.same_token_prefill_comparison(base, base.copy_with(host_copy_inside_timing: false)).level.diagnostic?.should be_true
  end
end
