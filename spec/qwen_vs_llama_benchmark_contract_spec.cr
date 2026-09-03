require "./spec_helper"
require "../src/ml/qwen_vs_llama_benchmark_contract"

describe ML::QwenVsLlamaBenchmarkContract do
  it "defaults to full logits without claiming strict apples-to-apples parity" do
    contract = ML::QwenVsLlamaBenchmarkContract
    ML::QwenVsLlamaBenchmarkContract::DEFAULT_PREFILL_HEAD.should eq(ML::QwenVsLlamaBenchmarkContract::HeadMode::FullLogits)
    ML::QwenVsLlamaBenchmarkContract::DEFAULT_DECODE_HEAD.should eq(ML::QwenVsLlamaBenchmarkContract::HeadMode::FullLogits)

    prefill = contract.prefill_comparison(ML::QwenVsLlamaBenchmarkContract::HeadMode::FullLogits, cached: false)
    decode = contract.decode_comparison(ML::QwenVsLlamaBenchmarkContract::HeadMode::FullLogits)
    prefill.level.diagnostic?.should be_true
    decode.level.diagnostic?.should be_true
    prefill.level.strict?.should be_false
    decode.level.strict?.should be_false
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
end
