require "./spec_helper"
require "../src/ml/gguf/qwen35_deltanet_scan_model"

describe ML::GGUF::Qwen35DeltaNetScanModel do
  it "rejects naive dense summaries at s=128 for short and medium prefill" do
    model = ML::GGUF::Qwen35DeltaNetScanModel
    dense = model.naive_dense_compose_token_equiv(128)

    pp64 = model.estimate(64, 16, dense, 1.25)
    pp256 = model.estimate(256, 16, dense, 1.25)

    pp64.speedup.should be < 1.0
    pp256.speedup.should be < 1.25
  end

  it "shows the compose-cost budget needed for a 1.25x pp256 block scan" do
    est = ML::GGUF::Qwen35DeltaNetScanModel.estimate(256, 16, 32.0, 1.25)

    est.scan_levels.should eq(4)
    est.max_compose_for_target.should be_close(43.2, 1.0e-9)
    est.speedup.should be > 1.25
  end

  it "keeps the short-prompt adversary visible" do
    est = ML::GGUF::Qwen35DeltaNetScanModel.estimate(64, 16, 32.0, 1.25)

    est.scan_levels.should eq(2)
    est.speedup.should be < 1.25
  end

  it "shows compact prefix scan needs rank compression on long prompts" do
    model = ML::GGUF::Qwen35DeltaNetScanModel
    uncapped = model.rank_growth_estimate(1024, 16, 128, nil)
    capped = model.rank_growth_estimate(1024, 16, 128, 128)

    uncapped.max_rank_on_path.should eq(1024)
    uncapped.speedup.should be < 1.0
    capped.speedup.should be > 1.25
  end
end
