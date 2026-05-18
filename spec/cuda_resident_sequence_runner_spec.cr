require "./spec_helper"
require "../src/ml/cuda/driver"

describe ML::CUDA::ResidentSequenceRunner do
  it "runs only the configured active token prefix on the token-local path" do
    seen = [] of Int32
    runner = ML::CUDA::ResidentSequenceRunner.new(
      4,
      -> { },
      -> { },
      ->(tok : Int32) { seen << tok })

    runner.active_tokens = 2
    runner.run_sequence

    seen.should eq([0, 1])
    runner.run_repeated(3).should eq(6)
  end

  it "passes active token count to sequence overrides" do
    seen = [] of Int32
    runner = ML::CUDA::ResidentSequenceRunner.new(
      8,
      -> { },
      -> { },
      ->(_tok : Int32) { },
      nil,
      ->(active_tokens : Int32) { seen << active_tokens })

    runner.active_tokens = 4
    runner.run_sequence

    seen.should eq([4])
  end

  it "rejects invalid active token counts" do
    runner = ML::CUDA::ResidentSequenceRunner.new(
      4,
      -> { },
      -> { },
      ->(_tok : Int32) { })

    expect_raises(ArgumentError, "active_tokens must be positive") do
      runner.active_tokens = 0
    end

    expect_raises(ArgumentError, "active_tokens must be <= tokens") do
      runner.active_tokens = 5
    end
  end
end
