require "spec"
require "../src/ml/gguf/inkling2_engine"

describe ML::GGUF::Inkling2Math do
  it "mixes normalized block values with a separate query for each controller" do
    bank = ML::GGUF::Inkling2Math.normalize_bank([[2_f32, 0_f32], [0_f32, 1_f32]])
    mix = ML::GGUF::Inkling2Math.depth_mix(bank, [1_f32, 0_f32, 0_f32, 1_f32])
    preferred = (Math.sqrt(2) / (1 + Math.exp(-1))).to_f32
    other = Math.sqrt(2).to_f32 - preferred
    mix[0][0].should be_close(preferred, 1e-5_f32)
    mix[0][1].should be_close(other, 1e-5_f32)
    mix[1][0].should be_close(other, 1e-5_f32)
    mix[1][1].should be_close(preferred, 1e-5_f32)
  end

  it "selects experts by biased scores and combines them with unbiased weights" do
    logits = [Math.log(3).to_f32, 0_f32, -Math.log(3).to_f32]
    experts = ML::GGUF::Inkling2Math.route(logits, [0_f32, 0_f32, 0.6_f32], 2, 1.5_f32)
    experts.map(&.[0]).should eq([2, 0])
    experts[0][1].should be_close(0.375_f32, 1e-6_f32)
    experts[1][1].should be_close(1.125_f32, 1e-6_f32)
  end

  it "measures changes in the direction of the normalized loop bank" do
    previous = ML::GGUF::Inkling2Math.normalize_bank([[2_f32, 0_f32]])
    rescaled = ML::GGUF::Inkling2Math.normalize_bank([[4_f32, 0_f32]])
    changed = ML::GGUF::Inkling2Math.normalize_bank([[0_f32, 2_f32]])
    ML::GGUF::Inkling2Math.relative_delta(previous, rescaled).should be_close(0_f32, 1e-6_f32)
    ML::GGUF::Inkling2Math.relative_delta(previous, changed).should be_close(Math.sqrt(2).to_f32, 1e-6_f32)
  end

  it "requires two successive small changes to stop the core loop" do
    threshold = 0.1_f32
    ML::GGUF::Inkling2Math.below_loop_threshold?(nil, 0.05_f32, threshold).should be_false
    ML::GGUF::Inkling2Math.below_loop_threshold?(0.05_f32, 0.12_f32, threshold).should be_false
    ML::GGUF::Inkling2Math.below_loop_threshold?(0.12_f32, 0.05_f32, threshold).should be_false
    ML::GGUF::Inkling2Math.below_loop_threshold?(0.05_f32, 0.1_f32, threshold).should be_true
  end
end

describe ML::GGUF::Inkling2Engine do
  it "keeps Mini at its trained depth and allows bounded research overrides" do
    ML::GGUF::Inkling2Engine.resolve_passes(3, false, nil).should eq(3)
    ML::GGUF::Inkling2Engine.resolve_passes(3, false, 3).should eq(3)
    expect_raises(ArgumentError, /requires 3/) { ML::GGUF::Inkling2Engine.resolve_passes(3, false, 10) }
    ML::GGUF::Inkling2Engine.resolve_passes(3, true, 1).should eq(1)
    ML::GGUF::Inkling2Engine.resolve_passes(3, true, 10).should eq(10)
    expect_raises(ArgumentError, /1 to 10/) { ML::GGUF::Inkling2Engine.resolve_passes(3, true, 0) }
    expect_raises(ArgumentError, /1 to 10/) { ML::GGUF::Inkling2Engine.resolve_passes(3, true, 11) }
    expect_raises(ArgumentError, /1 to 10/) { ML::GGUF::Inkling2Engine.resolve_passes(11, true, nil) }
  end
end
