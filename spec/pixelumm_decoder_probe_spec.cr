require "spec"
require "../examples/support/pixelumm_decoder_probe"

describe PixelUMMDecoderProbe do
  it "warms both gate/up modes before same-input ABBA and poisoned alternate-input controls" do
    arms = PixelUMMDecoderProbe.shared_gate_up_schedule
    arms.size.should eq(8)
    arms.first.kind.should eq("a_warm_shared")
    arms.map(&.shared_gate_up).should eq([true, false, false, true, true, false, false, true])
    arms.map(&.alternate).should eq([false, false, false, false, false, false, true, true])
    arms.map(&.poisoned).should eq([false, false, false, false, false, false, true, true])
    arms.none?(&.profile).should be_true
    arms.map(&.kind).uniq.size.should eq(8)
  end

  it "admits shared gate/up only with F32 FFN capability and an idle chain" do
    plan = ML::Video::PixelUMMDecoderBlockMPS::Plan
    plan.validate_shared_gate_up_configuration!(ffn_f32: true, prepare_shared_gate_up: true)
    plan.validate_shared_gate_up_configuration!(ffn_f32: false, prepare_shared_gate_up: false)
    expect_raises(ArgumentError, /F32/) do
      plan.validate_shared_gate_up_configuration!(ffn_f32: false, prepare_shared_gate_up: true)
    end
    plan.validate_shared_gate_up_change!(prepared: true, enabled: true, idle: true)
    expect_raises(ArgumentError, /prepared/) do
      plan.validate_shared_gate_up_change!(prepared: false, enabled: true, idle: true)
    end
    [true, false].each do |enabled|
      expect_raises(ArgumentError, /in flight/) do
        plan.validate_shared_gate_up_change!(prepared: true, enabled: enabled, idle: false)
      end
    end
  end

  it "rejects unqualified shared gate/up phase profiling before submission" do
    plan = ML::Video::PixelUMMDecoderBlockMPS::Plan
    plan.validate_encode_mode!(profile: false, shared_gate_up: true)
    plan.validate_encode_mode!(profile: true, shared_gate_up: false)
    expect_raises(ArgumentError, /not qualified/) do
      plan.validate_encode_mode!(profile: true, shared_gate_up: true)
    end
  end

  it "brackets the profiled input and poison controls with opposite profiler orders on input A" do
    arms = PixelUMMDecoderProbe.variance_schedule
    arms.size.should eq(6)
    arms.map(&.profile).should eq([false, true, true, true, true, false])
    arms.map(&.alternate).should eq([false, false, true, true, false, false])
    arms.map(&.poisoned).should eq([false, false, false, true, false, false])
    arms.map(&.kind).uniq.size.should eq(6)
  end

  it "admits the derived unique full-J aggregate without counting the borrowed head input twice" do
    block_named = 22_958_105_572_i64
    head_weights = 25_188_352_i64
    config = ML::Video::PixelUMMDecoderHeadMPS::Config.new(temporal: 6, height: 44, width: 80)
    head_owned = ML::Video::PixelUMMDecoderHeadMPS::Plan.owned_bytes(config, PixelUMMHeadProbe::CAPTURES)

    head_owned.should eq(1_124_851_204_i64)
    required = PixelUMMDecoderProbe.aggregate_named_bytes(
      block_named: block_named, head_weight_named: head_weights, head_owned_named: head_owned)
    required.should eq(24_108_145_128_i64)
    PixelUMMDecoderProbe.validate_aggregate_budget!(required, required).should eq(required)
    expect_raises(ArgumentError, /aggregate/) do
      PixelUMMDecoderProbe.validate_aggregate_budget!(required, required - 1)
    end
  end

  it "requires the head fixture input to be exactly the authenticated block-35 output" do
    digest = "a" * 64
    PixelUMMDecoderProbe.validate_pair!(digest, digest)
    expect_raises(ArgumentError, /block35/) do
      PixelUMMDecoderProbe.validate_pair!(digest, "b" * 64)
    end
  end

  it "rejects head encoding until all block encoding returned on the same open chain" do
    PixelUMMDecoderProbe.validate_head_order!(blocks_encoded: true, same_chain: true, chain_open: true)
    {
      {blocks_encoded: false, same_chain: true, chain_open: true},
      {blocks_encoded: true, same_chain: false, chain_open: true},
      {blocks_encoded: true, same_chain: true, chain_open: false},
    }.each do |case_values|
      expect_raises(ArgumentError, /head.*after.*blocks/) do
        PixelUMMDecoderProbe.validate_head_order!(**case_values)
      end
    end
  end
end
