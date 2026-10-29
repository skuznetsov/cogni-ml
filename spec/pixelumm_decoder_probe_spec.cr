require "spec"
require "../examples/support/pixelumm_decoder_probe"

describe PixelUMMDecoderProbe do
  it "holds baseline input and geometry fixed across both A/A observer orders" do
    normal = PixelUMMDecoderProbe.aa_profile_schedule
    reverse = PixelUMMDecoderProbe.aa_profile_schedule(reverse: true)
    normal.map(&.kind).should eq(["a_aa_warm_off", "a_aa_0", "a_aa_1", "a_aa_2", "a_aa_3", "a_aa_late_off"])
    normal.map(&.profile).should eq([false, false, true, true, false, false])
    reverse.map(&.profile).should eq([false, true, false, false, true, false])
    [normal, reverse].each do |arms|
      arms.none?(&.alternate).should be_true
      arms.none?(&.combined_attention).should be_true
      arms.none?(&.shared_gate_up).should be_true
      arms.map(&.poisoned).should eq([true, false, false, false, false, false])
    end
  end

  it "paces from prior terminal return and rejects diagnostic overruns before submission" do
    PixelUMMDecoderProbe.aa_pause_remaining_ns(nil, 1_i64).should eq(0_i64)
    PixelUMMDecoderProbe.aa_pause_remaining_ns(1_i64, 36_000_000_001_i64).should eq(14_000_000_000_i64)
    PixelUMMDecoderProbe.aa_pause_remaining_ns(1_i64, 50_000_000_001_i64).should eq(0_i64)
    expect_raises(ArgumentError, /deadline/) do
      PixelUMMDecoderProbe.aa_pause_remaining_ns(1_i64, 50_000_000_002_i64)
    end
    expect_raises(ArgumentError, /clock/) do
      PixelUMMDecoderProbe.aa_pause_remaining_ns(2_i64, 1_i64)
    end
    expect_raises(ArgumentError, /clock/) do
      PixelUMMDecoderProbe.aa_pause_remaining_ns(Int64::MAX - 1, Int64::MAX)
    end
  end

  it "profiles combined attention in both orders with poisoned off warmups and late off controls" do
    normal = PixelUMMDecoderProbe.combined_attention_profile_schedule
    reverse = PixelUMMDecoderProbe.combined_attention_profile_schedule(reverse: true)
    normal.size.should eq(8)
    normal.map(&.profile).should eq([false, false, true, true, true, true, false, false])
    normal.map(&.combined_attention).should eq([true, false, false, true, true, false, false, true])
    reverse.map(&.combined_attention).should eq(normal.map { |arm| !arm.combined_attention })
    reverse.map(&.profile).should eq(normal.map(&.profile))
    normal.map(&.poisoned).should eq([true, true, false, false, false, false, false, false])
    reverse.map(&.poisoned).should eq(normal.map(&.poisoned))
    normal.none?(&.alternate).should be_true
    normal.none?(&.shared_gate_up).should be_true
    normal.map(&.kind).uniq.size.should eq(8)
    normal[2, 4].map(&.kind).should eq((0...4).map { |i| "a_profile_#{i}" })
  end

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

  it "gates combined attention on both inputs before unpoisoned same-input ABBA timing" do
    arms = PixelUMMDecoderProbe.combined_attention_schedule
    arms.size.should eq(8)

    gates = arms[0, 4]
    timed = arms[4, 4]
    gates.all? { |arm| arm.kind.starts_with?("gate_") }.should be_true
    timed.all? { |arm| arm.kind.starts_with?("timed_") }.should be_true
    arms.map(&.kind).uniq.size.should eq(8)

    gates.map(&.combined_attention).should eq([true, false, false, true])
    gates.map(&.alternate).should eq([false, false, true, true])
    gates.all?(&.poisoned).should be_true

    timed.map(&.combined_attention).should eq([false, true, true, false])
    timed.none?(&.alternate).should be_true
    timed.none?(&.poisoned).should be_true

    arms.none?(&.profile).should be_true
    arms.none?(&.shared_gate_up).should be_true
  end

  it "reverses combined-attention warm and timing orders and limits correctness-only to gates" do
    normal = PixelUMMDecoderProbe.combined_attention_schedule
    reverse = PixelUMMDecoderProbe.combined_attention_schedule(reverse: true)
    normal.size.should eq(8)
    reverse.size.should eq(8)
    reverse.map(&.combined_attention).should eq(normal.map { |arm| !arm.combined_attention })
    reverse.map(&.alternate).should eq(normal.map(&.alternate))
    reverse.map(&.poisoned).should eq(normal.map(&.poisoned))
    reverse[0, 2].map(&.combined_attention).should eq([false, true])
    reverse[2, 2].map(&.combined_attention).should eq([true, false])
    reverse[4, 4].map(&.combined_attention).should eq([true, false, false, true])

    correctness = PixelUMMDecoderProbe.combined_attention_schedule(correctness_only: true)
    correctness.size.should eq(4)
    correctness.all? { |arm| arm.kind.starts_with?("gate_") }.should be_true
    correctness.all?(&.poisoned).should be_true
    correctness.map(&.combined_attention).should eq([true, false, false, true])
    correctness.none?(&.profile).should be_true
    correctness.none?(&.shared_gate_up).should be_true

    reverse_correctness = PixelUMMDecoderProbe.combined_attention_schedule(
      reverse: true, correctness_only: true)
    reverse_correctness.size.should eq(4)
    reverse_correctness.map(&.combined_attention).should eq([false, true, true, false])
    reverse_correctness.map(&.alternate).should eq([false, false, true, true])
    reverse_correctness.all?(&.poisoned).should be_true
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

  it "changes scale/softmax selection only when prepared and idle" do
    plan = ML::Video::PixelUMMDecoderBlockMPS::Plan
    [true, false].each do |enabled|
      plan.validate_scale_softmax_change!(prepared: true, enabled: enabled, idle: true)
      expect_raises(ArgumentError, /in flight/) do
        plan.validate_scale_softmax_change!(prepared: true, enabled: enabled, idle: false)
      end
      expect_raises(ArgumentError, /prepared/) do
        plan.validate_scale_softmax_change!(prepared: false, enabled: enabled, idle: true)
      end
    end
  end

  it "qualifies both encoded graph topologies only with the required capabilities" do
    plan = ML::Video::PixelUMMDecoderBlockMPS::Plan
    [true, false].each do |shared|
      plan.validate_encode_mode!(profile: true, shared_gate_up: shared,
        prepared: true, ffn_f32: true, api_profile_enabled: true)
      expect_raises(ArgumentError, /API chain/) do
        plan.validate_encode_mode!(profile: true, shared_gate_up: shared,
          prepared: true, ffn_f32: true, api_profile_enabled: false)
      end
    end
    expect_raises(ArgumentError, /prepared/) do
      plan.validate_encode_mode!(profile: true, shared_gate_up: true,
        prepared: false, ffn_f32: true, api_profile_enabled: true)
    end
    expect_raises(ArgumentError, /F32/) do
      plan.validate_encode_mode!(profile: true, shared_gate_up: true,
        prepared: true, ffn_f32: false, api_profile_enabled: true)
    end
    plan.validate_encode_mode!(profile: false, shared_gate_up: false,
      prepared: false, ffn_f32: false, api_profile_enabled: false)
  end

  it "profiles both shared graph orders after two observer-off warmups within the bounded schedule" do
    arms = PixelUMMDecoderProbe.shared_gate_up_profile_schedule
    arms.size.should eq(6)
    arms.map(&.shared_gate_up).should eq([true, false, false, true, true, false])
    arms.map(&.profile).should eq([false, false, true, true, true, true])
    arms.none?(&.alternate).should be_true
    arms.none?(&.poisoned).should be_true
    arms.map(&.kind).uniq.size.should eq(6)
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
