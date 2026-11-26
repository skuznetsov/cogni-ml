require "spec"
require "../examples/support/pixelumm_ffn_weight_reuse_probe"

alias P = PixelUMMFFNWeightReuseProbe

module PixelUMMFFNWeightReuseProbeSpecSupport
  extend self

  def lifecycle_entry(index : Int32, mode : String, kind : String,
                      capture_mode : Bool = false,
                      baseline_schedule : String = "window-major") : P::ScheduleEntry
    {index: index, mode: mode, kind: kind, capture_mode: capture_mode,
     baseline_schedule: baseline_schedule, candidate_schedule: "block-major",
     timing_eligible: kind == "timed"}
  end
end

describe PixelUMMFFNWeightReuseProbe do
  it "admits the exact T10H18W32 inventory below the bounded 9 GiB cap" do
    budget = P.preflight_budget(P::MAX_NAMED_BYTES)
    budget[:workspace_bytes].should be > 0
    budget[:input_weight_bytes].should eq(877_954_048_i64)
    budget[:stage_bytes].should eq(603_979_776_i64)
    budget[:expected_capture_bytes].should eq(2_264_924_160_i64)
    budget[:metric_bytes].should eq(P::METRIC_BYTES)
    budget[:workspace_bytes].should eq(3_634_642_020_i64)
    budget[:total_bytes].should eq(7_381_500_468_i64)
    budget[:total_bytes].should be <= 9_i64 * P::GIB

    expect_raises(ArgumentError) do
      P.preflight_budget(budget[:total_bytes] - 1)
    end
    expect_raises(ArgumentError) do
      P.preflight_budget(P::MAX_NAMED_BYTES + 1)
    end
  end

  it "accepts only BF16 captures and the pinned BF16/F32 input geometry" do
    P.validate_stage_geometry!("BF16", [1, 5760, 4096], 47_185_920_i64).should be_nil
    P.validate_stage_geometry!("BF16", [1, 5760, 12288], 141_557_760_i64).should be_nil
    expect_raises(ArgumentError) do
      P.validate_stage_geometry!("F32", [1, 5760, 4096], 94_371_840_i64)
    end
    expect_raises(ArgumentError) do
      P.validate_stage_geometry!("BF16", [1, 5760, 4096], 47_185_918_i64)
    end

    x = PixelUMMBlockProbe::Record.new(JSON.parse({dtype: "BF16", shape: [1, 5760, 4096],
                                                   offset: 0, nbytes: 47_185_920, sha256: "a" * 64}.to_json), 100_000_000_i64)
    rope = PixelUMMBlockProbe::Record.new(JSON.parse({dtype: "F32", shape: [1, 5760, 1, 128],
                                                      offset: 0, nbytes: 2_949_120, sha256: "b" * 64}.to_json), 100_000_000_i64)
    P.validate_input_geometry!("x", x).should be_nil
    P.validate_input_geometry!("cos", rope).should be_nil
    P.validate_input_geometry!("sin", rope).should be_nil
    expect_raises(ArgumentError) { P.validate_input_geometry!("cos", x) }
  end

  it "keeps packed comparator mismatches localizable in strict failure diagnostics" do
    metric = {failures: 3_u32, different_words: 17_u32,
              nonfinite: 2_u32, elements: 4096_u32}
    P.metric_diagnostic(metric).should eq(
      "failures=3 different_words=17 nonfinite=2 elements=4096")
  end

  it "binds the two overlapping saved-noise preview windows to their fixed source slices" do
    P.validate_window_slice!("w0", 0, 10, 0, 40).should be_nil
    P.validate_window_slice!("w1", 5, 10, 20, 40).should be_nil
    expect_raises(ArgumentError) { P.validate_window_slice!("w1", 0, 10, 20, 40) }
    expect_raises(ArgumentError) { P.validate_window_slice!("w0", 0, 10, 0, 30) }
  end

  it "uses baseline schedule selection to isolate staging from loop reordering" do
    window_major = P.schedule
    window_major.size.should eq(14)
    window_major[0...2].map(&.[:mode]).should eq(["baseline", "staged_f32"])
    window_major[2...6].map(&.[:mode]).should eq(["baseline", "staged_f32", "staged_f32", "baseline"])
    window_major[6..].map(&.[:mode]).should eq(["baseline", "staged_f32", "staged_f32", "baseline",
                                                "staged_f32", "baseline", "baseline", "staged_f32"])
    window_major.each { |request| request[:baseline_schedule].should eq("window-major") }
    P.plan_order("baseline", "window-major").should eq([0, 1, 2, 3])
    P.plan_order("staged_f32", "window-major").should eq([0, 2, 1, 3])

    block_major = P.schedule("block-major")
    block_major.each { |request| request[:baseline_schedule].should eq("block-major") }
    P.plan_order("baseline", "block-major").should eq([0, 2, 1, 3])
    expect_raises(ArgumentError) { P.schedule("random") }
    expect_raises(ArgumentError) { P.plan_order("staged_f32", "random") }
  end

  it "rejects stale input identities and out-of-order or skipped native plans" do
    ids = ["window-0", "window-1"]
    baseline = PixelUMMFFNWeightReuseProbeSpecSupport.lifecycle_entry(0, "baseline", "correctness", capture_mode: true)
    lifecycle = P::RequestLifecycle.new
    expect_raises(ArgumentError) do
      lifecycle.begin_request!(baseline, 0, ["stale", ids[1]], ids)
    end
    expect_raises(ArgumentError) do
      lifecycle.begin_request!(PixelUMMFFNWeightReuseProbeSpecSupport.lifecycle_entry(1, "baseline", "correctness", capture_mode: true),
        1, ids, ids)
    end

    lifecycle.begin_request!(baseline, 0, ids, ids)
    lifecycle.mark_submission_attempt!
    expect_raises(ArgumentError) { lifecycle.mark_plan_encoded!(1) }
    lifecycle.mark_plan_encoded!(0)
    lifecycle.mark_plan_encoded!(1)
    lifecycle.mark_plan_encoded!(2)
    lifecycle.mark_plan_encoded!(3)
    lifecycle.finish_request!(completed: true, wait_count: 1).should be_nil
    lifecycle.next_index.should eq(1)
  end

  it "stages each block once before consumers and checks every correctness consumer" do
    ids = ["window-0", "window-1"]
    correctness = PixelUMMFFNWeightReuseProbeSpecSupport.lifecycle_entry(0, "staged_f32", "correctness", capture_mode: true)
    lifecycle = P::RequestLifecycle.new
    lifecycle.begin_request!(correctness, 0, ids, ids)
    order = P.plan_order("staged_f32", correctness[:baseline_schedule])
    conversions = [] of Int32
    checks = [] of Int32
    order.each do |plan_slot|
      if lifecycle.stage_needed?(plan_slot)
        lifecycle.mark_stage_enqueued!(plan_slot)
        conversions << plan_slot % 2
      end
      lifecycle.mark_stage_checked!(plan_slot)
      checks << plan_slot
      lifecycle.mark_plan_encoded!(plan_slot)
    end
    conversions.should eq([0, 1])
    checks.should eq([0, 2, 1, 3])
    lifecycle.finish_request!(completed: true, wait_count: 1).should be_nil

    timing = PixelUMMFFNWeightReuseProbeSpecSupport.lifecycle_entry(0, "staged_f32", "timed")
    lifecycle = P::RequestLifecycle.new
    lifecycle.begin_request!(timing, 0, ids, ids)
    order.each do |plan_slot|
      if lifecycle.stage_needed?(plan_slot)
        lifecycle.mark_stage_enqueued!(plan_slot)
      end
      lifecycle.mark_plan_encoded!(plan_slot)
    end
    lifecycle.finish_request!(completed: true, wait_count: 1).should be_nil
  end

  it "rejects duplicate conversion, missing identity checks, timing checks and incomplete waits" do
    ids = ["window-0", "window-1"]
    correctness = PixelUMMFFNWeightReuseProbeSpecSupport.lifecycle_entry(0, "staged_f32", "correctness", capture_mode: true)
    lifecycle = P::RequestLifecycle.new
    lifecycle.begin_request!(correctness, 0, ids, ids)
    lifecycle.mark_stage_enqueued!(0)
    expect_raises(ArgumentError) { lifecycle.mark_stage_enqueued!(0) }
    expect_raises(ArgumentError) { lifecycle.mark_plan_encoded!(0) }
    lifecycle.mark_stage_checked!(0)
    lifecycle.mark_plan_encoded!(0)
    lifecycle.abort!
    lifecycle.release_allowed?.should be_false
    expect_raises(ArgumentError) do
      lifecycle.mark_cleanup_wait!(completed: false, wait_count: 1)
    end
    lifecycle.mark_cleanup_wait!(completed: true, wait_count: 1).should be_nil
    lifecycle.release_allowed?.should be_true

    timing = PixelUMMFFNWeightReuseProbeSpecSupport.lifecycle_entry(0, "staged_f32", "timed")
    timing_lifecycle = P::RequestLifecycle.new
    timing_lifecycle.begin_request!(timing, 0, ids, ids)
    timing_lifecycle.mark_stage_enqueued!(0)
    expect_raises(ArgumentError) { timing_lifecycle.mark_stage_checked!(0) }
    timing_lifecycle.abort!

    incomplete = P::RequestLifecycle.new
    incomplete.begin_request!(PixelUMMFFNWeightReuseProbeSpecSupport.lifecycle_entry(0, "baseline", "warmup"), 0, ids, ids)
    incomplete.mark_submission_attempt!
    4.times { |slot| incomplete.mark_plan_encoded!(slot.to_i32) }
    expect_raises(ArgumentError) { incomplete.finish_request!(completed: true, wait_count: 0) }
    incomplete.abort!
    incomplete.release_allowed?.should be_false
    expect_raises(ArgumentError) do
      incomplete.mark_cleanup_wait!(completed: true, wait_count: 0)
    end
  end
end
