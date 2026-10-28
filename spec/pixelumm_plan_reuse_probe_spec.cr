require "spec"
require "../examples/support/pixelumm_plan_reuse_probe"

describe PixelUMMPlanReuseProbe do
  it "admits softmax/PV mode only for the two-Plan unshared comparison before Metal setup" do
    PixelUMMPlanReuseProbe.validate_softmax_pv_configuration!(2, false, true).should be_nil
    PixelUMMPlanReuseProbe.validate_softmax_pv_configuration!(1, false, false).should be_nil
    expect_raises(ArgumentError) do
      PixelUMMPlanReuseProbe.validate_softmax_pv_configuration!(1, false, true)
    end
    expect_raises(ArgumentError) do
      PixelUMMPlanReuseProbe.validate_softmax_pv_configuration!(2, true, true)
    end

    PixelUMMPlanReuseProbe.fused_softmax_pv_for_plan!(true, 2, false, 0).should be_false
    PixelUMMPlanReuseProbe.fused_softmax_pv_for_plan!(true, 2, false, 1).should be_true
    PixelUMMPlanReuseProbe.fused_softmax_pv_for_plan!(false, 1, false, 0).should be_false
    expect_raises(ArgumentError) do
      PixelUMMPlanReuseProbe.fused_softmax_pv_for_plan!(true, 2, false, 2)
    end
  end

  it "admits the direct F32 check only inside the two-Plan softmax/PV experiment" do
    PixelUMMPlanReuseProbe.validate_softmax_pv_f32_configuration!(true, true, 2, false).should be_nil
    PixelUMMPlanReuseProbe.validate_softmax_pv_f32_configuration!(false, false, 2, false).should be_nil
    expect_raises(ArgumentError) do
      PixelUMMPlanReuseProbe.validate_softmax_pv_f32_configuration!(false, true, 2, false)
    end
    expect_raises(ArgumentError) do
      PixelUMMPlanReuseProbe.validate_softmax_pv_f32_configuration!(true, true, 1, false)
    end
    expect_raises(ArgumentError) do
      PixelUMMPlanReuseProbe.validate_softmax_pv_f32_configuration!(true, true, 2, true)
    end
  end

  it "keeps the connected named-buffer budget independent of resident Plan count" do
    small_one = PixelUMMPlanReuseProbe.validate_budget!(2048, 1,
      PixelUMMConnectedResourceProbe::MAX_NAMED_BYTES)
    small_two = PixelUMMPlanReuseProbe.validate_budget!(2048, 2,
      PixelUMMConnectedResourceProbe::MAX_NAMED_BYTES)
    small_one.should eq(small_two)
    small_one[:total_bytes].should eq(1_098_228_484_i64)

    full_one = PixelUMMPlanReuseProbe.validate_budget!(21120, 1,
      PixelUMMConnectedResourceProbe::MAX_NAMED_BYTES)
    full_two = PixelUMMPlanReuseProbe.validate_budget!(21120, 2,
      PixelUMMConnectedResourceProbe::MAX_NAMED_BYTES)
    full_one.should eq(full_two)
    full_one[:total_bytes].should eq(7_375_674_980_i64)
    full_two[:total_bytes].should be <= PixelUMMConnectedResourceProbe::MAX_NAMED_BYTES

    expect_raises(ArgumentError) do
      PixelUMMPlanReuseProbe.validate_budget!(21120, 2, full_two[:total_bytes] - 1)
    end
    expect_raises(ArgumentError) do
      PixelUMMPlanReuseProbe.validate_budget!(21120, 2,
        PixelUMMConnectedResourceProbe::MAX_NAMED_BYTES + 1)
    end
    expect_raises(ArgumentError) { PixelUMMPlanReuseProbe.validate_budget!(21120, 0, 8_i64 * PixelUMMPlanReuseProbe::GIB) }
    expect_raises(ArgumentError) { PixelUMMPlanReuseProbe.validate_budget!(21120, 3, 8_i64 * PixelUMMPlanReuseProbe::GIB) }
  end

  it "preflights F32 panels and the 176-byte eleven-slot metrics read before Metal" do
    small = PixelUMMPlanReuseProbe.validate_budget!(2048, 2,
      PixelUMMConnectedResourceProbe::MAX_NAMED_BYTES, true)
    small[:metric_bytes].should eq(176_i64)
    small[:total_bytes].should eq(1_119_200_052_i64)

    full = PixelUMMPlanReuseProbe.validate_budget!(21120, 2,
      PixelUMMConnectedResourceProbe::MAX_NAMED_BYTES, true)
    full[:metric_bytes].should eq(176_i64)
    full[:total_bytes].should eq(7_551_835_796_i64)
    full[:total_bytes].should be <= 8_i64 * PixelUMMPlanReuseProbe::GIB
    expect_raises(ArgumentError) do
      PixelUMMPlanReuseProbe.validate_budget!(21120, 2, full[:total_bytes] - 1, true)
    end
    expect_raises(ArgumentError) do
      PixelUMMPlanReuseProbe.validate_budget!(21120, 2,
        PixelUMMConnectedResourceProbe::MAX_NAMED_BYTES + 1, true)
    end
  end

  it "derives bounded per-tile F32 coverage and checks both aggregates plus the NaN positive control" do
    small = PixelUMMPlanReuseProbe.softmax_pv_f32_geometry(2048)
    small[:tiles].size.should eq(16)
    small[:probabilities_snapshot_bytes].should eq(18_874_368_i64)
    small[:attended_snapshot_bytes].should eq(2_097_152_i64)
    small[:probability_elements].should eq(75_497_472_u32)
    small[:attended_elements].should eq(8_388_608_u32)
    PixelUMMPlanReuseProbe.validate_f32_snapshot_sizes!(small,
      18_874_368_i64, 2_097_152_i64).should be_nil
    expect_raises(ArgumentError) do
      PixelUMMPlanReuseProbe.validate_f32_snapshot_sizes!(small,
        18_874_367_i64, 2_097_152_i64)
    end

    full = PixelUMMPlanReuseProbe.softmax_pv_f32_geometry(21120)
    full[:tiles].size.should eq(60)
    full[:probabilities_snapshot_bytes].should eq(169_869_312_i64)
    full[:attended_snapshot_bytes].should eq(6_291_456_i64)
    full[:probability_elements].should eq(2_229_534_720_u32)
    full[:attended_elements].should eq(86_507_520_u32)
    full[:tiles].map(&.[:query_tokens]).sum.should eq(21_120)

    metrics = Array.new(PixelUMMPlanReuseProbe::F32_CHECK_METRIC_SLOTS) do
      {failures: 0_u32, different_words: 0_u32, nonfinite: 0_u32, elements: 0_u32}
    end
    # Tolerance-qualified numeric differences are reported without imposing
    # an accidental bitwise-equality requirement on the F32 comparator.
    metrics[8] = {failures: 0_u32, different_words: 17_u32, nonfinite: 0_u32,
                  elements: full[:probability_elements]}
    metrics[9] = {failures: 0_u32, different_words: 0_u32, nonfinite: 0_u32,
                  elements: full[:attended_elements]}
    metrics[10] = {failures: 1_u32, different_words: 1_u32, nonfinite: 1_u32,
                   elements: 1_u32}
    PixelUMMPlanReuseProbe.validate_softmax_pv_f32_metrics!(metrics, full).should be_nil

    expect_raises(ArgumentError) do
      PixelUMMPlanReuseProbe.validate_softmax_pv_f32_metrics!(metrics[0...10], full)
    end
    wrong_coverage = metrics.dup
    wrong_coverage[8] = wrong_coverage[8].merge({elements: full[:probability_elements] - 1_u32})
    expect_raises(ArgumentError) do
      PixelUMMPlanReuseProbe.validate_softmax_pv_f32_metrics!(wrong_coverage, full)
    end
    wrong_finite = metrics.dup
    wrong_finite[9] = wrong_finite[9].merge({nonfinite: 1_u32})
    expect_raises(ArgumentError) do
      PixelUMMPlanReuseProbe.validate_softmax_pv_f32_metrics!(wrong_finite, full)
    end
    broken_positive_control = metrics.dup
    broken_positive_control[10] = broken_positive_control[10].merge({different_words: 0_u32})
    expect_raises(ArgumentError) do
      PixelUMMPlanReuseProbe.validate_softmax_pv_f32_metrics!(broken_positive_control, full)
    end
  end

  it "requires F32 snapshot bindings to be distinct from every other owner binding" do
    base = {"fixture.input.x" => 10_u64, "comparator.metrics" => 20_u64}
    good = base.merge({
      "comparator.local_probabilities_f32_snapshot" => 30_u64,
      "comparator.local_attended_f32_snapshot"      => 40_u64,
    })
    PixelUMMPlanReuseProbe.validate_f32_snapshot_identity!(good).should be_nil
    expect_raises(ArgumentError) do
      PixelUMMPlanReuseProbe.validate_f32_snapshot_identity!(base)
    end
    expect_raises(ArgumentError) do
      PixelUMMPlanReuseProbe.validate_f32_snapshot_identity!(base.merge({
        "comparator.local_probabilities_f32_snapshot" => 30_u64,
        "comparator.local_attended_f32_snapshot"      => 30_u64,
      }))
    end
    expect_raises(ArgumentError) do
      PixelUMMPlanReuseProbe.validate_f32_snapshot_identity!(base.merge({
        "comparator.local_probabilities_f32_snapshot" => 10_u64,
        "comparator.local_attended_f32_snapshot"      => 40_u64,
      }))
    end
  end

  it "validates the connected fixture geometry and shared binding identity" do
    PixelUMMPlanReuseProbe.validate_geometry!([6, 44, 80], 21120).should be_nil
    expect_raises(ArgumentError) do
      PixelUMMPlanReuseProbe.validate_geometry!([6, 44, 79], 21120)
    end

    first = {"input" => 100_u64, "capture" => 200_u64, "scratch" => 300_u64}
    PixelUMMPlanReuseProbe.validate_shared_bindings!([first, first.dup]).should be_nil
    expect_raises(ArgumentError) do
      PixelUMMPlanReuseProbe.validate_shared_bindings!([first, {"input" => 101_u64, "capture" => 200_u64, "scratch" => 300_u64}])
    end
    expect_raises(ArgumentError) do
      PixelUMMPlanReuseProbe.validate_shared_bindings!([first, {"input" => 100_u64, "capture" => 200_u64}])
    end
  end

  it "requires distinct non-null native Plan handles" do
    PixelUMMPlanReuseProbe.validate_distinct_plan_handles!([101_u64], 1).should be_nil
    PixelUMMPlanReuseProbe.validate_distinct_plan_handles!([101_u64, 202_u64], 2).should be_nil
    expect_raises(ArgumentError) do
      PixelUMMPlanReuseProbe.validate_distinct_plan_handles!([101_u64, 101_u64], 2)
    end
    expect_raises(ArgumentError) do
      PixelUMMPlanReuseProbe.validate_distinct_plan_handles!([0_u64, 202_u64], 2)
    end
    expect_raises(ArgumentError) do
      PixelUMMPlanReuseProbe.validate_distinct_plan_handles!([101_u64], 2)
    end
  end

  it "preserves the native Plan's fixed 13/9/26/14 binding ABI" do
    PixelUMMPlanReuseProbe.validate_native_binding_widths!(13, 9, 26, 14).should be_nil
    expect_raises(ArgumentError) do
      PixelUMMPlanReuseProbe.validate_native_binding_widths!(13, 9, 7, 14)
    end
    expect_raises(ArgumentError) do
      PixelUMMPlanReuseProbe.validate_native_binding_widths!(13, 9, 25, 14)
    end
  end

  it "schedules six requests with two warmups and balanced two-Plan ABBA measurements" do
    one = PixelUMMPlanReuseProbe.schedule(1, reverse_warmups: true, reverse_first: true)
    one.size.should eq(6)
    one.map(&.[:index]).should eq([0, 1, 2, 3, 4, 5])
    one.map(&.[:plan_index]).should eq([0, 0, 0, 0, 0, 0])
    one.map(&.[:warmup]).should eq([true, true, false, false, false, false])

    two = PixelUMMPlanReuseProbe.schedule(2, reverse_warmups: false, reverse_first: false)
    two.map(&.[:plan_index]).should eq([0, 1, 0, 1, 1, 0])
    reverse = PixelUMMPlanReuseProbe.schedule(2, reverse_warmups: true, reverse_first: true)
    reverse.map(&.[:plan_index]).should eq([1, 0, 1, 0, 0, 1])

    duplicate_selection = two.dup
    duplicate_selection[1] = {index: 1_i32, plan_index: 0_i32, warmup: true}
    expect_raises(ArgumentError) do
      PixelUMMPlanReuseProbe.validate_schedule!(duplicate_selection, 2)
    end
    bad_rows = two.dup
    bad_rows.pop
    expect_raises(ArgumentError) { PixelUMMPlanReuseProbe.validate_schedule!(bad_rows, 2) }
    invalid_plan = two.dup
    invalid_plan[4] = {index: 4_i32, plan_index: 2_i32, warmup: false}
    expect_raises(ArgumentError) { PixelUMMPlanReuseProbe.validate_schedule!(invalid_plan, 2) }
  end

  it "rejects concurrent, repeated, or invalid Plan selection and guards partial submissions" do
    lifecycle = PixelUMMPlanReuseProbe::SubmissionLifecycle.new
    lifecycle.release_allowed?.should be_true
    expect_raises(ArgumentError) { lifecycle.begin_request!(2, 2) }

    lifecycle.begin_request!(0, 2)
    lifecycle.release_allowed?.should be_true # No GPU submission means teardown remains legal.
    expect_raises(ArgumentError) { lifecycle.begin_request!(1, 2) }
    lifecycle.mark_attempt!
    lifecycle.release_allowed?.should be_false
    expect_raises(ArgumentError) { lifecycle.begin_request!(1, 2) }
    expect_raises(ArgumentError) { lifecycle.require_selected_plan!(1) }
    expect_raises(ArgumentError) { lifecycle.require_readable!(completed: true, wait_count: 1) }
    expect_raises(ArgumentError) { lifecycle.mark_chain_released! }

    lifecycle.mark_plan_encoded!(0)
    expect_raises(ArgumentError) { lifecycle.mark_plan_encoded!(0) }
    lifecycle.mark_comparisons_encoded!
    expect_raises(ArgumentError) { lifecycle.mark_comparisons_encoded! }

    expect_raises(ArgumentError) { lifecycle.mark_terminal_wait!(completed: true, wait_count: 0) }
    expect_raises(ArgumentError) { lifecycle.mark_terminal_wait!(completed: true, wait_count: 2) }
    lifecycle.release_allowed?.should be_false
    lifecycle.quarantined?.should be_true
    expect_raises(ArgumentError) { lifecycle.mark_terminal_wait!(completed: true, wait_count: 1) }
    expect_raises(ArgumentError) { lifecycle.begin_request!(1, 2) }
  end

  it "permits reuse only after the sole successful wait and chain release" do
    lifecycle = PixelUMMPlanReuseProbe::SubmissionLifecycle.new
    lifecycle.begin_request!(0, 2)
    lifecycle.mark_attempt!
    lifecycle.mark_plan_encoded!(0)
    lifecycle.mark_comparisons_encoded!
    expect_raises(ArgumentError) { lifecycle.require_readable!(completed: true, wait_count: 1) }
    lifecycle.mark_terminal_wait!(completed: true, wait_count: 1)
    lifecycle.require_readable!(completed: true, wait_count: 1).should be_nil
    expect_raises(ArgumentError) { lifecycle.require_readable!(completed: false, wait_count: 1) }
    expect_raises(ArgumentError) { lifecycle.require_readable!(completed: true, wait_count: 2) }
    lifecycle.release_allowed?.should be_true
    expect_raises(ArgumentError) { lifecycle.begin_request!(1, 2) }
    lifecycle.mark_chain_released!
    lifecycle.begin_request!(1, 2)
    lifecycle.mark_attempt!
    lifecycle.abort!
    lifecycle.mark_terminal_wait!(completed: true, wait_count: 1)
    lifecycle.release_allowed?.should be_true
    lifecycle.quarantined?.should be_true
    lifecycle.mark_chain_released!
    expect_raises(ArgumentError) { lifecycle.begin_request!(0, 2) }
  end

  it "quarantines a request whose sole terminal wait did not complete" do
    lifecycle = PixelUMMPlanReuseProbe::SubmissionLifecycle.new
    lifecycle.begin_request!(0, 1)
    lifecycle.mark_attempt!
    expect_raises(ArgumentError) { lifecycle.mark_terminal_wait!(completed: false, wait_count: 1) }
    lifecycle.release_allowed?.should be_false
    lifecycle.quarantined?.should be_true
    expect_raises(ArgumentError) { lifecycle.mark_terminal_wait!(completed: true, wait_count: 1) }
    expect_raises(ArgumentError) { lifecycle.begin_request!(0, 1) }
  end
end
