require "spec"
require "../examples/support/pixelumm_window_batch_full_probe"

describe PixelUMMWindowBatchFullProbe do
  it "admits the exact 36-plan candidate inventory under a 24 GiB total cap" do
    PixelUMMWindowBatchFullProbe.candidate_named_bytes.should eq(931_136_064_i64)
    PixelUMMWindowBatchFullProbe::EXPECTED_BASELINE_NAMED_BYTES.should eq(24_108_145_128_i64)
    PixelUMMWindowBatchFullProbe::EXPECTED_TOTAL_NAMED_BYTES.should eq(25_039_281_192_i64)

    budget = PixelUMMWindowBatchFullProbe.admit_candidate_budget!(
      PixelUMMWindowBatchFullProbe::EXPECTED_BASELINE_NAMED_BYTES,
      PixelUMMWindowBatchFullProbe::MAX_NAMED_BYTES)
    budget.candidate_named_bytes.should eq(931_136_064_i64)
    budget.total_named_bytes.should eq(25_039_281_192_i64)
    budget.max_named_bytes.should eq(24_i64 * 1024 * 1024 * 1024)

    expect_raises(ArgumentError, /exactly 36 plans/) do
      PixelUMMWindowBatchFullProbe.candidate_named_bytes(35)
    end
    expect_raises(ArgumentError, /exceeds named budget/) do
      PixelUMMWindowBatchFullProbe.admit_candidate_budget!(
        PixelUMMWindowBatchFullProbe::EXPECTED_BASELINE_NAMED_BYTES,
        PixelUMMWindowBatchFullProbe::EXPECTED_TOTAL_NAMED_BYTES - 1)
    end
    expect_raises(ArgumentError, /baseline named inventory mismatch/) do
      PixelUMMWindowBatchFullProbe.admit_candidate_budget!(
        PixelUMMWindowBatchFullProbe::EXPECTED_BASELINE_NAMED_BYTES + 1,
        PixelUMMWindowBatchFullProbe::MAX_NAMED_BYTES)
    end
    expect_raises(ArgumentError, /at most 24 GiB/) do
      PixelUMMWindowBatchFullProbe.admit_candidate_budget!(
        PixelUMMWindowBatchFullProbe::EXPECTED_BASELINE_NAMED_BYTES,
        PixelUMMWindowBatchFullProbe::MAX_NAMED_BYTES + 1)
    end
  end

  it "rejects an invalid report, budget, or reverse mode before file creation" do
    expect_raises(ArgumentError, /absolute/) do
      PixelUMMWindowBatchFullProbe.validate_cli!("report.json",
        PixelUMMWindowBatchFullProbe::MAX_NAMED_BYTES, timed: false, reverse: false)
    end
    expect_raises(ArgumentError, /positive and at most 24 GiB/) do
      PixelUMMWindowBatchFullProbe.validate_cli!("/tmp/report.json", 0_i64,
        timed: false, reverse: false)
    end
    expect_raises(ArgumentError, /reverse requires --timed/) do
      PixelUMMWindowBatchFullProbe.validate_cli!("/tmp/report.json",
        PixelUMMWindowBatchFullProbe::MAX_NAMED_BYTES, timed: false, reverse: true)
    end
    PixelUMMWindowBatchFullProbe.validate_cli!("/tmp/report.json",
      PixelUMMWindowBatchFullProbe::MAX_NAMED_BYTES, timed: true, reverse: true)
  end

  it "runs four poisoned gates before the optional warm and ABBA/BAAB timed cohort" do
    correctness = PixelUMMWindowBatchFullProbe.schedule(timed: false)
    correctness.size.should eq(4)
    correctness.map(&.alternate).should eq([false, false, true, true])
    correctness.map(&.selected).should eq([true, false, false, true])
    correctness.all?(&.poison).should be_true
    correctness.all?(&.compare).should be_true
    correctness.none?(&.timing_eligible).should be_true

    timed = PixelUMMWindowBatchFullProbe.schedule(timed: true)
    timed.size.should eq(15)
    timed[0, 4].map(&.selected).should eq([true, false, false, true])
    timed[0, 4].map(&.alternate).should eq([false, false, true, true])
    timed[0, 4].all?(&.poison).should be_true
    timed[0, 4].all?(&.compare).should be_true

    timed[4, 2].map(&.selected).should eq([false, true])
    timed[4, 2].all?(&.alternate).should be_false
    timed[4, 2].none?(&.compare).should be_true
    timed[6, 8].map(&.selected).should eq([false, true, true, false, true, false, false, true])
    timed[6, 8].all?(&.timing_eligible).should be_true
    timed[6, 7].none?(&.compare).should be_true
    timed[13].compare.should be_true
    timed[13].timing_eligible.should be_true

    tail = timed.last
    tail.kind.should eq("final_candidate_tail")
    tail.alternate.should be_false
    tail.selected.should be_true
    tail.compare.should be_true
    tail.poison.should be_false
    tail.timing_eligible.should be_false
  end

  it "reverses gate, warm, and timed mode assignment but keeps the final candidate tail" do
    normal = PixelUMMWindowBatchFullProbe.schedule(timed: true)
    reverse = PixelUMMWindowBatchFullProbe.schedule(timed: true, reverse: true)
    reverse.map(&.selected).should eq(normal.map { |arm| !arm.selected }.tap do |modes|
      modes[-1] = true
    end)
    reverse.map(&.alternate).should eq(normal.map(&.alternate))
    reverse.map(&.poison).should eq(normal.map(&.poison))
    reverse.map(&.compare).should eq(normal.map(&.compare))
    reverse.map(&.timing_eligible).should eq(normal.map(&.timing_eligible))
    reverse[4, 2].map(&.selected).should eq([true, false])
    reverse[6, 8].map(&.selected).should eq([true, false, false, true, false, true, true, false])
    reverse.last.selected.should be_true
    expect_raises(ArgumentError, /requires timed mode/) do
      PixelUMMWindowBatchFullProbe.schedule(timed: false, reverse: true)
    end
  end
end
