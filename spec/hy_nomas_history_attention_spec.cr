require "spec"
require "./support/video_allocation_counter"
require "../src/ml/metal/device"
require "../src/ml/video/hy_nomas_history_attention"
require "../examples/support/hy_nomas_history_attention_probe_guard"

describe ML::Video::HyNomasHistoryAttentionWorkspace do
  it "admits a no-history square control without allocating" do
    square = ML::Video::HyNomasHistoryAttentionWorkspace.geometry(0, 23, 16, 128)

    square.no_history_control.should be_true
    square.history_rows.should eq(0)
    square.query_rows.should eq(23)
    square.current_rows.should eq(23)
    square.key_rows.should eq(23)
    square.query_chunk.should eq(23)
    square.graph_calls.should eq(1)
    square.query_elements.should eq(23 * 16 * 128)
    square.current_elements.should eq(23 * 16 * 128)
    square.history_elements.should eq(0)
    square.key_elements.should eq(23 * 16 * 128)
    square.max_score_elements.should eq(16_i64 * 23 * 23)
    square.max_score_bytes.should eq(16_i64 * 23 * 23 * 4)
    square.workspace_bytes.should eq(4_i64 * 23 * 16 * 128 * 4)
    square.active_history_pair_bytes.should eq(0)
    square.current_io_bytes.should eq(4_i64 * 23 * 16 * 128 * 4)
  end

  it "bounds rectangular-history scores by one 256-row all-head tile" do
    continuation = ML::Video::HyNomasHistoryAttentionWorkspace.geometry(
      23, 513, 16, 128, current_rows: 7)
    compatible_default = ML::Video::HyNomasHistoryAttentionWorkspace.geometry(23, 513, 16, 128)

    continuation.no_history_control.should be_false
    continuation.history_rows.should eq(23)
    continuation.query_rows.should eq(513)
    continuation.current_rows.should eq(7)
    continuation.key_rows.should eq(30)
    continuation.query_chunk.should eq(256)
    continuation.graph_calls.should eq(3)
    continuation.query_elements.should eq(513 * 16 * 128)
    continuation.history_elements.should eq(23 * 16 * 128)
    continuation.current_elements.should eq(7 * 16 * 128)
    continuation.key_elements.should eq(30 * 16 * 128)
    continuation.max_score_elements.should eq(16_i64 * 256 * 30)
    continuation.max_score_bytes.should eq(16_i64 * 256 * 30 * 4)
    continuation.workspace_bytes.should eq(
      2_i64 * (256_i64 * 16 * 128 + 30_i64 * 16 * 128) * 4)
    continuation.active_history_pair_bytes.should eq(2_i64 * 23 * 16 * 128 * 4)
    continuation.current_io_bytes.should eq(
      2_i64 * (513_i64 * 16 * 128 + 7_i64 * 16 * 128) * 4)

    compatible_default.current_rows.should eq(513)
    compatible_default.key_rows.should eq(536)
    compatible_default.current_elements.should eq(compatible_default.query_elements)
    compatible_default.current_io_bytes.should eq(4_i64 * 513 * 16 * 128 * 4)
  end

  it "allows a rectangular zero-history attention shape without calling it the square control" do
    rectangular = ML::Video::HyNomasHistoryAttentionWorkspace.geometry(0, 9, 4, 8, 3)

    rectangular.no_history_control.should be_false
    rectangular.history_rows.should eq(0)
    rectangular.query_rows.should eq(9)
    rectangular.current_rows.should eq(3)
    rectangular.key_rows.should eq(3)
    rectangular.current_elements.should eq(3 * 4 * 8)
    rectangular.current_io_bytes.should eq(2_i64 * (9_i64 * 4 * 8 + 3_i64 * 4 * 8) * 4)
  end

  it "rejects malformed or native-limit geometry before allocating" do
    allocations = ML::MetalBuffer.video_spec_allocations
    [{-1, 1, 1, 1}, {0, 0, 1, 1}, {0, 1, 0, 1}, {0, 1, 1, 0},
     {Int32::MAX, 1, 1, 1}, {0, Int32::MAX, 2, 1},
     {0, 1, Int32::MAX, 2}].each do |invalid|
      expect_raises(ArgumentError) do
        ML::Video::HyNomasHistoryAttentionWorkspace.geometry(*invalid)
      end
    end
    [{0, 4, 1, 1, 0}, {0, 4, 1, 1, 5}, {Int32::MAX, 4, 1, 1, 1}].each do |invalid|
      expect_raises(ArgumentError) do
        ML::Video::HyNomasHistoryAttentionWorkspace.geometry(*invalid)
      end
    end
    ML::MetalBuffer.video_spec_allocations.should eq(allocations)
  end
end

describe "HyNomasHistoryAttentionProbe release-guard classifier" do
  it "recognizes both live plan guard messages only before chain completion" do
    messages = [
      "Complete Hy-Nomas history attention chain before releasing plan",
      "Complete compact Hy-Nomas MPS attention chain before release",
    ]
    messages.each do |message|
      HyNomasHistoryAttentionProbe.inflight_chain_release_rejected?(false, message).should be_true
      HyNomasHistoryAttentionProbe.inflight_chain_release_rejected?(true, message).should be_false
    end
  end

  it "does not count unrelated argument errors as release-guard rejections" do
    HyNomasHistoryAttentionProbe.inflight_chain_release_rejected?(false,
      "History attention plan has been released").should be_false
    HyNomasHistoryAttentionProbe.inflight_chain_release_rejected?(false, nil).should be_false
  end
end
