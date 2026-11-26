require "spec"
require "./support/video_allocation_counter"
require "../src/ml/video/hy_nomas_continuation_history"

describe ML::Video::HyNomasContinuationHistory do
  it "accounts exact per-segment text rows for new prompts" do
    geometry = ML::Video::HyNomasContinuationHistory.geometry(
      3, 10, 120, 128, 54, 2, 2,
      segment_text_tokens: [80, 120, 60],
      segment_cache_text: [true, true, true])

    geometry.query_rows.should eq(130)
    geometry.storage_rows.should eq(138)
    geometry.workspace_geometries.map(&.history_rows).should eq([0, 90, 220])
    geometry.workspace_geometries.map(&.current_rows).should eq([90, 130, 70])
    geometry.workspace_geometries.map(&.key_rows).should eq([90, 220, 290])
    geometry.writer_workspace_geometries.map(&.current_rows).should eq([90, 130])
    geometry.max_history_rows.should eq(220)
    geometry.boundary_cache_elements(1).should eq(90 * 2 * 2)
    geometry.boundary_cache_elements(2).should eq(220 * 2 * 2)
    geometry.append_elements(0).should eq(90 * 2 * 2)
    geometry.append_elements(1).should eq(130 * 2 * 2)
    geometry.append_offset(1).should eq(90 * 2 * 2)
  end

  it "keeps a shared prompt cached once and exposes current text only to its writer" do
    geometry = ML::Video::HyNomasContinuationHistory.geometry(
      3, 10, 120, 128, 54, 2, 2,
      segment_text_tokens: [120, 120, 120],
      segment_cache_text: [true, false, false])

    geometry.workspace_geometries.map(&.history_rows).should eq([0, 130, 140])
    geometry.workspace_geometries.map(&.current_rows).should eq([130, 10, 10])
    geometry.writer_workspace_geometries.map(&.current_rows).should eq([130, 130])
    geometry.append_elements(0).should eq(130 * 2 * 2)
    geometry.append_elements(1).should eq(10 * 2 * 2)
    geometry.append_offset(1).should eq(130 * 2 * 2)
    geometry.workspace_geometries[2].history_rows.should eq(140)
    geometry.max_history_rows.should eq(140)
  end

  it "copies validated per-segment text geometry away from mutable callers" do
    text_tokens = [80, 120, 60]
    cache_text = [true, true, true]
    geometry = ML::Video::HyNomasContinuationHistory.geometry(
      3, 10, 120, 128, 54, 2, 2,
      segment_text_tokens: text_tokens,
      segment_cache_text: cache_text)

    text_tokens[0] = 1
    cache_text[1] = false
    geometry.segment_text_tokens.should eq([80, 120, 60])
    geometry.segment_cache_text.should eq([true, true, true])
    geometry.append_elements(0).should eq(90 * 2 * 2)

    returned_text_tokens = geometry.segment_text_tokens
    returned_cache_text = geometry.segment_cache_text
    returned_text_tokens[0] = 1
    returned_cache_text[0] = false
    geometry.append_elements(0).should eq(90 * 2 * 2)
    geometry.segment_text_tokens.should eq([80, 120, 60])
  end

  it "rejects malformed prompt history before allocation" do
    allocations = ML::MetalBuffer.video_spec_allocations
    invalid = [
      {tokens: [80, 120], cache: [true, true]},
      {tokens: [80, 120, 60], cache: [true, true]},
      {tokens: [80, 121, 60], cache: [true, true, true]},
      {tokens: [80, 0, 60], cache: [true, true, true]},
      {tokens: [80, 120, 60], cache: [false, true, true]},
      {tokens: [80, 120, 60], cache: [true, false, true]},
      {tokens: [80, 119, 60], cache: [true, false, true]},
    ]
    invalid.each do |profile|
      expect_raises(ArgumentError) do
        ML::Video::HyNomasContinuationHistory.geometry(
          3, 10, 120, 128, 54, 2, 2,
          segment_text_tokens: profile[:tokens],
          segment_cache_text: profile[:cache])
      end
    end
    ML::MetalBuffer.video_spec_allocations.should eq(allocations)
  end
end
