require "spec"
require "./support/video_allocation_counter"
require "../src/ml/metal/device"
require "../src/ml/video/hy_nomas_history_attention"

describe ML::Video::HyNomasHistoryAttentionWorkspace do
  it "accounts for per-row labels and one query-tile mask" do
    allocations = ML::MetalBuffer.video_spec_allocations
    baseline = ML::Video::HyNomasHistoryAttentionWorkspace.geometry(
      2, 5, 2, 4, current_rows: 3)
    masked = ML::Video::HyNomasHistoryAttentionWorkspace.geometry(
      2, 5, 2, 4, current_rows: 3,
      query_labels: [0, 1, 1, -1, 2],
      key_labels: [0, 1, -1, 2, 0])

    masked.masked.should be_true
    masked.mask_elements.should eq(5 * 5)
    masked.query_label_bytes.should eq(5_i64 * sizeof(Int32))
    masked.key_label_bytes.should eq(5_i64 * sizeof(Int32))
    masked.action_mask_bytes.should eq(5_i64 * 5 * sizeof(Float32) + 10_i64 * sizeof(Int32))
    masked.workspace_bytes.should eq(baseline.workspace_bytes + masked.action_mask_bytes)
    ML::MetalBuffer.video_spec_allocations.should eq(allocations)
  end

  it "leaves the unmasked allocation geometry unchanged" do
    unmasked = ML::Video::HyNomasHistoryAttentionWorkspace.geometry(23, 513, 16, 128, 7)

    unmasked.masked.should be_false
    unmasked.mask_elements.should eq(0)
    unmasked.action_mask_bytes.should eq(0)
    unmasked.workspace_bytes.should eq(
      2_i64 * (256_i64 * 16 * 128 + 30_i64 * 16 * 128) * sizeof(Float32))
  end

  it "rejects partial, malformed, or all-hidden active-query labels before allocation" do
    allocations = ML::MetalBuffer.video_spec_allocations
    valid_query_labels = [1, 2]
    valid_key_labels = [1, 2, -1]
    invalid_labels = [
      {nil, valid_key_labels},
      {valid_query_labels, nil},
      {[1], valid_key_labels},
      {valid_query_labels, [1, 2]},
      {[0, -2], valid_key_labels},
      {valid_query_labels, [1, -2, -1]},
      {[1, -1], [-1, -1, -1]},
      {[0, 1], [1, 2, -1]},
    ]
    invalid_labels.each do |query_labels, key_labels|
      expect_raises(ArgumentError) do
        ML::Video::HyNomasHistoryAttentionWorkspace.geometry(
          1, 2, 1, 2, current_rows: 2,
          query_labels: query_labels, key_labels: key_labels)
      end
    end
    ML::MetalBuffer.video_spec_allocations.should eq(allocations)
  end

  it "allows padded queries when no active query needs a visible key" do
    masked = ML::Video::HyNomasHistoryAttentionWorkspace.geometry(
      2, 2, 1, 2, current_rows: 1,
      query_labels: [-1, -1], key_labels: [-1, -1, -1])

    masked.masked.should be_true
    masked.action_mask_bytes.should be > 0
  end

  it "allows an action query to use only its own action keys" do
    masked = ML::Video::HyNomasHistoryAttentionWorkspace.geometry(
      0, 2, 1, 2, query_labels: [3, 4], key_labels: [3, 4])

    masked.masked.should be_true
    masked.action_mask_bytes.should be > 0
  end

  it "bounds masks across a full 256-row tile and a two-row tail" do
    query_labels = Array(Int32).new(256, 1) << -1 << 2
    key_labels = [0, 1, -1, 2, 2]
    masked = ML::Video::HyNomasHistoryAttentionWorkspace.geometry(
      3, 258, 1, 2, current_rows: 2,
      query_labels: query_labels, key_labels: key_labels)

    masked.query_chunk.should eq(256)
    masked.graph_calls.should eq(2)
    masked.mask_elements.should eq(256 * 5)
    masked.query_label_bytes.should eq(258_i64 * sizeof(Int32))
    masked.key_label_bytes.should eq(5_i64 * sizeof(Int32))
    masked.action_mask_bytes.should eq(256_i64 * 5 * sizeof(Float32) +
                                       263_i64 * sizeof(Int32))
  end
end
