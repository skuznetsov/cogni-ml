require "spec"
require "./support/video_allocation_counter"
require "../src/ml/video/hy_nomas_continuation_history"

describe ML::Video::HyNomasContinuationHistory do
  it "admits immutable per-segment labels with the writer sharing the caption plan" do
    q0 = [0, 0, 0, 0, 0, -1]
    k0 = [0, 0, 0, 0, 0]
    q1 = [0, 1, 1, 0, 0, 1]
    k1 = Array(Int32).new(5, 0) + [0, 0, 0, 0, 0, 1]
    labels = [ML::Video::HyNomasContinuationHistory::AttentionLabels.new(q0, k0),
              ML::Video::HyNomasContinuationHistory::AttentionLabels.new(q1, k1)]
    plain = ML::Video::HyNomasContinuationHistory.geometry(2, 4, 2, 2, 54, 1, 2,
      segment_text_tokens: [1, 2], segment_cache_text: [true, true])
    masked = ML::Video::HyNomasContinuationHistory.geometry(2, 4, 2, 2, 54, 1, 2,
      segment_text_tokens: [1, 2], segment_cache_text: [true, true],
      segment_attention_labels: labels)
    masked.workspace_geometries.all?(&.masked).should be_true
    masked.writer_workspace_geometries.first.should eq(masked.workspace_geometries.first)
    masked.named_bytes.should eq(plain.named_bytes + masked.workspace_geometries.sum(&.action_mask_bytes))
    q0[0] = 99
    k1[0] = -1
    labels.clear
    stored = masked.segment_attention_labels.not_nil!
    stored.size.should eq(2)
    stored[0].query[0].should eq(0)
    stored[1].key[0].should eq(0)
    copy = stored[0].query
    copy[0] = -1
    stored[0].query[0].should eq(0)
  end

  it "accounts for BF16 masked attention scratch without changing persistent history" do
    label = ML::Video::HyNomasContinuationHistory::AttentionLabels
    labels = [label.new([0, 0, 0, -1], [0, 0, 0]),
              label.new([0, 1, 0, 1], [0, 0, 0, 0, 0, 0, 1])]
    f32 = ML::Video::HyNomasContinuationHistory.geometry(2, 2, 2, 2, 54, 1, 2,
      segment_text_tokens: [1, 2], segment_cache_text: [true, true],
      segment_attention_labels: labels)
    bf16 = ML::Video::HyNomasContinuationHistory.geometry(2, 2, 2, 2, 54, 1, 2,
      segment_text_tokens: [1, 2], segment_cache_text: [true, true],
      segment_attention_labels: labels, source_bf16_attention: true)

    bf16.cache_bytes.should eq(f32.cache_bytes)
    bf16.workspace_geometries.all?(&.source_bf16_attention).should be_true
    bf16.writer_workspace_geometries.first.should eq(bf16.workspace_geometries.first)
    bf16.workspace_geometries.zip(f32.workspace_geometries).each do |source, stock|
      labels_bytes = stock.query_label_bytes + stock.key_label_bytes
      source.workspace_bytes.should eq((stock.workspace_bytes - labels_bytes) // 2 + labels_bytes)
      source.action_mask_bytes.should eq((stock.action_mask_bytes - labels_bytes) // 2 + labels_bytes)
    end
  end

  it "rejects label profiles requiring an unimplemented distinct masked writer" do
    label = ML::Video::HyNomasContinuationHistory::AttentionLabels.new(
      Array(Int32).new(6, 0), Array(Int32).new(6, 0))
    expect_raises(ArgumentError, /one profile/) do
      ML::Video::HyNomasContinuationHistory.geometry(2, 4, 2, 2, 54, 1, 2,
        segment_attention_labels: [label])
    end
    expect_raises(ArgumentError, /cached current text/) do
      ML::Video::HyNomasContinuationHistory.geometry(2, 4, 2, 2, 54, 1, 2,
        segment_text_tokens: [2, 2], segment_cache_text: [true, false],
        segment_attention_labels: [label, label])
    end
  end

  it "restores only a fresh schedule at an extendable post-writer boundary" do
    schedule = ML::Video::HyNomasContinuationHistorySchedule.new(3, 2)
    [-1, 0, 3, 4].each do |invalid|
      expect_raises(ArgumentError) { schedule.restore_boundary!(invalid) }
    end
    schedule.next_segment.should eq(0)
    schedule.restore_boundary!(1)
    schedule.next_segment.should eq(1)
    schedule.denoise_passes.should eq(0)
    expect_raises(ArgumentError) { schedule.restore_boundary!(2) }
    expect_raises(ArgumentError) { schedule.select_segment!(0, false) }
    4.times do
      schedule.select_segment!(1, false)
      2.times { |layer| schedule.finish_layer!(layer) }
    end
    schedule.select_segment!(1, true)
    2.times { |layer| schedule.finish_layer!(layer) }
    schedule.next_segment.should eq(2)
  end

  it "rejects restoring a partially encoded history schedule" do
    schedule = ML::Video::HyNomasContinuationHistorySchedule.new(2, 2)
    schedule.select_segment!(0, false)
    expect_raises(ArgumentError) { schedule.restore_boundary!(1) }
    2.times { |layer| schedule.finish_layer!(layer) }
    expect_raises(ArgumentError) { schedule.restore_boundary!(1) }
  end

  it "keeps one text block for later chunks of the same scene" do
    allocations = ML::MetalBuffer.video_spec_allocations
    geometry = ML::Video::HyNomasContinuationHistory.geometry(
      4, 4320, 557, 576, 54, 16, 128,
      text_cache_policy: ML::Video::HyNomasContinuationHistory::TextCachePolicy::OncePerScene)

    geometry.workspace_geometries.map(&.history_rows).should eq([0, 4877, 9197, 13517])
    geometry.workspace_geometries.map(&.key_rows).should eq([4877, 9197, 13517, 17837])
    geometry.workspace_geometries.map(&.current_rows).should eq([4877, 4320, 4320, 4320])
    geometry.writer_workspace_geometries.map(&.key_rows).should eq([4877, 9754, 14074])
    geometry.writer_workspace_geometries.map(&.current_rows).should eq([4877, 4877, 4877])
    geometry.max_history_rows.should eq(13517)
    geometry.cache_bytes.should eq(13517_i64 * 16 * 128 * 4 * 2 * 54)
    geometry.append_elements(0).should eq(4877 * 16 * 128)
    geometry.append_elements(1).should eq(4320 * 16 * 128)
    geometry.append_offset(2).should eq(9197 * 16 * 128)
    [-1, 3, 4].each do |invalid|
      expect_raises(ArgumentError) { geometry.append_elements(invalid) }
      expect_raises(ArgumentError) { geometry.append_offset(invalid) }
    end
    # The first writer shares the square denoise plan; later writers need
    # transient current-text keys even though only video keys are persisted.
    expected_workspace = geometry.workspace_geometries.sum(&.workspace_bytes) +
                         geometry.writer_workspace_geometries.skip(1).sum(&.workspace_bytes)
    geometry.workspace_bytes.should eq(expected_workspace)
    geometry.named_bytes.should eq(geometry.cache_bytes + expected_workspace)
    ML::MetalBuffer.video_spec_allocations.should eq(allocations)
  end

  it "admits compact active rows separately from padded storage rows" do
    allocations = ML::MetalBuffer.video_spec_allocations
    geometry = ML::Video::HyNomasContinuationHistory.geometry(
      6, 4320, 354, 384, 54, 16, 128)

    geometry.query_rows.should eq(4674)
    geometry.storage_rows.should eq(4704)
    geometry.denoise_steps.should eq(4)
    geometry.query_elements.should eq(4674 * 16 * 128)
    geometry.storage_elements.should eq(4704 * 16 * 128)
    geometry.max_history_rows.should eq(5 * 4674)
    geometry.cache_elements_per_buffer.should eq(5 * 4674 * 16 * 128)
    geometry.workspace_geometries.size.should eq(6)
    geometry.workspace_geometries[0].history_rows.should eq(0)
    geometry.workspace_geometries[0].key_rows.should eq(4674)
    geometry.workspace_geometries[1].history_rows.should eq(4674)
    geometry.workspace_geometries[1].key_rows.should eq(2 * 4674)
    geometry.workspace_geometries[5].history_rows.should eq(5 * 4674)
    geometry.workspace_geometries[5].key_rows.should eq(6 * 4674)
    geometry.cache_bytes.should eq(5_i64 * 4674 * 16 * 128 * 4 * 2 * 54)
    geometry.named_bytes.should eq(geometry.cache_bytes + geometry.workspace_bytes)
    ML::Video::HyNomasContinuationHistory.named_bytes(
      6, 4320, 354, 384, 54, 16, 128).should eq(geometry.named_bytes)
    ML::Video::HyNomasContinuationHistory.named_bytes(segment_count: 6,
      video_tokens: 4320, valid_text_tokens: 354, storage_text_tokens: 384,
      layers: 54, heads: 16, head_dim: 128).should eq(geometry.named_bytes)
    ML::MetalBuffer.video_spec_allocations.should eq(allocations)
  end

  it "calculates the two-segment qualification footprint without allocating" do
    allocations = ML::MetalBuffer.video_spec_allocations
    geometry = ML::Video::HyNomasContinuationHistory.geometry(
      2, 4320, 354, 384, 54, 16, 128)

    geometry.max_history_rows.should eq(4674)
    geometry.workspace_geometries.size.should eq(2)
    geometry.workspace_geometries[1].history_rows.should eq(4674)
    geometry.workspace_geometries[1].key_rows.should eq(2 * 4674)
    geometry.cache_bytes.should eq(4674_i64 * 16 * 128 * 4 * 2 * 54)
    ML::MetalBuffer.video_spec_allocations.should eq(allocations)
  end

  it "shares the first writer workspace for a two-chunk shared scene" do
    allocations = ML::MetalBuffer.video_spec_allocations
    geometry = ML::Video::HyNomasContinuationHistory.geometry(
      2, 4320, 557, 576, 54, 16, 128,
      text_cache_policy: ML::Video::HyNomasContinuationHistory::TextCachePolicy::OncePerScene)
    geometry.workspace_geometries.map(&.history_rows).should eq([0, 4877])
    geometry.workspace_geometries.map(&.key_rows).should eq([4877, 9197])
    geometry.writer_workspace_geometries.should eq([geometry.workspace_geometries.first])
    geometry.workspace_bytes.should eq(geometry.workspace_geometries.sum(&.workspace_bytes))
    geometry.max_history_rows.should eq(4877)
    ML::MetalBuffer.video_spec_allocations.should eq(allocations)
  end

  it "rejects invalid and overflowing geometry before any allocation" do
    allocations = ML::MetalBuffer.video_spec_allocations
    [
      {1, 4320, 354, 384, 54, 16, 128},
      {7, 4320, 354, 384, 54, 16, 128},
      {2, 4320, 0, 384, 54, 16, 128},
      {2, 4320, 385, 384, 54, 16, 128},
      {2, 4320, 354, 384, 55, 16, 128},
      {6, Int32::MAX, 1, 1, 54, 16, 128},
      {6, 1, 1, Int32::MAX, 54, 16, 128},
      {6, 2, 2, 2, 54, Int32::MAX, Int32::MAX},
    ].each do |invalid|
      ML::Video::HyNomasContinuationHistory::TextCachePolicy.each do |policy|
        expect_raises(ArgumentError) do
          ML::Video::HyNomasContinuationHistory.geometry(*invalid, text_cache_policy: policy)
        end
      end
    end
    ML::MetalBuffer.video_spec_allocations.should eq(allocations)
  end

  it "requires four read-only denoise forwards and one separate writer forward" do
    schedule = ML::Video::HyNomasContinuationHistorySchedule.new(2, 2)
    schedule.denoise_steps.should eq(4)
    expect_raises(ArgumentError) { schedule.select_segment!(0, true) }

    4.times do |pass|
      schedule.select_segment!(0, false)
      expect_raises(ArgumentError) { schedule.finish_layer!(1) }
      schedule.finish_layer!(0)
      expect_raises(ArgumentError) { schedule.select_segment!(0, false) }
      schedule.finish_layer!(1)
      schedule.denoise_passes.should eq(pass + 1)
      schedule.next_segment.should eq(0)
    end
    schedule.awaiting_writer?.should be_true
    expect_raises(ArgumentError) { schedule.select_segment!(0, false) }

    schedule.select_segment!(0, true)
    schedule.finish_layer!(0)
    schedule.next_segment.should eq(0)
    schedule.finish_layer!(1)
    schedule.next_segment.should eq(1)
    schedule.denoise_passes.should eq(0)

    expect_raises(ArgumentError) { schedule.select_segment!(1, true) }
    4.times do
      schedule.select_segment!(1, false)
      schedule.finish_layer!(0)
      schedule.finish_layer!(1)
    end
    schedule.complete?.should be_true
    schedule.next_segment.should eq(2)
    expect_raises(ArgumentError) { schedule.select_segment!(2, false) }
  end

  it "admits one opt-in fixed-reference T0 writer without denoising" do
    schedule = ML::Video::HyNomasContinuationHistorySchedule.new(3, 2,
      reference_initialization: true)

    expect_raises(ArgumentError) { schedule.select_segment!(0, false) }
    expect_raises(ArgumentError) { schedule.select_reference_init!(1) }
    schedule.select_reference_init!(0)
    expect_raises(ArgumentError) { schedule.select_reference_init!(0) }
    expect_raises(ArgumentError) { schedule.select_segment!(1, false) }
    expect_raises(ArgumentError) { schedule.finish_layer!(1) }
    schedule.finish_layer!(0)
    expect_raises(ArgumentError) { schedule.select_segment!(1, false) }
    schedule.finish_layer!(1)

    schedule.next_segment.should eq(1)
    schedule.denoise_passes.should eq(0)
    schedule.awaiting_writer?.should be_false
    schedule.reference_init_required?.should be_false
    expect_raises(ArgumentError) { schedule.select_reference_init!(0) }

    4.times do
      schedule.select_segment!(1, false)
      schedule.finish_layer!(0)
      schedule.finish_layer!(1)
    end
    schedule.awaiting_writer?.should be_true
    schedule.select_segment!(1, writer: true)
    schedule.finish_layer!(0)
    schedule.finish_layer!(1)
    schedule.next_segment.should eq(2)
    schedule.denoise_passes.should eq(0)
  end

  it "keeps a reference writer incomplete until every layer finishes" do
    schedule = ML::Video::HyNomasContinuationHistorySchedule.new(2, 2,
      reference_initialization: true)
    schedule.select_reference_init!(0)
    schedule.finish_layer!(0)

    expect_raises(ArgumentError) { schedule.select_segment!(1, writer: false) }
    expect_raises(ArgumentError) { schedule.select_reference_init!(0) }

    schedule.finish_layer!(1)
    schedule.next_segment.should eq(1)
    schedule.select_segment!(1, writer: false)
  end

  it "rejects reference restore and keeps ordinary generation separate" do
    ordinary = ML::Video::HyNomasContinuationHistorySchedule.new(2, 1)
    expect_raises(ArgumentError) { ordinary.select_reference_init!(0) }

    reference = ML::Video::HyNomasContinuationHistorySchedule.new(2, 1,
      reference_initialization: true)
    expect_raises(ArgumentError) { reference.restore_boundary!(1) }
    reference.reference_init_required?.should be_true
    expect_raises(ArgumentError) { reference.select_segment!(1, writer: false) }

    ordinary.restore_boundary!(1)
    ordinary.select_segment!(1, writer: false)
  end

  it "holds a 54-layer segment open for every denoise pass before its writer" do
    schedule = ML::Video::HyNomasContinuationHistorySchedule.new(2, 54)
    4.times do
      schedule.select_segment!(0, false)
      54.times do |layer|
        schedule.validate_layer!(layer)
        schedule.finish_layer!(layer)
      end
      schedule.idle?.should be_true
      schedule.next_segment.should eq(0)
    end
    schedule.awaiting_writer?.should be_true
    schedule.select_segment!(0, true)
    53.times do |layer|
      schedule.finish_layer!(layer)
      schedule.next_segment.should eq(0)
    end
    schedule.finish_layer!(53)
    schedule.next_segment.should eq(1)
    schedule.denoise_passes.should eq(0)
  end
end
