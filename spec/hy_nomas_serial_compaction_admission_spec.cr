require "spec"
require "../src/ml/video/hy_nomas_serial_mps"

describe ML::Video::HyNomasConfig do
  it "admits only a valid batch-one prefix with broadcast text modulation" do
    config = ML::Video::HyNomasConfig.new(batches: 1, video_tokens: 5280,
      text_tokens: 1280, dim: 2048, heads: 16, head_dim: 128, ffn_dim: 8192)

    compact = config.compact_valid_text_prefix(23)

    config.text_tokens.should eq(1280)
    compact.text_tokens.should eq(23)
    compact.image_rows.should eq(config.image_rows)
    compact.dim.should eq(config.dim)
    compact.joint_tokens.should eq(5303)
    ML::Video::HyNomasMPSSerialWorkspace.compaction_resident_bytes(config, 23).should eq(188_416_i64)
    ML::Video::HyNomasMPSSerialWorkspace.compaction_resident_bytes(config, 1280).should eq(0_i64)

    [-1, 0, 1281].each do |invalid_count|
      expect_raises(ArgumentError, /valid text prefix/) do
        config.compact_valid_text_prefix(invalid_count)
      end
    end
  end

  it "admits aligned storage while keeping the attention prefix exact" do
    config = ML::Video::HyNomasConfig.new(batches: 1, video_tokens: 5280,
      text_tokens: 1280, dim: 2048, heads: 16, head_dim: 128, ffn_dim: 8192)

    {23 => 188_416_i64, 32 => 262_144_i64, 64 => 524_288_i64,
     128 => 1_048_576_i64, 256 => 2_097_152_i64, 512 => 4_194_304_i64,
     1280 => 0_i64}.each do |storage, resident_bytes|
      compact = config.compact_valid_text_prefix(23, storage)
      compact.text_tokens.should eq(storage)
      compact.joint_tokens.should eq(5280 + storage)
      ML::Video::HyNomasMPSSerialWorkspace.compaction_resident_bytes(config, 23, storage).should eq(resident_bytes)
      ML::Video::HyNomasMPSWorkspace.attention_tokens_for(compact, 23).should eq(5303)
    end

    # Omitting storage retains the compatibility behavior: allocate exactly
    # the valid prefix, with no separate padded block geometry.
    config.compact_valid_text_prefix(23).text_tokens.should eq(23)
    ML::Video::HyNomasMPSSerialWorkspace.compaction_resident_bytes(config, 23).should eq(188_416_i64)
  end

  it "rejects storage below the valid prefix or beyond the source text extent" do
    config = ML::Video::HyNomasConfig.new(batches: 1, video_tokens: 6,
      text_tokens: 8, dim: 64, heads: 2, head_dim: 32, ffn_dim: 256)

    [0, 2, 9].each do |invalid_storage|
      expect_raises(ArgumentError, /storage/) do
        config.compact_valid_text_prefix(3, invalid_storage)
      end
      expect_raises(ArgumentError, /storage/) do
        ML::Video::HyNomasMPSSerialWorkspace.compaction_resident_bytes(config, 3, invalid_storage)
      end
    end
  end

  it "rejects compact geometry that is outside the serial-prefix contract" do
    config = ML::Video::HyNomasConfig.new(batches: 1, video_tokens: 6,
      text_tokens: 8, dim: 64, heads: 2, head_dim: 32, ffn_dim: 256)

    [
      config.copy_with(batches: 2),
      config.copy_with(segment_count: 2),
      config.copy_with(chunk_size_tokens: 1),
      config.copy_with(history_tokens: 1),
      config.copy_with(has_mask: true),
      config.copy_with(has_camera: true),
      config.copy_with(action_windows: true),
      config.copy_with(txt_modulation_rows: config.text_rows),
    ].each do |unsupported|
      expect_raises(ArgumentError) do
        unsupported.compact_valid_text_prefix(3)
      end
    end
  end
end
