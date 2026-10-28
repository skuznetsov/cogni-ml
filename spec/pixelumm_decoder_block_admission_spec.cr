require "spec"
require "../src/ml/video/pixelumm_decoder_block_mps"
require "./support/video_allocation_counter"

private def pixelumm_original_index(temporal : Int32, height : Int32, width : Int32,
                                    t : Int32, h : Int32, w : Int32) : Int32
  ((t * height) + h) * width + w
end

private def pixelumm_finish_pass(cursor : ML::Video::PixelUMMDecoderBlockMPS::SerialPassCursor,
                                 chain_id : UInt64) : Nil
  cursor.slot_count.times do |slot|
    cursor.validate_slot!(chain_id, slot.to_i32,
      previous_terminal: false, previous_wait_count: 0)
    cursor.finish_slot!(chain_id, slot.to_i32)
  end
end

private def pixelumm_expected_originals(layout,
                                        temporal : Int32, height : Int32, width : Int32,
                                        tile_tokens : Int32, tile : Int32,
                                        query : Bool) : Array(Int32)
  tiles_w = (width + tile_tokens - 1) // tile_tokens
  tile_h, tile_w = tile.divmod(tiles_w)
  center_h = tile_h.clamp(1, Math.max((height + tile_tokens - 1) // tile_tokens - 2, 1))
  center_w = tile_w.clamp(1, Math.max(tiles_w - 2, 1))
  inverse = layout.inverse_permutation
  originals = [] of Int32

  temporal.times do |t|
    height.times do |h|
      width.times do |w|
        same_query_tile = h // tile_tokens == tile_h && w // tile_tokens == tile_w
        in_key_neighborhood = (h // tile_tokens - center_h).abs <= 1 &&
                              (w // tile_tokens - center_w).abs <= 1
        if query ? same_query_tile : in_key_neighborhood
          originals << pixelumm_original_index(temporal, height, width, t, h, w)
        end
      end
    end
  end

  originals.sort_by! { |original| inverse[original] }
  originals
end

private def pixelumm_assert_local_layout(layout,
                                         temporal : Int32, height : Int32, width : Int32,
                                         tile_tokens : Int32) : Nil
  tokens = temporal * height * width
  tiles_h = (height + tile_tokens - 1) // tile_tokens
  tiles_w = (width + tile_tokens - 1) // tile_tokens
  layout.local_attention?.should be_true
  layout.tile_count.should eq(tiles_h * tiles_w)
  layout.permutation.size.should eq(tokens)
  layout.inverse_permutation.size.should eq(tokens)

  permutation = layout.permutation
  inverse = layout.inverse_permutation
  sorted_permutation = permutation.dup.sort!
  sorted_permutation.should eq(Array(Int32).new(tokens) { |index| index.to_i32 })
  permutation.each_with_index do |original, tile_slot|
    inverse[original].should eq(tile_slot.to_i32)
  end

  observed_max_query = 0
  observed_max_key = 0
  layout.tile_count.times do |tile|
    query_slots = layout.query_indices(tile.to_i32)
    key_slots = layout.key_indices(tile.to_i32)
    query_originals = query_slots.map { |slot| layout.permutation[slot] }
    key_originals = key_slots.map { |slot| layout.permutation[slot] }
    query_originals.should eq(pixelumm_expected_originals(layout, temporal, height, width,
      tile_tokens, tile.to_i32, true))
    key_originals.should eq(pixelumm_expected_originals(layout, temporal, height, width,
      tile_tokens, tile.to_i32, false))
    key_slots.uniq.size.should eq(key_slots.size)
    observed_max_query = Math.max(observed_max_query, query_slots.size)
    observed_max_key = Math.max(observed_max_key, key_slots.size)
  end

  layout.max_query_tokens.should eq(observed_max_query)
  layout.max_key_tokens.should eq(observed_max_key)
end

describe ML::Video::PixelUMMDecoderBlockMPS::Config do
  it "matches the pinned fixed architecture and rejects malformed CPU-only configs" do
    config = ML::Video::PixelUMMDecoderBlockMPS::Config.new(
      temporal: 2, height: 33, width: 25)
    config.layout.local_attention?.should be_true

    expect_raises(ArgumentError) do
      ML::Video::PixelUMMDecoderBlockMPS::Config.new(temporal: 0, height: 33, width: 25)
    end
    expect_raises(ArgumentError) do
      ML::Video::PixelUMMDecoderBlockMPS::Config.new(temporal: 2, height: 0, width: 25)
    end
    expect_raises(ArgumentError) do
      ML::Video::PixelUMMDecoderBlockMPS::Config.new(temporal: 2, height: 33, width: 25,
        query_heads: 30)
    end
    expect_raises(ArgumentError) do
      ML::Video::PixelUMMDecoderBlockMPS::Config.new(temporal: 2, height: 33, width: 25,
        ff_dim: 12_287)
    end
    expect_raises(ArgumentError) do
      ML::Video::PixelUMMDecoderBlockMPS::Config.new(temporal: 2, height: 33, width: 25,
        tile_tokens: 4)
    end
  end

  it "matches upstream tile-major mask semantics for shifted edges and partial tiles" do
    config = ML::Video::PixelUMMDecoderBlockMPS::Config.new(
      temporal: 2, height: 33, width: 25, tile_tokens: 8)
    layout = config.layout
    pixelumm_assert_local_layout(layout, 2, 33, 25, 8)

    # A corner tile is shifted inward by the upstream clamp, and remains
    # connected across every temporal slice.
    corner_keys = layout.key_indices(0_i32).map { |slot| layout.permutation[slot] }
    corner_times = corner_keys.map { |index| index // (33 * 25) }.uniq.sort
    corner_times.should eq([0, 1])

    # Tile (4, 3) is the one-token-high, one-token-wide partial corner.
    layout.query_indices(19_i32).size.should eq(2)
    layout.max_query_tokens.should eq(128)
    layout.max_key_tokens.should eq(1_152)
    layout.max_key_tokens.should be < (2 * 33 * 25)
  end

  it "falls back to full attention when both spatial tile counts are at most three" do
    config = ML::Video::PixelUMMDecoderBlockMPS::Config.new(
      temporal: 2, height: 16, width: 24, tile_tokens: 8)
    layout = config.layout
    tokens = 2 * 16 * 24
    full_slots = Array(Int32).new(tokens) { |index| index.to_i32 }

    layout.local_attention?.should be_false
    layout.tile_count.should eq(1)
    layout.max_query_tokens.should eq(tokens)
    layout.max_key_tokens.should eq(tokens)
    layout.query_indices(0_i32).should eq(full_slots)
    layout.key_indices(0_i32).should eq(full_slots)
  end

  it "keeps the pinned T6/H44/W80 decoder schedule physically neighborhood-bounded" do
    config = ML::Video::PixelUMMDecoderBlockMPS::Config.new(
      temporal: 6, height: 44, width: 80)
    layout = config.layout

    layout.local_attention?.should be_true
    layout.tile_count.should eq(60)
    layout.max_query_tokens.should eq(384)
    layout.max_key_tokens.should eq(3_456)
    layout.max_key_tokens.should be < config.tokens
    layout.query_index_table.size.should eq(60 * (384 + 1))
    layout.key_index_table.size.should eq(60 * (3_456 + 1))
    ML::Video::PixelUMMDecoderBlockMPS::Plan.owned_bytes(config, layout)
      .should eq(2_469_764_068_i64)
  end

  it "preserves the two distinct key neighborhoods in the preview window" do
    config = ML::Video::PixelUMMDecoderBlockMPS::Config.new(
      temporal: 10, height: 18, width: 32)
    layout = config.layout
    pixelumm_assert_local_layout(layout, 10, 18, 32, 8)
    layout.tile_count.should eq(12)
    layout.max_query_tokens.should eq(640)
    layout.max_key_tokens.should eq(4_320)
    keys = Array.new(layout.tile_count) { |tile| layout.key_indices(tile.to_i32) }
    keys.uniq.size.should eq(2)
    keys.each_slice(4) do |row|
      row[0].should eq(keys[0])
      row[1].should eq(keys[0])
      row[2].should eq(keys[2])
      row[3].should eq(keys[2])
    end
  end
end

private def pixelumm_layer0_weight_bytes : Hash(String, Int64)
  {
    "input_norm.weight"     => 4_096_i64 * 2,
    "attn.q_proj.weight"    => 4_096_i64 * 4_096 * 2,
    "attn.k_proj.weight"    => 1_024_i64 * 4_096 * 2,
    "attn.v_proj.weight"    => 1_024_i64 * 4_096 * 2,
    "attn.o_proj.weight"    => 4_096_i64 * 4_096 * 2,
    "attn.q_norm_t.weight"  => 64_i64 * 2,
    "attn.q_norm_hw.weight" => 64_i64 * 2,
    "attn.k_norm_t.weight"  => 64_i64 * 2,
    "attn.k_norm_hw.weight" => 64_i64 * 2,
    "post_attn_norm.weight" => 4_096_i64 * 2,
    "mlp.gate_proj.weight"  => 12_288_i64 * 4_096 * 2,
    "mlp.up_proj.weight"    => 12_288_i64 * 4_096 * 2,
    "mlp.down_proj.weight"  => 4_096_i64 * 12_288 * 2,
  }
end

describe ML::Video::PixelUMMDecoderBlockMPS::Plan do
  it "rejects bad weight/input/table extents before requiring Metal buffers" do
    allocations = ML::MetalBuffer.video_spec_allocations
    config = ML::Video::PixelUMMDecoderBlockMPS::Config.new(
      temporal: 2, height: 33, width: 25)
    weights = pixelumm_layer0_weight_bytes
    input_bytes = config.tokens.to_i64 * 4_096 * 2
    cosine_bytes = config.tokens.to_i64 * 128 * 4
    sine_bytes = cosine_bytes
    required = ML::Video::PixelUMMDecoderBlockMPS::Plan.owned_bytes(config, config.layout)

    ML::Video::PixelUMMDecoderBlockMPS::Plan.validate_admission!(config,
      input_bytes: input_bytes, cosine_bytes: cosine_bytes, sine_bytes: sine_bytes,
      weights_bytes: weights, max_named_bytes: required)

    short_input = input_bytes - 2
    expect_raises(ArgumentError, /extent/) do
      ML::Video::PixelUMMDecoderBlockMPS::Plan.validate_admission!(config,
        input_bytes: short_input, cosine_bytes: cosine_bytes, sine_bytes: sine_bytes,
        weights_bytes: weights, max_named_bytes: required)
    end

    short_cosine = cosine_bytes - 4
    expect_raises(ArgumentError, /extent/) do
      ML::Video::PixelUMMDecoderBlockMPS::Plan.validate_admission!(config,
        input_bytes: input_bytes, cosine_bytes: short_cosine, sine_bytes: sine_bytes,
        weights_bytes: weights, max_named_bytes: required)
    end

    expect_raises(ArgumentError, /extent/) do
      ML::Video::PixelUMMDecoderBlockMPS::Plan.validate_admission!(config,
        input_bytes: input_bytes, cosine_bytes: cosine_bytes, sine_bytes: sine_bytes - 4,
        weights_bytes: weights, max_named_bytes: required)
    end

    malformed_weights = weights.dup
    malformed_weights["attn.k_proj.weight"] -= 2
    expect_raises(ArgumentError, /attn.k_proj.weight/) do
      ML::Video::PixelUMMDecoderBlockMPS::Plan.validate_admission!(config,
        input_bytes: input_bytes, cosine_bytes: cosine_bytes, sine_bytes: sine_bytes,
        weights_bytes: malformed_weights, max_named_bytes: required)
    end

    missing_weight = weights.dup
    missing_weight.delete("attn.v_proj.weight")
    expect_raises(ArgumentError, /exact 13/) do
      ML::Video::PixelUMMDecoderBlockMPS::Plan.validate_admission!(config,
        input_bytes: input_bytes, cosine_bytes: cosine_bytes, sine_bytes: sine_bytes,
        weights_bytes: missing_weight, max_named_bytes: required)
    end

    extra_weight = weights.dup
    extra_weight["decoder.head.weight"] = 1_i64
    expect_raises(ArgumentError, /exact 13/) do
      ML::Video::PixelUMMDecoderBlockMPS::Plan.validate_admission!(config,
        input_bytes: input_bytes, cosine_bytes: cosine_bytes, sine_bytes: sine_bytes,
        weights_bytes: extra_weight, max_named_bytes: required)
    end

    expect_raises(ArgumentError, /unique/) do
      ML::Video::PixelUMMDecoderBlockMPS::Plan.validate_admission!(config,
        input_bytes: input_bytes, cosine_bytes: cosine_bytes, sine_bytes: sine_bytes,
        weights_bytes: weights, max_named_bytes: required,
        captures: ["q_rope", "q_rope"])
    end
    expect_raises(ArgumentError, /unknown/) do
      ML::Video::PixelUMMDecoderBlockMPS::Plan.validate_admission!(config,
        input_bytes: input_bytes, cosine_bytes: cosine_bytes, sine_bytes: sine_bytes,
        weights_bytes: weights, max_named_bytes: required,
        captures: ["unbounded_scores"])
    end
    ML::MetalBuffer.video_spec_allocations.should eq(allocations)
  end

  it "adopts exact plan-owned budget and rejects a one-byte underestimate" do
    allocations = ML::MetalBuffer.video_spec_allocations
    config = ML::Video::PixelUMMDecoderBlockMPS::Config.new(
      temporal: 2, height: 33, width: 25)
    weights = pixelumm_layer0_weight_bytes
    required = ML::Video::PixelUMMDecoderBlockMPS::Plan.owned_bytes(config, config.layout)
    # This shape has a 128-query by 1,152-key maximum local attention geometry.
    required.should eq(236_300_140_i64)
    ML::Video::PixelUMMDecoderBlockMPS::Plan::CAPTURE_NAMES.size.should eq(26)
    statistics_capture_bytes = ML::Video::PixelUMMDecoderBlockMPS::Plan.owned_bytes(
      config, config.layout, ["input_norm_statistics"])
    statistics_capture_bytes.should eq(required + config.tokens.to_i64 * 4)
    q_statistics_bytes = ML::Video::PixelUMMDecoderBlockMPS::Plan.owned_bytes(
      config, config.layout, ["q_norm_statistics"])
    q_statistics_bytes.should eq(required + config.tokens.to_i64 * 32 * 2 * 4)
    k_statistics_bytes = ML::Video::PixelUMMDecoderBlockMPS::Plan.owned_bytes(
      config, config.layout, ["k_norm_statistics"])
    k_statistics_bytes.should eq(required + config.tokens.to_i64 * 8 * 2 * 4)
    q_inverse_bytes = ML::Video::PixelUMMDecoderBlockMPS::Plan.owned_bytes(
      config, config.layout, ["q_norm_inverse"])
    q_inverse_bytes.should eq(required + config.tokens.to_i64 * 32 * 2 * 4)
    k_inverse_bytes = ML::Video::PixelUMMDecoderBlockMPS::Plan.owned_bytes(
      config, config.layout, ["k_norm_inverse"])
    k_inverse_bytes.should eq(required + config.tokens.to_i64 * 8 * 2 * 4)
    qk_diagnostics = ["q_norm_statistics", "k_norm_statistics",
                      "q_norm_inverse", "k_norm_inverse"]
    qk_diagnostic_bytes = ML::Video::PixelUMMDecoderBlockMPS::Plan.owned_bytes(
      config, config.layout, qk_diagnostics)
    qk_diagnostic_bytes.should eq(required + config.tokens.to_i64 * 640)
    ML::Video::PixelUMMDecoderBlockMPS::Plan.validate_admission!(config,
      input_bytes: config.tokens.to_i64 * 4_096 * 2,
      cosine_bytes: config.tokens.to_i64 * 128 * 4,
      sine_bytes: config.tokens.to_i64 * 128 * 4,
      weights_bytes: weights, max_named_bytes: qk_diagnostic_bytes,
      captures: qk_diagnostics)
    ML::Video::PixelUMMDecoderBlockMPS::Plan.validate_admission!(config,
      input_bytes: config.tokens.to_i64 * 4_096 * 2,
      cosine_bytes: config.tokens.to_i64 * 128 * 4,
      sine_bytes: config.tokens.to_i64 * 128 * 4,
      weights_bytes: weights, max_named_bytes: required)
    expect_raises(ArgumentError, /budget/) do
      ML::Video::PixelUMMDecoderBlockMPS::Plan.validate_admission!(config,
        input_bytes: config.tokens.to_i64 * 4_096 * 2,
        cosine_bytes: config.tokens.to_i64 * 128 * 4,
        sine_bytes: config.tokens.to_i64 * 128 * 4,
        weights_bytes: weights, max_named_bytes: required - 1)
    end
    ML::MetalBuffer.video_spec_allocations.should eq(allocations)
  end

  it "admits the exact global fallback token bound and rejects larger grids before Metal allocation" do
    allocations = ML::MetalBuffer.video_spec_allocations
    weights = pixelumm_layer0_weight_bytes

    # Keep Config.layout's geometric fallback formula independent from Plan's
    # admission cap: exactly 2048 tokens still uses full attention and admits.
    boundary = ML::Video::PixelUMMDecoderBlockMPS::Config.new(
      temporal: 8, height: 16, width: 16)
    boundary.tokens.should eq(2_048)
    boundary.layout.local_attention?.should be_false
    boundary.layout.max_query_tokens.should eq(2_048)
    ML::Video::PixelUMMDecoderBlockMPS::Plan.validate_admission!(boundary,
      input_bytes: boundary.tokens.to_i64 * 4_096 * 2,
      cosine_bytes: boundary.tokens.to_i64 * 128 * 4,
      sine_bytes: boundary.tokens.to_i64 * 128 * 4,
      weights_bytes: weights, max_named_bytes: Int64::MAX)

    # The geometry formula remains full attention above the operational cap;
    # the CPU-only admission gate rejects it before constructing GPU buffers.
    oversized = ML::Video::PixelUMMDecoderBlockMPS::Config.new(
      temporal: 4, height: 24, width: 22)
    oversized.tokens.should eq(2_112)
    oversized.layout.local_attention?.should be_false
    oversized.layout.max_query_tokens.should eq(2_112)
    oversized.layout.max_key_tokens.should eq(2_112)
    expect_raises(ArgumentError, /full-attention.*2048/) do
      ML::Video::PixelUMMDecoderBlockMPS::Plan.validate_admission!(oversized,
        input_bytes: oversized.tokens.to_i64 * 4_096 * 2,
        cosine_bytes: oversized.tokens.to_i64 * 128 * 4,
        sine_bytes: oversized.tokens.to_i64 * 128 * 4,
        weights_bytes: weights, max_named_bytes: Int64::MAX)
    end

    ML::MetalBuffer.video_spec_allocations.should eq(allocations)
  end
end

describe ML::Video::PixelUMMDecoderBlockMPS::Workspace do
  it "reuses only two output banks and preserves the retained-output default" do
    allocations = ML::MetalBuffer.video_spec_allocations
    config = ML::Video::PixelUMMDecoderBlockMPS::Config.new(
      temporal: 6, height: 44, width: 80)
    output_bytes = 173_015_040_i64
    {1, 2, 3, 4, 16, 36}.each do |count|
      captures = Array.new(count) { [] of String }
      retained = ML::Video::PixelUMMDecoderBlockMPS::Workspace.owned_bytes(config, config.layout, captures)
      ring = ML::Video::PixelUMMDecoderBlockMPS::Workspace.owned_bytes(config, config.layout, captures,
        reuse_outputs: true)
      (retained - ring).should eq(Math.max(count - 2, 0).to_i64 * output_bytes)
      ML::Video::PixelUMMDecoderBlockMPS::Workspace.output_bank_count(count,
        reuse_outputs: true).should eq(Math.min(count, 2))
      count.times do |slot|
        ML::Video::PixelUMMDecoderBlockMPS::Workspace.output_bank_index(slot, count,
          reuse_outputs: true).should eq(slot % 2)
        ML::Video::PixelUMMDecoderBlockMPS::Workspace.output_bank_index(slot, count).should eq(slot)
      end
    end
    expect_raises(ArgumentError, /slot/) do
      ML::Video::PixelUMMDecoderBlockMPS::Workspace.output_bank_index(3, 3, reuse_outputs: true)
    end
    expect_raises(ArgumentError, /slot/) do
      ML::Video::PixelUMMDecoderBlockMPS::Workspace.output_bank_index(-1, 4, reuse_outputs: true)
    end
    expect_raises(ArgumentError, /slot/) do
      ML::Video::PixelUMMDecoderBlockMPS::Workspace.output_bank_count(0, reuse_outputs: true)
    end
    ML::MetalBuffer.video_spec_allocations.should eq(allocations)
  end

  it "rejects captures overwritten by later ring writers, including input aliases" do
    config = ML::Video::PixelUMMDecoderBlockMPS::Config.new(temporal: 6, height: 44, width: 80)
    {3, 4, 16}.each do |count|
      captures = Array.new(count) { ["attention"] }
      captures[count - 2] << "output"
      captures[count - 1].concat(["x", "output"])
      captures[0] << "x"
      ring = ML::Video::PixelUMMDecoderBlockMPS::Workspace.owned_bytes(config, config.layout, captures,
        reuse_outputs: true)
      ring.should be > 0
      stale_output = captures.map(&.dup)
      stale_output[0] << "output"
      expect_raises(ArgumentError, /overwritten/) do
        ML::Video::PixelUMMDecoderBlockMPS::Workspace.owned_bytes(config, config.layout, stale_output,
          reuse_outputs: true)
      end
      (1...count - 1).each do |slot|
        stale_input = captures.map(&.dup)
        stale_input[slot] << "x"
        expect_raises(ArgumentError, /overwritten/) do
          ML::Video::PixelUMMDecoderBlockMPS::Workspace.owned_bytes(config, config.layout, stale_input,
            reuse_outputs: true)
        end
      end
      # The same diagnostic aliases remain valid in the retained-output mode.
      ML::Video::PixelUMMDecoderBlockMPS::Workspace.owned_bytes(config, config.layout, stale_output).should be > ring
    end
  end

  it "admits the exact ring budget and rejects one byte less before Metal allocation" do
    allocations = ML::MetalBuffer.video_spec_allocations
    config = ML::Video::PixelUMMDecoderBlockMPS::Config.new(temporal: 6, height: 44, width: 80)
    weights = Array.new(16) { pixelumm_layer0_weight_bytes }
    captures = Array.new(16) { [] of String }
    owned = ML::Video::PixelUMMDecoderBlockMPS::Workspace.owned_bytes(config, config.layout, captures,
      reuse_outputs: true)
    unique = owned + config.tokens.to_i64 * (4096 * 2 + 128 * 8) + weights.sum(&.values.sum)
    ML::Video::PixelUMMDecoderBlockMPS::Workspace.validate_admission!(config,
      input_bytes: config.tokens.to_i64 * 4096 * 2,
      cosine_bytes: config.tokens.to_i64 * 128 * 4, sine_bytes: config.tokens.to_i64 * 128 * 4,
      weights_bytes_by_slot: weights, captures_by_slot: captures,
      max_unique_bytes: unique, reuse_outputs: true).should eq(unique)
    expect_raises(ArgumentError, /budget/) do
      ML::Video::PixelUMMDecoderBlockMPS::Workspace.validate_admission!(config,
        input_bytes: config.tokens.to_i64 * 4096 * 2,
        cosine_bytes: config.tokens.to_i64 * 128 * 4, sine_bytes: config.tokens.to_i64 * 128 * 4,
        weights_bytes_by_slot: weights, captures_by_slot: captures,
        max_unique_bytes: unique - 1, reuse_outputs: true)
    end
    ML::MetalBuffer.video_spec_allocations.should eq(allocations)
  end

  {% if flag?(:darwin) && !flag?(:cpu_only) %}
    it "rejects future-bank external aliases and wrong predecessor inputs without encoding" do
      config = ML::Video::PixelUMMDecoderBlockMPS::Config.new(temporal: 1, height: 1, width: 1)
      captures = Array.new(3) { [] of String }
      workspace = ML::Video::PixelUMMDecoderBlockMPS::Workspace.new(config,
        captures_by_slot: captures, reuse_outputs: true, max_owned_bytes: 1_i64 << 20)
      external = Array.new(4) { ML::MetalBuffer.new(8192_i64) }
      x, cosine, sine, weight = external
      begin
        a, b = workspace.output_buffer(0), workspace.output_buffer(1)
        workspace.output_buffer(2).same?(a).should be_true
        a.same?(b).should be_false
        workspace.output_bank_count.should eq(2)
        expect_raises(ArgumentError, /external bindings/) do
          workspace.validate_plan_binding!(config, 0, b, cosine, sine, [weight], [] of String)
        end
        expect_raises(ArgumentError, /external bindings/) do
          workspace.validate_plan_binding!(config, 0, x, b, sine, [weight], [] of String)
        end
        expect_raises(ArgumentError, /external bindings/) do
          workspace.validate_plan_binding!(config, 0, x, cosine, sine, [b], [] of String)
        end
        workspace.attach_plan!(0, x, cosine, sine, [weight], [] of String)
        expect_raises(ArgumentError, /prior slot output/) do
          workspace.validate_plan_binding!(config, 1, b, cosine, sine, [] of ML::MetalBuffer, [] of String)
        end
        workspace.attach_plan!(1, a, cosine, sine, [] of ML::MetalBuffer, [] of String)
        expect_raises(ArgumentError, /prior slot output/) do
          workspace.validate_plan_binding!(config, 2, a, cosine, sine, [] of ML::MetalBuffer, [] of String)
        end
        workspace.validate_plan_binding!(config, 2, b, cosine, sine, [] of ML::MetalBuffer, [] of String)
      ensure
        workspace.detach_failed_plan!(1)
        workspace.detach_failed_plan!(0)
        workspace.release
        external.each(&.release)
      end
    end
  {% end %}

  it "counts two full-grid slots with one transient set and disjoint outputs" do
    config = ML::Video::PixelUMMDecoderBlockMPS::Config.new(
      temporal: 6, height: 44, width: 80)
    layout = config.layout
    captures = [["x", "output"], ["x", "output"]]
    workspace_bytes = ML::Video::PixelUMMDecoderBlockMPS::Workspace.owned_bytes(
      config, layout, captures)
    workspace_bytes.should eq(2_642_779_108_i64)
    workspace_bytes.should eq(
      ML::Video::PixelUMMDecoderBlockMPS::Plan.owned_bytes(config, layout) +
      config.tokens.to_i64 * config.model_dim * 2)

    no_capture_bytes = ML::Video::PixelUMMDecoderBlockMPS::Workspace.owned_bytes(
      config, layout, [[] of String, [] of String])
    no_capture_bytes.should eq(workspace_bytes)
  end

  it "preflights all borrowed bindings and unique outputs before Metal allocation" do
    allocations = ML::MetalBuffer.video_spec_allocations
    config = ML::Video::PixelUMMDecoderBlockMPS::Config.new(
      temporal: 2, height: 33, width: 25)
    weights = pixelumm_layer0_weight_bytes
    captures = [["x", "output"], ["x", "output"]]
    owned = ML::Video::PixelUMMDecoderBlockMPS::Workspace.owned_bytes(
      config, config.layout, captures)
    owned.should eq(249_816_940_i64)
    unique = owned + config.tokens.to_i64 * 4_096 * 2 +
             config.tokens.to_i64 * 128 * 8 + weights.values.sum * 2
    unique.should eq(1_036_809_068_i64)

    accepted = ML::Video::PixelUMMDecoderBlockMPS::Workspace.validate_admission!(config,
      input_bytes: config.tokens.to_i64 * 4_096 * 2,
      cosine_bytes: config.tokens.to_i64 * 128 * 4,
      sine_bytes: config.tokens.to_i64 * 128 * 4,
      weights_bytes_by_slot: [weights, weights.dup],
      captures_by_slot: captures,
      max_unique_bytes: unique)
    accepted.should eq(unique)

    expect_raises(ArgumentError, /budget/) do
      ML::Video::PixelUMMDecoderBlockMPS::Workspace.validate_admission!(config,
        input_bytes: config.tokens.to_i64 * 4_096 * 2,
        cosine_bytes: config.tokens.to_i64 * 128 * 4,
        sine_bytes: config.tokens.to_i64 * 128 * 4,
        weights_bytes_by_slot: [weights, weights.dup],
        captures_by_slot: captures,
        max_unique_bytes: unique - 1)
    end

    expect_raises(ArgumentError, /slot counts/) do
      ML::Video::PixelUMMDecoderBlockMPS::Workspace.validate_admission!(config,
        input_bytes: config.tokens.to_i64 * 4_096 * 2,
        cosine_bytes: config.tokens.to_i64 * 128 * 4,
        sine_bytes: config.tokens.to_i64 * 128 * 4,
        weights_bytes_by_slot: [weights],
        captures_by_slot: captures,
        max_unique_bytes: Int64::MAX)
    end

    malformed = weights.dup
    malformed["attn.q_proj.weight"] -= 2
    expect_raises(ArgumentError, /q_proj.weight/) do
      ML::Video::PixelUMMDecoderBlockMPS::Workspace.validate_admission!(config,
        input_bytes: config.tokens.to_i64 * 4_096 * 2,
        cosine_bytes: config.tokens.to_i64 * 128 * 4,
        sine_bytes: config.tokens.to_i64 * 128 * 4,
        weights_bytes_by_slot: [weights, malformed],
        captures_by_slot: captures,
        max_unique_bytes: Int64::MAX)
    end
    ML::MetalBuffer.video_spec_allocations.should eq(allocations)
  end

  it "counts captures once per slot and treats x/output as aliases" do
    config = ML::Video::PixelUMMDecoderBlockMPS::Config.new(
      temporal: 2, height: 33, width: 25)
    no_captures = ML::Video::PixelUMMDecoderBlockMPS::Workspace.owned_bytes(
      config, config.layout, [[] of String, [] of String])
    alias_captures = ML::Video::PixelUMMDecoderBlockMPS::Workspace.owned_bytes(
      config, config.layout, [["x", "output"], ["x", "output"]])
    alias_captures.should eq(no_captures)

    one_capture = ML::Video::PixelUMMDecoderBlockMPS::Workspace.owned_bytes(
      config, config.layout, [["input_norm"], [] of String])
    one_capture.should eq(no_captures + config.tokens.to_i64 * 4_096 * 2)
    two_captures = ML::Video::PixelUMMDecoderBlockMPS::Workspace.owned_bytes(
      config, config.layout, [["input_norm"], ["input_norm"]])
    two_captures.should eq(no_captures + config.tokens.to_i64 * 4_096 * 4)

    expect_raises(ArgumentError, /unique/) do
      ML::Video::PixelUMMDecoderBlockMPS::Workspace.owned_bytes(
        config, config.layout, [["x", "x"], [] of String])
    end
    expect_raises(ArgumentError, /unknown/) do
      ML::Video::PixelUMMDecoderBlockMPS::Workspace.owned_bytes(
        config, config.layout, [["global_scores"], [] of String])
    end
  end

  it "admits four full-grid slots with shared transients counted once" do
    config = ML::Video::PixelUMMDecoderBlockMPS::Config.new(
      temporal: 6, height: 44, width: 80)
    weights = pixelumm_layer0_weight_bytes
    weights_by_slot = Array.new(4) { weights.dup }
    captures = Array.new(4) { ["attention", "output"] }

    output_bytes = config.tokens.to_i64 * config.model_dim * 2
    single_plan_bytes = ML::Video::PixelUMMDecoderBlockMPS::Plan.owned_bytes(
      config, config.layout)
    no_capture_bytes = ML::Video::PixelUMMDecoderBlockMPS::Workspace.owned_bytes(
      config, config.layout, Array.new(4) { [] of String })
    owned = ML::Video::PixelUMMDecoderBlockMPS::Workspace.owned_bytes(
      config, config.layout, captures)

    no_capture_bytes.should eq(single_plan_bytes + 3 * output_bytes)
    owned.should eq(no_capture_bytes + 4 * output_bytes)
    owned.should eq(3_680_869_348_i64)

    input_bytes = config.tokens.to_i64 * 4_096 * 2
    cosine_bytes = config.tokens.to_i64 * 128 * 4
    sine_bytes = cosine_bytes
    borrowed = input_bytes + cosine_bytes + sine_bytes + weights.values.sum * 4
    borrowed.should eq(1_738_213_376_i64)
    unique = owned + borrowed
    unique.should eq(5_419_082_724_i64)

    accepted = ML::Video::PixelUMMDecoderBlockMPS::Workspace.validate_admission!(config,
      input_bytes: input_bytes, cosine_bytes: cosine_bytes, sine_bytes: sine_bytes,
      weights_bytes_by_slot: weights_by_slot, captures_by_slot: captures,
      max_unique_bytes: unique)
    accepted.should eq(unique)

    expect_raises(ArgumentError, /budget/) do
      ML::Video::PixelUMMDecoderBlockMPS::Workspace.validate_admission!(config,
        input_bytes: input_bytes, cosine_bytes: cosine_bytes, sine_bytes: sine_bytes,
        weights_bytes_by_slot: weights_by_slot, captures_by_slot: captures,
        max_unique_bytes: unique - 1)
    end
    expect_raises(ArgumentError, /slot counts/) do
      ML::Video::PixelUMMDecoderBlockMPS::Workspace.validate_admission!(config,
        input_bytes: input_bytes, cosine_bytes: cosine_bytes, sine_bytes: sine_bytes,
        weights_bytes_by_slot: weights_by_slot[0...3], captures_by_slot: captures,
        max_unique_bytes: Int64::MAX)
    end
    expect_raises(ArgumentError, /budget/) do
      ML::Video::PixelUMMDecoderBlockMPS::Workspace.validate_admission!(config,
        input_bytes: input_bytes, cosine_bytes: cosine_bytes, sine_bytes: sine_bytes,
        weights_bytes_by_slot: weights_by_slot, captures_by_slot: captures,
        max_unique_bytes: 0_i64)
    end

    missing_fourth_weight = weights.dup
    missing_fourth_weight.delete("mlp.down_proj.weight")
    malformed_slots = [weights.dup, weights.dup, weights.dup, missing_fourth_weight]
    expect_raises(ArgumentError, /exact 13 PixelUMM layer0 weight metadata entries/) do
      ML::Video::PixelUMMDecoderBlockMPS::Workspace.validate_admission!(config,
        input_bytes: input_bytes, cosine_bytes: cosine_bytes, sine_bytes: sine_bytes,
        weights_bytes_by_slot: malformed_slots, captures_by_slot: captures,
        max_unique_bytes: Int64::MAX)
    end

    fallback = ML::Video::PixelUMMDecoderBlockMPS::Config.new(
      temporal: 4, height: 24, width: 24)
    expect_raises(ArgumentError, /fallback exceeds 2048 tokens/) do
      ML::Video::PixelUMMDecoderBlockMPS::Workspace.validate_admission!(fallback,
        input_bytes: fallback.tokens.to_i64 * 4_096 * 2,
        cosine_bytes: fallback.tokens.to_i64 * 128 * 4,
        sine_bytes: fallback.tokens.to_i64 * 128 * 4,
        weights_bytes_by_slot: weights_by_slot, captures_by_slot: captures,
        max_unique_bytes: Int64::MAX)
    end
  end

  it "admits eight full-grid slots with exact unique-byte accounting" do
    config = ML::Video::PixelUMMDecoderBlockMPS::Config.new(
      temporal: 6, height: 44, width: 80)
    weights = pixelumm_layer0_weight_bytes
    weights_by_slot = Array.new(8) { weights.dup }
    captures = Array.new(8) { ["attention", "output"] }

    output_bytes = config.tokens.to_i64 * config.model_dim * 2
    attention_capture_bytes = output_bytes
    single_plan_bytes = ML::Video::PixelUMMDecoderBlockMPS::Plan.owned_bytes(
      config, config.layout)
    no_capture_bytes = ML::Video::PixelUMMDecoderBlockMPS::Workspace.owned_bytes(
      config, config.layout, Array.new(8) { [] of String })
    owned = ML::Video::PixelUMMDecoderBlockMPS::Workspace.owned_bytes(
      config, config.layout, captures)

    # Shared transients/scratch are allocated once; each slot owns one output
    # bank, while output capture aliases that bank and attention is copied to a
    # separate per-slot capture buffer.
    no_capture_bytes.should eq(single_plan_bytes + 7 * output_bytes)
    no_capture_bytes.should eq(3_680_869_348_i64)
    owned.should eq(no_capture_bytes + 8 * attention_capture_bytes)
    owned.should eq(5_064_989_668_i64)

    input_bytes = config.tokens.to_i64 * 4_096 * 2
    cosine_bytes = config.tokens.to_i64 * 128 * 4
    sine_bytes = cosine_bytes
    borrowed = input_bytes + cosine_bytes + sine_bytes + weights.values.sum * 8
    borrowed.should eq(3_281_784_832_i64)
    unique = owned + borrowed
    unique.should eq(8_346_774_500_i64)
    unique.should be > UInt32::MAX

    accepted = ML::Video::PixelUMMDecoderBlockMPS::Workspace.validate_admission!(config,
      input_bytes: input_bytes, cosine_bytes: cosine_bytes, sine_bytes: sine_bytes,
      weights_bytes_by_slot: weights_by_slot, captures_by_slot: captures,
      max_unique_bytes: unique)
    accepted.should eq(unique)

    expect_raises(ArgumentError, /budget/) do
      ML::Video::PixelUMMDecoderBlockMPS::Workspace.validate_admission!(config,
        input_bytes: input_bytes, cosine_bytes: cosine_bytes, sine_bytes: sine_bytes,
        weights_bytes_by_slot: weights_by_slot, captures_by_slot: captures,
        max_unique_bytes: unique - 1)
    end

    expect_raises(ArgumentError, /slot counts/) do
      ML::Video::PixelUMMDecoderBlockMPS::Workspace.validate_admission!(config,
        input_bytes: input_bytes, cosine_bytes: cosine_bytes, sine_bytes: sine_bytes,
        weights_bytes_by_slot: weights_by_slot[0...7], captures_by_slot: captures,
        max_unique_bytes: Int64::MAX)
    end

    missing_slot7_weight = weights.dup
    missing_slot7_weight.delete("mlp.down_proj.weight")
    malformed_slots = weights_by_slot[0...7] + [missing_slot7_weight]
    expect_raises(ArgumentError, /exact 13 PixelUMM layer0 weight metadata entries/) do
      ML::Video::PixelUMMDecoderBlockMPS::Workspace.validate_admission!(config,
        input_bytes: input_bytes, cosine_bytes: cosine_bytes, sine_bytes: sine_bytes,
        weights_bytes_by_slot: malformed_slots, captures_by_slot: captures,
        max_unique_bytes: Int64::MAX)
    end

    fallback = ML::Video::PixelUMMDecoderBlockMPS::Config.new(
      temporal: 4, height: 24, width: 24)
    expect_raises(ArgumentError, /fallback exceeds 2048 tokens/) do
      ML::Video::PixelUMMDecoderBlockMPS::Workspace.validate_admission!(fallback,
        input_bytes: fallback.tokens.to_i64 * 4_096 * 2,
        cosine_bytes: fallback.tokens.to_i64 * 128 * 4,
        sine_bytes: fallback.tokens.to_i64 * 128 * 4,
        weights_bytes_by_slot: weights_by_slot, captures_by_slot: captures,
        max_unique_bytes: Int64::MAX)
    end
  end
end

describe ML::Video::PixelUMMDecoderBlockMPS::SerialPassCursor do
  it "supports the five-window 36-plan ceiling and rejects wider caps" do
    expect_raises(ArgumentError, /between 1 and 5/) do
      ML::Video::PixelUMMDecoderBlockMPS::SerialPassCursor.new(36, 0)
    end
    expect_raises(ArgumentError, /between 1 and 5/) do
      ML::Video::PixelUMMDecoderBlockMPS::SerialPassCursor.new(36, 6)
    end

    cursor = ML::Video::PixelUMMDecoderBlockMPS::SerialPassCursor.new(36, 5)
    5.times { pixelumm_finish_pass(cursor, 36_u64) }
    cursor.passes_on_chain.should eq(5)
    expect_raises(ArgumentError, /per-chain pass limit/) do
      cursor.validate_slot!(36_u64, 0,
        previous_terminal: false, previous_wait_count: 0)
    end
  end

  it "preserves one-pass default and allows only bounded ordered same-chain passes" do
    default_cursor = ML::Video::PixelUMMDecoderBlockMPS::SerialPassCursor.new(3)
    pixelumm_finish_pass(default_cursor, 1_u64)
    default_cursor.passes_on_chain.should eq(1)
    expect_raises(ArgumentError, /per-chain pass limit/) do
      default_cursor.validate_slot!(1_u64, 0,
        previous_terminal: false, previous_wait_count: 0)
    end

    bounded_cursor = ML::Video::PixelUMMDecoderBlockMPS::SerialPassCursor.new(3, 2)
    pixelumm_finish_pass(bounded_cursor, 7_u64)
    bounded_cursor.validate_slot!(7_u64, 0,
      previous_terminal: false, previous_wait_count: 0)
    bounded_cursor.finish_slot!(7_u64, 0)
    expect_raises(ArgumentError, /slot order/) do
      bounded_cursor.validate_slot!(7_u64, 2,
        previous_terminal: false, previous_wait_count: 0)
    end
    bounded_cursor.validate_slot!(7_u64, 1,
      previous_terminal: false, previous_wait_count: 0)
    bounded_cursor.finish_slot!(7_u64, 1)
    bounded_cursor.validate_slot!(7_u64, 2,
      previous_terminal: false, previous_wait_count: 0)
    bounded_cursor.finish_slot!(7_u64, 2)
    bounded_cursor.passes_on_chain.should eq(2)
    expect_raises(ArgumentError, /per-chain pass limit/) do
      bounded_cursor.validate_slot!(7_u64, 0,
        previous_terminal: false, previous_wait_count: 0)
    end
  end

  it "requires terminal wait for chain changes and keeps failed submissions poisoned" do
    cursor = ML::Video::PixelUMMDecoderBlockMPS::SerialPassCursor.new(2, 5)
    pixelumm_finish_pass(cursor, 10_u64)
    expect_raises(ArgumentError, /terminal completion/) do
      cursor.validate_slot!(11_u64, 0,
        previous_terminal: false, previous_wait_count: 1)
    end
    expect_raises(ArgumentError, /terminal completion/) do
      cursor.validate_slot!(11_u64, 0,
        previous_terminal: true, previous_wait_count: 0)
    end
    cursor.validate_slot!(11_u64, 0,
      previous_terminal: true, previous_wait_count: 1)
    cursor.passes_on_chain.should eq(0)
    cursor.finish_slot!(11_u64, 0)
    cursor.poison!(11_u64)
    cursor.poisoned?.should be_true
    expect_raises(ArgumentError, /poisoned until prior chain terminal completion/) do
      cursor.validate_slot!(12_u64, 0,
        previous_terminal: false, previous_wait_count: 0)
    end
    expect_raises(ArgumentError, /poisoned until prior chain terminal completion/) do
      cursor.validate_slot!(11_u64, 1,
        previous_terminal: true, previous_wait_count: 1)
    end
    expect_raises(ArgumentError, /slot zero/) do
      cursor.validate_slot!(12_u64, 1,
        previous_terminal: true, previous_wait_count: 1)
    end
    cursor.poisoned?.should be_true
    cursor.validate_slot!(12_u64, 0,
      previous_terminal: true, previous_wait_count: 1)
    cursor.poisoned?.should be_false
    cursor.passes_on_chain.should eq(0)
  end
end
