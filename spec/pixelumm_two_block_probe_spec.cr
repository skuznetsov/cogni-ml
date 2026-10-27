require "spec"
require "../examples/support/pixelumm_two_block_probe"

private def pixelumm_pair_weight_records(block_count : Int32 = 2) : Hash(String, PixelUMMTwoBlockProbe::Record)
  result = {} of String => PixelUMMTwoBlockProbe::Record
  offset = 0_i64
  block_count.times do |slot|
    PixelUMMBlockProbe.weight_shapes.each do |name, shape|
      nbytes = shape.reduce(2_i64) { |n, v| n * v }
      value = JSON.parse({dtype: "BF16", shape: shape, offset: offset,
                          nbytes: nbytes, sha256: "a" * 64}.to_json)
      result["block#{slot}.#{name}"] = PixelUMMTwoBlockProbe::Record.new(value, block_count.to_i64 * 385_892_864)
      offset += nbytes
    end
  end
  result
end

private def with_pixelumm_tensor_file(&)
  file = File.tempfile("pixelumm-pair-", ".json")
  file.close
  path = File.realpath(file.path)
  payload = path + ".bin"
  begin
    File.write(payload, "abcd")
    raw = {payload: File.basename(payload), payload_bytes: 4,
           payload_sha256: Digest::SHA256.hexdigest("abcd")}.to_json
    File.write(path, raw)
    yield path, payload, Digest::SHA256.hexdigest(raw)
  ensure
    File.delete(path) if File.exists?(path)
    File.delete(payload) if File.exists?(payload)
  end
end

describe PixelUMMTwoBlockProbe do
  it "admits only the observed retained or alternating bank identity pattern" do
    PixelUMMTwoBlockProbe.validate_output_bank_layout!((0...16).to_a, 16, false)
    PixelUMMTwoBlockProbe.validate_output_bank_layout!(Array.new(16) { |slot| slot % 2 }, 16, true)
    expect_raises(ArgumentError, /bank/) do
      PixelUMMTwoBlockProbe.validate_output_bank_layout!(Array.new(16, 0), 16, true)
    end
    expect_raises(ArgumentError, /bank/) do
      PixelUMMTwoBlockProbe.validate_output_bank_layout!((0...16).to_a, 16, true)
    end
    wrong = Array.new(16) { |slot| slot % 2 }
    wrong[10] = 1
    expect_raises(ArgumentError, /bank/) do
      PixelUMMTwoBlockProbe.validate_output_bank_layout!(wrong, 16, true)
    end
    expect_raises(ArgumentError, /bank/) do
      PixelUMMTwoBlockProbe.validate_output_bank_layout!([0, 1], 16, true)
    end
  end

  it "rejects a valid-looking receipt digest from another source" do
    source = JSON.parse({checkpoint_subset_sha256: "a" * 64, source_receipt_sha256: "b" * 64,
                         checkpoint_subset_bytes: 1000, source_extent: [20, 1020]}.to_json)
    cache = JSON.parse({selected_block: 1, subset_sha256: "a" * 64,
                        source_receipt_sha256: "b" * 64, subset_bytes: 1000, source_extent: [20, 1020]}.to_json)
    PixelUMMTwoBlockProbe.validate_receipt_binding!(cache, source, 1)
    cache.as_h["source_receipt_sha256"] = JSON::Any.new("c" * 64)
    expect_raises(ArgumentError, /receipt/) { PixelUMMTwoBlockProbe.validate_receipt_binding!(cache, source, 1) }
  end

  it "pins source revision and header identity before tensor payload reads" do
    identity = JSON.parse({upstream_revision:        PixelUMMBlockProbe::UPSTREAM_REVISION,
                           upstream_model_sha256:    PixelUMMTwoBlockProbe::MODEL_SHA256,
                           hf_revision:              PixelUMMTwoBlockProbe::HF_REVISION,
                           hf_source_path:           PixelUMMTwoBlockProbe::SOURCE_PATH,
                           checkpoint_header_sha256: PixelUMMTwoBlockProbe::HEADER_SHA256}.to_json)
    PixelUMMTwoBlockProbe.validate_identity!(identity)
    {"hf_revision", "checkpoint_header_sha256"}.each do |key|
      changed = JSON.parse(identity.to_json)
      changed.as_h[key] = JSON::Any.new("0" * (key == "hf_revision" ? 40 : 64))
      expect_raises(ArgumentError, /identity/) { PixelUMMTwoBlockProbe.validate_identity!(changed) }
    end
  end

  it "rejects a same-byte-size capture dtype relabel instead of halving word coverage" do
    value = JSON.parse({dtype: "F32", shape: [1, 2048, 2048], offset: 0,
                        nbytes: 2048_i64 * 4096 * 2, sha256: "a" * 64}.to_json)
    record = PixelUMMTwoBlockProbe::Record.new(value, 2048_i64 * 4096 * 2)
    expect_raises(ArgumentError, /dtype/) do
      PixelUMMTwoBlockProbe.validate_capture!("block0.output", record, 2048)
    end
    value.as_h["dtype"] = JSON::Any.new("BF16")
    value.as_h["shape"] = JSON.parse([1, 2048, 4096].to_json)
    record = PixelUMMTwoBlockProbe::Record.new(value, 2048_i64 * 4096 * 2)
    PixelUMMTwoBlockProbe.validate_capture!("block0.output", record, 2048)
  end

  it "admits exactly the pinned BF16 weight geometry for both blocks without Metal" do
    weights = pixelumm_pair_weight_records
    PixelUMMTwoBlockProbe.validate_weights!(weights)
    weights.size.should eq(26)
    weights.values.sum(&.nbytes).should eq(771_785_728_i64)
  end

  it "rejects a same-sized swapped channel layout and a wrong block owner" do
    weights = pixelumm_pair_weight_records
    record = weights["block1.attn.k_proj.weight"]
    value = JSON.parse({dtype: "BF16", shape: [4096, 1024], offset: record.offset,
                        nbytes: record.nbytes, sha256: record.sha256}.to_json)
    weights["block1.attn.k_proj.weight"] = PixelUMMTwoBlockProbe::Record.new(value, 2_i64 * 1024**3)
    expect_raises(ArgumentError, /shape/) { PixelUMMTwoBlockProbe.validate_weights!(weights) }
    weights = pixelumm_pair_weight_records
    weights["block2.attn.q_proj.weight"] = weights.delete("block1.attn.q_proj.weight").not_nil!
    expect_raises(ArgumentError, /missing/) { PixelUMMTwoBlockProbe.validate_weights!(weights) }
  end

  it "admits exactly four owners and rejects missing, extra or swapped fourth-block weights" do
    weights = pixelumm_pair_weight_records(4)
    PixelUMMTwoBlockProbe.validate_weights!(weights, 4)
    weights.size.should eq(52)
    weights.values.sum(&.nbytes).should eq(1_543_571_456_i64)
    expect_raises(ArgumentError, /weights/) { PixelUMMTwoBlockProbe.validate_weights!(weights) }
    weights["block4.attn.q_proj.weight"] = weights.delete("block3.attn.q_proj.weight").not_nil!
    expect_raises(ArgumentError, /missing/) { PixelUMMTwoBlockProbe.validate_weights!(weights, 4) }
    weights = pixelumm_pair_weight_records(4)
    weights.delete("block3.mlp.down_proj.weight")
    expect_raises(ArgumentError, /weights/) { PixelUMMTwoBlockProbe.validate_weights!(weights, 4) }
    expect_raises(ArgumentError, /count/) { PixelUMMTwoBlockProbe.validate_weights!(pixelumm_pair_weight_records, 3) }
  end

  it "requires the immediate predecessor in all four metadata handoffs" do
    4.times do |slot|
      expected = slot == 0 ? "inputs.x" : "captures.block#{slot - 1}.output"
      PixelUMMTwoBlockProbe.validate_input_source!(expected, slot)
    end
    {2, 3}.each do |slot|
      expect_raises(ArgumentError, /predecessor/) do
        PixelUMMTwoBlockProbe.validate_input_source!("captures.block0.output", slot)
      end
    end
  end

  it "admits exactly eight weight owners and all seven direct GPU handoffs" do
    weights = pixelumm_pair_weight_records(8)
    PixelUMMTwoBlockProbe.validate_weights!(weights, 8)
    weights.size.should eq(104)
    weights.values.sum(&.nbytes).should eq(3_087_142_912_i64)
    8.times do |slot|
      expected = slot == 0 ? "inputs.x" : "captures.block#{slot - 1}.output"
      PixelUMMTwoBlockProbe.validate_input_source!(expected, slot)
    end
    weights["block8.attn.q_proj.weight"] = weights.delete("block7.attn.q_proj.weight").not_nil!
    expect_raises(ArgumentError, /missing/) { PixelUMMTwoBlockProbe.validate_weights!(weights, 8) }
    expect_raises(ArgumentError, /count/) { PixelUMMTwoBlockProbe.validate_weights!(pixelumm_pair_weight_records(8), 7) }
    expect_raises(ArgumentError, /predecessor/) do
      PixelUMMTwoBlockProbe.validate_input_source!("captures.block5.output", 7)
    end
  end

  it "admits exactly sixteen weight owners and the fifteen direct GPU handoffs" do
    weights = pixelumm_pair_weight_records(16)
    PixelUMMTwoBlockProbe.validate_weights!(weights, 16)
    weights.size.should eq(208)
    weights.values.sum(&.nbytes).should eq(6_174_285_824_i64)
    16.times do |slot|
      expected = slot == 0 ? "inputs.x" : "captures.block#{slot - 1}.output"
      PixelUMMTwoBlockProbe.validate_input_source!(expected, slot)
    end
    PixelUMMTwoBlockProbe.capture_slot("block15.output", 16).should eq(15)
    expect_raises(ArgumentError, /missing/) do
      weights["block16.attn.q_proj.weight"] = weights.delete("block15.attn.q_proj.weight").not_nil!
      PixelUMMTwoBlockProbe.validate_weights!(weights, 16)
    end
    expect_raises(ArgumentError, /count/) { PixelUMMTwoBlockProbe.validate_weights!(pixelumm_pair_weight_records(16), 17) }
    expect_raises(ArgumentError, /predecessor/) do
      PixelUMMTwoBlockProbe.validate_input_source!("captures.block0.output", 15)
    end
  end

  it "keeps prior depth ceilings while setting explicit sixteen-block byte ceilings" do
    {2, 4, 8}.each do |count|
      PixelUMMTwoBlockProbe.validate_block_count!(count)
      PixelUMMTwoBlockProbe.named_budget_limit(count).should eq(8_i64 * 1024**3)
      PixelUMMTwoBlockProbe.weight_cache_payload_limit(count).should eq(4_i64 * 1024**3)
    end
    PixelUMMTwoBlockProbe.fixture_payload_limit(2).should eq(4_i64 * 1024**3)
    PixelUMMTwoBlockProbe.fixture_payload_limit(4).should eq(4_i64 * 1024**3)
    PixelUMMTwoBlockProbe.fixture_payload_limit(8).should eq(8_i64 * 1024**3)
    PixelUMMTwoBlockProbe.fixture_payload_limit(16).should eq(16_i64 * 1024**3)
    PixelUMMTwoBlockProbe.weight_cache_payload_limit(16).should eq(8_i64 * 1024**3)
    PixelUMMTwoBlockProbe.named_budget_limit(16).should eq(16_i64 * 1024**3)
    PixelUMMTwoBlockProbe.fixture_schema(16).should eq("cogni_ml.pixelumm_sixteen_block_reference/v1")
    expect_raises(ArgumentError, /count/) { PixelUMMTwoBlockProbe.validate_block_count!(17) }
    expect_raises(ArgumentError, /count/) { PixelUMMTwoBlockProbe.named_budget_limit(17) }
  end

  it "requires the exact sixteen-block real-context workspace byte budget" do
    config = PixelUMMTwoBlockProbe::Decoder::Config.new(temporal: 6, height: 44, width: 80)
    slot_weights = PixelUMMBlockProbe.weight_shapes.to_h do |name, shape|
      {name, shape.reduce(2_i64) { |bytes, extent| bytes * extent }}
    end
    weights_by_slot = Array.new(16) { slot_weights.dup }
    captures_by_slot = Array.new(16) { ["attention", "output"] }
    tokens = 6 * 44 * 80
    input_bytes = tokens.to_i64 * 4096 * 2
    table_bytes = tokens.to_i64 * 128 * 4
    required = PixelUMMTwoBlockProbe::Decoder::Workspace.validate_admission!(config,
      input_bytes: input_bytes, cosine_bytes: table_bytes, sine_bytes: table_bytes,
      weights_bytes_by_slot: weights_by_slot, captures_by_slot: captures_by_slot,
      max_unique_bytes: Int64::MAX)
    required.should eq(14_202_158_052_i64)
    PixelUMMTwoBlockProbe::Decoder::Workspace.validate_admission!(config,
      input_bytes: input_bytes, cosine_bytes: table_bytes, sine_bytes: table_bytes,
      weights_bytes_by_slot: weights_by_slot, captures_by_slot: captures_by_slot,
      max_unique_bytes: required).should eq(required)
    expect_raises(ArgumentError, /budget/) do
      PixelUMMTwoBlockProbe::Decoder::Workspace.validate_admission!(config,
        input_bytes: input_bytes, cosine_bytes: table_bytes, sine_bytes: table_bytes,
        weights_bytes_by_slot: weights_by_slot, captures_by_slot: captures_by_slot,
        max_unique_bytes: required - 1)
    end
  end

  it "requires complete sixteen-block capture inventories at each admitted grid" do
    small = (0...16).flat_map do |slot|
      PixelUMMTwoBlockProbe::Decoder::Plan::CAPTURE_NAMES.map { |stage| "block#{slot}.#{stage}" }.to_a
    end
    full = (0...16).flat_map { |slot| ["block#{slot}.attention", "block#{slot}.output"] }
    small.size.should eq(416)
    full.size.should eq(32)
    PixelUMMTwoBlockProbe.validate_serial_capture_inventory!(small, [2, 32, 32], 16)
    PixelUMMTwoBlockProbe.validate_serial_capture_inventory!(full, [6, 44, 80], 16)
    expect_raises(ArgumentError, /every expected/) do
      PixelUMMTwoBlockProbe.validate_serial_capture_inventory!(full, [2, 32, 32], 16)
    end
    expect_raises(ArgumentError, /every expected/) do
      PixelUMMTwoBlockProbe.validate_serial_capture_inventory!(small[0...-1], [2, 32, 32], 16)
    end
  end

  it "validates capture records across the double-digit block boundary" do
    value = JSON.parse({dtype: "BF16", shape: [1, 2048, 4096], offset: 0,
                        nbytes: 2048_i64 * 4096 * 2, sha256: "a" * 64}.to_json)
    record = PixelUMMTwoBlockProbe::Record.new(value, 2048_i64 * 4096 * 2)
    {9, 10, 15}.each do |slot|
      PixelUMMTwoBlockProbe.validate_capture!("block#{slot}.output", record, 2048, 16)
      PixelUMMTwoBlockProbe.capture_stage("block#{slot}.output", 16).should eq("output")
    end
  end

  it "authenticates exact numeric block ownership without depending on string sort order" do
    {2, 4, 8, 16}.each do |count|
      keys = (0...count).map(&.to_s).reverse
      PixelUMMTwoBlockProbe.validate_cache_block_keys!(keys, count)
      missing = keys[0...-1]
      expect_raises(ArgumentError, /inventory/) { PixelUMMTwoBlockProbe.validate_cache_block_keys!(missing, count) }
      keys[0] = count.to_s
      expect_raises(ArgumentError, /inventory/) { PixelUMMTwoBlockProbe.validate_cache_block_keys!(keys, count) }
    end
  end

  it "rejects mixed FFN graph precision across sixteen serial slots" do
    {false, true}.each do |enabled|
      PixelUMMTwoBlockProbe.validate_ffn_precision_slots!(Array(Bool).new(16, enabled), 16, enabled)
    end
    mixed = Array(Bool).new(16, false)
    mixed[15] = true
    expect_raises(ArgumentError, /precision/) do
      PixelUMMTwoBlockProbe.validate_ffn_precision_slots!(mixed, 16, false)
    end
    expect_raises(ArgumentError, /precision/) do
      PixelUMMTwoBlockProbe.validate_ffn_precision_slots!(Array(Bool).new(15, false), 16, false)
    end
  end

  it "checks sixteen-block real-context input provenance metadata" do
    input_sha = "a" * 64
    record = PixelUMMTwoBlockProbe::Record.new(JSON.parse({dtype: "BF16", shape: [1, 1, 4096],
                                                           offset: 0, nbytes: 8192, sha256: input_sha}.to_json), 8192_i64)
    document = JSON.parse({input_provenance: {kind: "real_latent_xt_pixelumm_stem",
                                              latent_sha256: "b" * 64, xt_sha256: "c" * 64,
                                              stem_weights_sha256: "d" * 64,
                                              derived_x_sha256: input_sha}}.to_json)
    PixelUMMTwoBlockProbe.validate_real_input_provenance_metadata!(document, record)
    document["input_provenance"].as_h["kind"] = JSON::Any.new("seeded_standard_normal")
    expect_raises(ArgumentError, /provenance metadata/) do
      PixelUMMTwoBlockProbe.validate_real_input_provenance_metadata!(document, record)
    end
    document["input_provenance"].as_h["kind"] = JSON::Any.new("real_latent_xt_pixelumm_stem")
    document["input_provenance"].as_h["derived_x_sha256"] = JSON::Any.new("e" * 64)
    expect_raises(ArgumentError, /provenance metadata/) do
      PixelUMMTwoBlockProbe.validate_real_input_provenance_metadata!(document, record)
    end
    document["input_provenance"].as_h["derived_x_sha256"] = JSON::Any.new(input_sha)
    document["input_provenance"].as_h["extra_claim"] = JSON::Any.new("ignored")
    expect_raises(ArgumentError, /fields/) do
      PixelUMMTwoBlockProbe.validate_real_input_provenance_metadata!(document, record)
    end
  end

  it "limits eighth-block captures to the eight-block context" do
    value = JSON.parse({dtype: "BF16", shape: [1, 2048, 4096], offset: 0,
                        nbytes: 2048_i64 * 4096 * 2, sha256: "a" * 64}.to_json)
    record = PixelUMMTwoBlockProbe::Record.new(value, 2048_i64 * 4096 * 2)
    PixelUMMTwoBlockProbe.validate_capture!("block7.output", record, 2048, 8)
    expect_raises(ArgumentError, /ownership/) { PixelUMMTwoBlockProbe.validate_capture!("block7.output", record, 2048, 4) }
    expect_raises(ArgumentError, /ownership/) { PixelUMMTwoBlockProbe.validate_capture!("block8.output", record, 2048, 8) }
  end

  it "requires all 208 small stages or exactly 16 full boundaries for eight blocks" do
    small = (0...8).flat_map do |slot|
      PixelUMMTwoBlockProbe::Decoder::Plan::CAPTURE_NAMES.map { |stage| "block#{slot}.#{stage}" }.to_a
    end
    full = (0...8).flat_map { |slot| ["block#{slot}.attention", "block#{slot}.output"] }
    small.size.should eq(208)
    PixelUMMTwoBlockProbe.validate_eight_capture_inventory!(small, [2, 32, 32])
    PixelUMMTwoBlockProbe.validate_eight_capture_inventory!(full, [6, 44, 80])
    expect_raises(ArgumentError, /every expected/) { PixelUMMTwoBlockProbe.validate_eight_capture_inventory!(full, [2, 32, 32]) }
    expect_raises(ArgumentError, /every expected/) { PixelUMMTwoBlockProbe.validate_eight_capture_inventory!(small, [6, 44, 80]) }
    expect_raises(ArgumentError, /every expected/) { PixelUMMTwoBlockProbe.validate_eight_capture_inventory!(small[0...-1], [2, 32, 32]) }
  end

  it "admits fourth-block captures only in the four-block context" do
    value = JSON.parse({dtype: "BF16", shape: [1, 2048, 4096], offset: 0,
                        nbytes: 2048_i64 * 4096 * 2, sha256: "a" * 64}.to_json)
    record = PixelUMMTwoBlockProbe::Record.new(value, 2048_i64 * 4096 * 2)
    PixelUMMTwoBlockProbe.validate_capture!("block3.output", record, 2048, 4)
    expect_raises(ArgumentError, /ownership/) { PixelUMMTwoBlockProbe.validate_capture!("block3.output", record, 2048) }
    expect_raises(ArgumentError, /ownership/) { PixelUMMTwoBlockProbe.validate_capture!("block4.output", record, 2048, 4) }
  end

  {4, 8}.each do |block_count|
    it "rejects wrong predecessor metadata in a complete #{block_count}-slot fixture before payload authentication" do
      tokens = block_count == 8 ? 21120 : 2048
      with_pixelumm_tensor_file do |path, payload, _|
        weights = pixelumm_pair_weight_records(block_count).transform_values do |record|
          JSON.parse({dtype: record.dtype, shape: record.shape, offset: record.offset,
                      nbytes: record.nbytes, sha256: record.sha256}.to_json)
        end
        offset = block_count.to_i64 * 385_892_864
        make_record = ->(dtype : String, shape : Array(Int32)) do
          nbytes = shape.reduce(dtype == "BF16" ? 2_i64 : 4_i64) { |n, value| n * value }
          value = JSON.parse({dtype: dtype, shape: shape, offset: offset,
                              nbytes: nbytes, sha256: "a" * 64}.to_json)
          offset += nbytes
          value
        end
        inputs = {"x"   => make_record.call("BF16", [1, tokens, 4096]),
                  "cos" => make_record.call("F32", [1, tokens, 1, 128]),
                  "sin" => make_record.call("F32", [1, tokens, 1, 128])}
        captures = {} of String => JSON::Any
        blocks = Array.new(block_count) do |slot|
          {"attention", "output"}.each do |stage|
            shape = stage == "attention" ? [1, tokens, 32, 128] : [1, tokens, 4096]
            captures["block#{slot}.#{stage}"] = make_record.call("BF16", shape)
          end
          prefix = "block#{slot}."
          {index: slot, weight_prefix: prefix,
           input_source: slot == 0 ? "inputs.x" : "captures.block#{slot - 1}.output",
           output_key: "captures.block#{slot}.output",
           weight_keys: weights.keys.select(&.starts_with?(prefix)),
           capture_keys: captures.keys.select(&.starts_with?(prefix)),
           checkpoint_subset_sha256: slot == 0 ? PixelUMMBlockProbe::SUBSET_SHA256 : "b" * 64,
           source_receipt_sha256: "c" * 64}
        end
        document = JSON.parse({schema: block_count == 8 ? "cogni_ml.pixelumm_eight_block_reference/v1" : "cogni_ml.pixelumm_four_block_reference/v1",
                               upstream_revision: PixelUMMBlockProbe::UPSTREAM_REVISION,
                               upstream_model_sha256: PixelUMMTwoBlockProbe::MODEL_SHA256,
                               hf_revision: PixelUMMTwoBlockProbe::HF_REVISION,
                               hf_source_path: PixelUMMTwoBlockProbe::SOURCE_PATH,
                               checkpoint_header_sha256: PixelUMMTwoBlockProbe::HEADER_SHA256,
                               grid: block_count == 8 ? [6, 44, 80] : [2, 32, 32], tile_tokens: 8, token_order: "tile_major",
                               payload: File.basename(payload), payload_bytes: offset, payload_sha256: "d" * 64,
                               inputs: inputs, weights: weights, captures: captures,
                               capture_order: captures.keys, blocks: blocks}.to_json)
        # Sparse extent only: deliberately unauthenticated payload. Fixture
        # metadata admission must not stream it or allocate any GPU buffers.
        File.open(payload, "w") { |io| io.truncate(offset) }
        raw = document.to_json
        File.write(path, raw)
        fixture = PixelUMMTwoBlockProbe::Fixture.new(path, Digest::SHA256.hexdigest(raw), block_count)
        fixture.block_count.should eq(block_count)
        fixture.captures_by_slot.size.should eq(block_count)
        if block_count == 8
          (offset > 4_i64 * 1024**3).should be_true
          expect_raises(ArgumentError, /extent/) { PixelUMMTwoBlockProbe::TensorFile.new(path, Digest::SHA256.hexdigest(raw)) }
          expect_raises(ArgumentError, /extent/) { PixelUMMTwoBlockProbe::Fixture.new(path, Digest::SHA256.hexdigest(raw), 4) }
          oversized = JSON.parse(document.to_json)
          oversized.as_h["payload_bytes"] = JSON::Any.new(8_i64 * 1024**3 + 1)
          File.open(payload, "r+") { |io| io.truncate(8_i64 * 1024**3 + 1) }
          File.write(path, oversized.to_json)
          expect_raises(ArgumentError, /extent/) do
            PixelUMMTwoBlockProbe::Fixture.new(path, Digest::SHA256.hexdigest(oversized.to_json), 8)
          end
          File.open(payload, "r+") { |io| io.truncate(offset) }
          trailing = JSON.parse(document.to_json)
          trailing.as_h["payload_bytes"] = JSON::Any.new(offset + 1)
          File.open(payload, "r+") { |io| io.truncate(offset + 1) }
          File.write(path, trailing.to_json)
          expect_raises(ArgumentError, /trailing/) do
            PixelUMMTwoBlockProbe::Fixture.new(path, Digest::SHA256.hexdigest(trailing.to_json), 8)
          end
          trailing["captures"]["block7.output"].as_h["offset"] = JSON::Any.new(trailing["captures"]["block7.output"]["offset"].as_i64 + 1)
          File.write(path, trailing.to_json)
          expect_raises(ArgumentError, /missing/) do
            PixelUMMTwoBlockProbe::Fixture.new(path, Digest::SHA256.hexdigest(trailing.to_json), 8)
          end
          File.open(payload, "r+") { |io| io.truncate(offset) }
        end
        (2...block_count).each do |slot|
          changed = JSON.parse(document.to_json)
          changed["blocks"][slot].as_h["input_source"] = JSON::Any.new("captures.block0.output")
          raw = changed.to_json
          File.write(path, raw)
          expect_raises(ArgumentError, /predecessor/) do
            PixelUMMTwoBlockProbe::Fixture.new(path, Digest::SHA256.hexdigest(raw), block_count)
          end
        end
      end
    end
  end

  it "requires trusted metadata and rejects payload corruption before upload" do
    with_pixelumm_tensor_file do |path, payload, sha|
      expect_raises(ArgumentError, /identity/) { PixelUMMTwoBlockProbe::TensorFile.new(path, "0" * 64) }
      fixture = PixelUMMTwoBlockProbe::TensorFile.new(path, sha)
      File.write(payload, "abce")
      expect_raises(ArgumentError, /hash/) { fixture.verify_payload! }
    end
  end

  it "checks individual records and rejects overlaps and trailing cache bytes" do
    with_pixelumm_tensor_file do |path, payload, sha|
      fixture = PixelUMMTwoBlockProbe::TensorFile.new(path, sha)
      record = PixelUMMTwoBlockProbe::Record.new(JSON.parse({dtype: "BF16", shape: [1],
                                                             offset: 0, nbytes: 2, sha256: Digest::SHA256.hexdigest("ab")}.to_json), 4_i64)
      expect_raises(ArgumentError, /overlapping/) { fixture.validate_ranges!([record, record]) }
      expect_raises(ArgumentError, /trailing/) { fixture.validate_ranges!([record], contiguous: true) }
      expect_raises(ArgumentError, /authenticated/) { fixture.read(record) }
      fixture.verify_payload!
      String.new(fixture.read(record)).should eq("ab")
      File.write(payload, "xbcd")
      expect_raises(ArgumentError, /record hash/) { fixture.read(record) }
    end
  end
end
