require "spec"
require "../examples/support/pixelumm_ffn_isolated_probe"

private alias IsolatedRecord = NamedTuple(dtype: String, shape: Array(Int32), offset: Int64, nbytes: Int64, sha256: String)
private DIGEST = "a" * 64

private def isolated_record(shape : Array(Int32), offset : Int64) : IsolatedRecord
  nbytes = shape.reduce(1_i64) { |product, dimension| product * dimension } * 2
  {dtype: "BF16", shape: shape, offset: offset, nbytes: nbytes, sha256: DIGEST}
end

private def isolated_fixture_document(grid : Array(Int32)) : JSON::Any
  tokens = grid.product
  groups = {
    "inputs"   => {} of String => IsolatedRecord,
    "weights"  => {} of String => IsolatedRecord,
    "captures" => {} of String => IsolatedRecord,
  }
  entries = [
    {"inputs", "post_attn_norm", [1, tokens, 4096]},
    {"inputs", "product", [1, tokens, 12288]},
    {"weights", "mlp.gate_proj.weight", [12288, 4096]},
    {"weights", "mlp.up_proj.weight", [12288, 4096]},
    {"weights", "mlp.down_proj.weight", [4096, 12288]},
    {"captures", "gate_proj", [1, tokens, 12288]},
    {"captures", "up_proj", [1, tokens, 12288]},
    {"captures", "down_proj", [1, tokens, 4096]},
  ] of Tuple(String, String, Array(Int32))
  offset = 0_i64
  entries.each do |group, name, shape|
    record = isolated_record(shape, offset)
    groups[group][name] = record
    offset += record[:nbytes]
  end

  source_records = {} of String => IsolatedRecord
  parent_records = {} of String => IsolatedRecord
  source_entries = [
    {"inputs.x", "x", [1, tokens, 4096]},
    {"captures.block0.attention", "attention", [1, tokens, 32, 128]},
    {"weights.block0.attn.o_proj.weight", "attn.o_proj.weight", [4096, 4096]},
    {"weights.block0.post_attn_norm.weight", "post_attn_norm.weight", [4096]},
    {"weights.block0.mlp.gate_proj.weight", "mlp.gate_proj.weight", [12288, 4096]},
    {"weights.block0.mlp.up_proj.weight", "mlp.up_proj.weight", [12288, 4096]},
    {"weights.block0.mlp.down_proj.weight", "mlp.down_proj.weight", [4096, 12288]},
  ] of Tuple(String, String, Array(Int32))
  parent_offset = 0_i64
  source_entries.each do |source_name, parent_name, shape|
    record = isolated_record(shape, parent_offset)
    source_records[source_name] = record
    parent_records[parent_name] = record
    parent_offset += record[:nbytes]
  end
  full = tokens == 21120
  parent_schema = full ? "cogni_ml.pixelumm_thirty_six_block_reference/v1" : "cogni_ml.pixelumm_block_reference/v1"
  receipt = full ? DIGEST : nil
  source_replay = {
    "kind"                                => "saved_block0_attention_upstream_suffix_mps",
    "parent_schema"                       => parent_schema,
    "parent_manifest_sha256"              => DIGEST,
    "parent_payload_sha256"               => DIGEST,
    "parent_payload_bytes"                => parent_offset + 1_i64,
    "source_metadata_sha256"              => DIGEST,
    "source_payload_sha256"               => DIGEST,
    "source_receipt_sha256"               => receipt,
    "selected_parent_records"             => parent_records,
    "source_records"                      => source_records,
    "selected_record_hashes_verified"     => true,
    "parent_payload_full_hash_recomputed" => false,
    "block0_subset_sha256"                => PixelUMMFFNIsolatedProbe::SUBSET_SHA256,
    "parent_input_provenance"             => (full ? {"kind" => "real_latent_xt_pixelumm_stem"} : nil),
    "parent_block0"                       => (full ? {"source_receipt_sha256" => DIGEST} : nil),
    "cpu_fallback"                        => false,
  }
  JSON.parse({
    "schema"                => PixelUMMFFNIsolatedProbe::SCHEMA,
    "upstream_revision"     => PixelUMMFFNIsolatedProbe::UPSTREAM_REVISION,
    "upstream_model_sha256" => PixelUMMFFNIsolatedProbe::MODEL_SHA256,
    "model_sha256"          => PixelUMMFFNIsolatedProbe::MODEL_SHA256,
    "grid"                  => grid,
    "comparison"            => {"atol" => 2e-5, "rtol" => 2e-5},
    "token_order"           => "tile_major",
    "tile_tokens"           => 8,
    "runtime_dtype"         => "BF16",
    "payload_bytes"         => offset,
    "payload_sha256"        => DIGEST,
    "inputs"                => groups["inputs"],
    "weights"               => groups["weights"],
    "captures"              => groups["captures"],
    "capture_order"         => PixelUMMFFNIsolatedProbe::CAPTURE_NAMES,
    "source_replay"         => source_replay,
  }.to_json)
end

private def isolated_words(values : Array(UInt16)) : Bytes
  bytes = Bytes.new(values.size * 2)
  values.each_with_index { |value, index| IO::ByteFormat::LittleEndian.encode(value, bytes[index * 2, 2]) }
  bytes
end

describe PixelUMMFFNIsolatedProbe do
  describe "fixture inventory and provenance" do
    it "accepts true J2048 and full-J metadata with exact disjoint payload coverage" do
      small = PixelUMMFFNIsolatedProbe::Metadata.new(isolated_fixture_document([2, 32, 32]))
      small.tokens.should eq(2048)
      small.inputs.keys.sort.should eq(PixelUMMFFNIsolatedProbe::INPUT_NAMES.sort)
      small.weights.keys.sort.should eq(PixelUMMFFNIsolatedProbe::WEIGHT_NAMES.sort)
      small.captures.keys.sort.should eq(PixelUMMFFNIsolatedProbe::CAPTURE_NAMES.sort)

      full = PixelUMMFFNIsolatedProbe::Metadata.new(isolated_fixture_document([6, 44, 80]))
      full.tokens.should eq(21120)
      full.inputs["post_attn_norm"].shape.should eq([1, 21120, 4096])
      full.captures["down_proj"].shape.should eq([1, 21120, 4096])
    end

    it "rejects missing or overlapping records instead of accepting a mixed inventory" do
      missing = isolated_fixture_document([2, 32, 32])
      missing["captures"].as_h.delete("up_proj")
      expect_raises(ArgumentError) { PixelUMMFFNIsolatedProbe::Metadata.new(missing) }

      overlapping = isolated_fixture_document([2, 32, 32])
      records = overlapping["source_replay"]["source_records"].as_h
      parent_records = overlapping["source_replay"]["selected_parent_records"].as_h
      source_key = "weights.block0.mlp.up_proj.weight"
      parent_key = "mlp.up_proj.weight"
      changed = JSON.parse({"dtype" => "BF16", "shape" => [12288, 4096], "offset" => 0,
                            "nbytes" => 100663296, "sha256" => DIGEST}.to_json)
      records[source_key] = changed
      parent_records[parent_key] = changed
      expect_raises(ArgumentError) { PixelUMMFFNIsolatedProbe::Metadata.new(overlapping) }
    end
  end

  it "admits only a complete named-buffer budget before any Metal allocation" do
    full_bytes = PixelUMMFFNIsolatedProbe.named_bytes(21120)
    full_bytes.should eq(3_589_276_048_i64)
    PixelUMMFFNIsolatedProbe.validate_budget!(21120, 4_i64 * PixelUMMFFNIsolatedProbe::GIB).should eq(full_bytes)
    expect_raises(ArgumentError) { PixelUMMFFNIsolatedProbe.validate_budget!(21120, full_bytes - 1) }
    PixelUMMFFNIsolatedProbe.validate_budget!(2048, 4_i64 * PixelUMMFFNIsolatedProbe::GIB).should be < full_bytes
  end

  it "keeps warmup/preparation reversal independent from measured order and stage order" do
    modes = ->(warmups : Bool, first : Bool) do
      PixelUMMFFNIsolatedProbe.schedule(warmups, first).map(&.[:mode])
    end
    modes.call(false, false).should eq(["separate", "shared", "separate", "shared", "shared", "separate"])
    modes.call(true, false).should eq(["shared", "separate", "separate", "shared", "shared", "separate"])
    modes.call(false, true).should eq(["separate", "shared", "shared", "separate", "separate", "shared"])
    PixelUMMFFNIsolatedProbe.plan_order(false).should eq(PixelUMMFFNIsolatedProbe::PLAN_NAMES)
    PixelUMMFFNIsolatedProbe.plan_order(true).should eq(PixelUMMFFNIsolatedProbe::PLAN_NAMES.reverse)
    PixelUMMFFNIsolatedProbe.stage_order(false).should eq(["gate_up", "down"])
    PixelUMMFFNIsolatedProbe.stage_order(true).should eq(["down", "gate_up"])
  end

  it "uses the pinned tolerance and rejects nonfinite values in the CPU audit comparator" do
    expected = isolated_words([0x3f80_u16, 0x4000_u16, 0x0000_u16])
    within = isolated_words([0x3f80_u16, 0x4000_u16, 0x3700_u16])
    PixelUMMFFNIsolatedProbe.compare(expected, within).should eq(
      {failures: 0_u32, different_words: 1_u32, nonfinite: 0_u32, elements: 3_u32})

    outside = isolated_words([0x3f81_u16, 0x4000_u16, 0x0000_u16])
    PixelUMMFFNIsolatedProbe.compare(expected, outside).should eq(
      {failures: 1_u32, different_words: 1_u32, nonfinite: 0_u32, elements: 3_u32})

    nonfinite = isolated_words([0x7fc0_u16, 0x4000_u16, 0x0000_u16])
    PixelUMMFFNIsolatedProbe.compare(expected, nonfinite).should eq(
      {failures: 1_u32, different_words: 1_u32, nonfinite: 1_u32, elements: 3_u32})
  end

  it "requires one successful terminal wait after any blit or graph encode attempt" do
    lifecycle = PixelUMMFFNIsolatedProbe::SubmissionLifecycle.new
    lifecycle.release_allowed?(false, 0).should be_true

    lifecycle.mark_attempt! # Before poison/canonical blit encoder creation.
    lifecycle.release_allowed?(true, 1).should be_false
    expect_raises(ArgumentError) { lifecycle.mark_terminal_wait!(false, 1) }
    expect_raises(ArgumentError) { lifecycle.mark_terminal_wait!(true, 2) }
    lifecycle.terminal_wait_succeeded?.should be_false

    lifecycle.mark_terminal_wait!(true, 1)
    lifecycle.release_allowed?(true, 1).should be_true
    expect_raises(ArgumentError) { lifecycle.mark_attempt! } # Even partial graph encodes stay owned.
  end

  it "counts canonical BF16 word differences exactly" do
    canonical = isolated_words([0x3f80_u16, 0x4000_u16, 0x0000_u16])
    same = isolated_words([0x3f80_u16, 0x4000_u16, 0x0000_u16])
    changed = isolated_words([0x3f80_u16, 0x4001_u16, 0x0001_u16])
    PixelUMMFFNIsolatedProbe.count_word_differences(canonical, same).should eq(0_u32)
    PixelUMMFFNIsolatedProbe.count_word_differences(canonical, changed).should eq(2_u32)
    expect_raises(ArgumentError) do
      PixelUMMFFNIsolatedProbe.count_word_differences(canonical, Bytes.new(3, 0_u8))
    end
  end
end
