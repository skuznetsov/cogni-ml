require "spec"
require "../examples/support/pixelumm_two_block_probe"

private def pixelumm_pair_weight_records : Hash(String, PixelUMMTwoBlockProbe::Record)
  result = {} of String => PixelUMMTwoBlockProbe::Record
  offset = 0_i64
  2.times do |slot|
    PixelUMMBlockProbe.weight_shapes.each do |name, shape|
      nbytes = shape.reduce(2_i64) { |n, v| n * v }
      value = JSON.parse({dtype: "BF16", shape: shape, offset: offset,
                          nbytes: nbytes, sha256: "a" * 64}.to_json)
      result["block#{slot}.#{name}"] = PixelUMMTwoBlockProbe::Record.new(value, 2_i64 * 1024**3)
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
