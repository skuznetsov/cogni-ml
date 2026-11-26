require "spec"
require "../examples/support/hy_nomas_reference_attention_probe"
require "../examples/support/hy_nomas_reference_block0_probe"

describe HyNomasReferenceAttentionProbe do
  it "requires the explicit BF16 graph mode and source receipt identity" do
    HyNomasReferenceBlock0Probe.validate_precision_options!(HyNomasReferenceAttentionProbe::PRECISION, "bf16-graph", true)
    HyNomasReferenceAttentionProbe.validate_options!(HyNomasReferenceAttentionProbe::PRECISION, "/tmp/source.json", "a" * 64, false, "", "")
    expect_raises(ArgumentError) { HyNomasReferenceBlock0Probe.validate_precision_options!(HyNomasReferenceAttentionProbe::PRECISION, "f32-accum", true) }
    expect_raises(ArgumentError) { HyNomasReferenceAttentionProbe.validate_options!(HyNomasReferenceAttentionProbe::PRECISION, "", "", false, "", "") }
    expect_raises(ArgumentError) { HyNomasReferenceAttentionProbe.validate_options!(HyNomasReferenceAttentionProbe::PRECISION, "/tmp/source.json", "a" * 64, true, "", "") }
    expect_raises(ArgumentError) { HyNomasReferenceAttentionProbe.validate_options!("f32-control", "/tmp/source.json", "a" * 64, false, "", "") }
  end

  it "binds four physical BF16 tensors and accounts for the complete safetensors file" do
    fixtures = HyNomasReferenceAttentionProbe::FIXTURE_KEYS.to_h { |key| {key, "a" * 64} }
    header = HyNomasReferenceAttentionProbe.header(fixtures)
    (header.bytesize % 8).should eq(0)
    parsed = JSON.parse(header)
    parsed["__metadata__"]["format"].as_s.should eq(HyNomasReferenceAttentionProbe::SCHEMA)
    HyNomasReferenceAttentionProbe::TENSOR_NAMES.each_with_index do |name, index|
      tensor = parsed[name]
      tensor["dtype"].as_s.should eq("BF16")
      tensor["shape"].as_a.map(&.as_i).should eq([1, 4849, 16, 128])
      tensor["data_offsets"].as_a.map(&.as_i64).should eq([index.to_i64 * 19_861_504, (index.to_i64 + 1) * 19_861_504])
    end
    file_bytes = 8_i64 + header.bytesize + 4_i64 * HyNomasReferenceAttentionProbe::TENSOR_BYTES
    named = {"reference_attention_buffers" => HyNomasReferenceAttentionProbe::NAMED_BYTES,
             "total_named_bytes"           => HyNomasReferenceAttentionProbe::NAMED_BYTES}
    receipt = HyNomasReferenceAttentionProbe.receipt_json("native.safetensors", "b" * 64, file_bytes,
      Array.new(4, "c" * 64), fixtures, "d" * 64, named, 0_i64, 0_i64, 0, 1)
    document = JSON.parse(receipt)
    document["tensor_artifact"]["bytes"].as_i64.should eq(file_bytes)
    document["attention_executed"].as_bool.should be_true
    document["cache_append_executed"].as_bool.should be_false
    document["source_attention_receipt_sha256"].as_s.should eq("d" * 64)
    expect_raises(ArgumentError) { HyNomasReferenceAttentionProbe.receipt_json("native.safetensors", "b" * 64, file_bytes,
      Array.new(4, "c" * 64), fixtures, "d" * 64, named, 1_i64, 0_i64, 0, 1) }
    expect_raises(ArgumentError) { HyNomasReferenceAttentionProbe.receipt_json("native.safetensors", "b" * 64, file_bytes - 8,
      Array.new(4, "c" * 64), fixtures, "d" * 64, named, 0_i64, 0_i64, 0, 1) }
  end
end
