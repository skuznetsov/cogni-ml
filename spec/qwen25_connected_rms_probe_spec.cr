require "spec"
require "../examples/support/qwen25_connected_rms_probe"

private def connected_gain_fixture(&block : String, JSON::Any, String ->) : Nil
  root = File.join(Dir.tempdir, "qwen25-connected-rms-#{Random.rand(1_000_000_000)}")
  Dir.mkdir(root)
  shard = File.join(root, "gain.safetensors")
  begin
    raw = Bytes[0x80_u8, 0x3f_u8, 0_u8, 0x40_u8]
    metadata = JSON.parse({key: Qwen25RMSStatisticsProbe::NORM_WEIGHT_KEY,
                           shard: "gain.safetensors", dtype: "BF16", shape: [2],
                           nbytes: 4, tensor_sha256: Digest::SHA256.hexdigest(raw)}.to_json)
    header = {Qwen25RMSStatisticsProbe::NORM_WEIGHT_KEY => {
      "dtype" => "BF16", "shape" => [2], "data_offsets" => [2, 6],
    }}.to_json
    File.open(shard, "wb") do |io|
      io.write_bytes(header.bytesize.to_u64, IO::ByteFormat::LittleEndian)
      io << header
      io.write(Bytes[0xff_u8, 0xff_u8])
      io.write(raw)
      io.write(Bytes[0xee_u8])
    end
    yield root, metadata, shard
  ensure
    File.delete(shard) if File.exists?(shard)
    Dir.delete(root)
  end
end

describe Qwen25ConnectedRMSProbe do
  it "admits the exact named footprint and rejects a one-byte-short budget" do
    Qwen25ConnectedRMSProbe.required_named_bytes(2, 4).should eq(228)
    Qwen25ConnectedRMSProbe.required_named_bytes(1280, 3584).should eq(100_958_212)
    Qwen25ConnectedRMSProbe.admit_named_bytes!(2, 4, 228).should eq(228)
    expect_raises(ArgumentError, /budget rejected/) do
      Qwen25ConnectedRMSProbe.admit_named_bytes!(2, 4, 227)
    end
    expect_raises(ArgumentError) { Qwen25ConnectedRMSProbe.output_layout(1281, 3584) }
    expect_raises(ArgumentError) { Qwen25ConnectedRMSProbe.output_layout(1, 0) }
  end

  it "labels the production scalar only after its in-place root overwrite" do
    layout = Qwen25ConnectedRMSProbe.output_layout(2, 4)
    layout.size.should eq(13)
    layout.select { |output| output[:arm] == "fed_production" }.map(&.[:suffix]).should eq(
      %w(rsqrt_f32 normalized_bf16 gained_output_bf16))
  end

  it "copies only the authenticated selected gain extent" do
    connected_gain_fixture do |root, metadata, shard|
      gain = Qwen25ConnectedRMSProbe.load_gain(root, metadata, 2)
      gain[:raw].should eq(Bytes[0x80_u8, 0x3f_u8, 0_u8, 0x40_u8])
      gain[:tensor_sha256].should eq(metadata["tensor_sha256"].as_s)
      gain[:path].should eq(File.realpath(shard))
      gain[:shard_nbytes].should eq(File.size(shard))
    end
  end

  it "rejects same-size gain mutation and invalid tensor extents" do
    connected_gain_fixture do |root, metadata, shard|
      File.open(shard, "r+b") do |io|
        io.seek(-3, IO::Seek::End)
        io.write(Bytes[0xff_u8])
      end
      expect_raises(ArgumentError, /digest mismatch/) do
        Qwen25ConnectedRMSProbe.load_gain(root, metadata, 2)
      end
    end
    connected_gain_fixture do |root, metadata, shard|
      header = {Qwen25RMSStatisticsProbe::NORM_WEIGHT_KEY => {
        "dtype" => "BF16", "shape" => [2], "data_offsets" => [0, 4],
      }}.to_json
      File.open(shard, "wb") do |io|
        io.write_bytes(header.bytesize.to_u64, IO::ByteFormat::LittleEndian)
        io << header
      end
      expect_raises(ArgumentError, /extent outside/) do
        Qwen25ConnectedRMSProbe.load_gain(root, metadata, 2)
      end
    end
  end

  it "distinguishes a tiny raw BF16 difference from the unchanged tolerance" do
    result = Qwen25ConnectedRMSProbe.compare_bf16("tiny", Bytes[0_u8, 0_u8], Bytes[1_u8, 0_u8], 1_i64)
    result[:raw_unequal].should eq(1)
    result[:finite_failures].should eq(0)
    result[:first_raw_unequal][0][:actual_bits].should eq("0001")
    result = Qwen25ConnectedRMSProbe.compare_bf16("ulp", Bytes[0x80_u8, 0x3f_u8], Bytes[0x81_u8, 0x3f_u8], 1_i64)
    result[:finite_failures].should eq(1)
  end

  it "rejects nonfinite BF16 values even if their words agree" do
    nonfinite = Bytes[0x80_u8, 0x7f_u8, 0xc0_u8, 0x7f_u8]
    result = Qwen25ConnectedRMSProbe.compare_bf16("nonfinite", nonfinite, nonfinite, 1_i64)
    result[:raw_unequal].should eq(0)
    result[:finite_failures].should eq(2)
    expect_raises(ArgumentError) do
      Qwen25ConnectedRMSProbe.compare_bf16("unaligned", Bytes[0_u8], Bytes[0_u8], 1_i64)
    end
  end
end
