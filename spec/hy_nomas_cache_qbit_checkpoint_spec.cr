require "spec"
require "digest/sha256"
require "file/tempfile"
require "../src/ml/video/hy_nomas_resume_bundle"
require "../examples/support/hy_nomas_cache_qbit_checkpoint"

private def hy_nomas_qbit_checkpoint_sha256(path : String) : String
  digest = Digest::SHA256.new
  File.open(path, "rb") do |io|
    chunk = Bytes.new(64 * 1024)
    while (count = io.read(chunk)) > 0
      digest.update(chunk[0, count])
    end
  end
  digest.hexfinal
end

private def hy_nomas_qbit_checkpoint_write(path : String, bytes : Bytes) : String
  ML::Video::HyNomasResumeBundle.write(path, "contract",
    {"record" => bytes.size.to_i64}) do |_name, io|
    io.write(bytes)
  end
end

describe ML::Video::HyNomasResumeBundle do
  it "opens an authenticated bounded inventory and streams variable records" do
    path = File.tempname("hy-nomas-qbit-bounded", ".bundle")
    expected = Bytes.new(70_003) { |index| (index % 251).to_u8 }
    digest = hy_nomas_qbit_checkpoint_write(path, expected)
    reader = ML::Video::HyNomasResumeBundle.open_bounded(path, digest,
      "contract", {"record" => 80_000_i64})
    begin
      reader.inventory.should eq({"record" => expected.size.to_i64})
      copy = Bytes.new(expected.size)
      reader.copy_to!("record", copy)
      copy.should eq(expected)
      output = IO::Memory.new
      reader.each_chunk("record") do |chunk|
        chunk.size.should be <= ML::Video::HyNomasResumeBundle::IO_CHUNK_BYTES
        output.write(chunk)
      end
      output.to_slice.should eq(expected)
      consumed_digest = Digest::SHA256.new
      consumed_digest.update(expected)
      reader.verify_record_digest!("record", consumed_digest.hexfinal)
      expect_raises(ArgumentError, /maximum/) do
        ML::Video::HyNomasResumeBundle.open_bounded(path, digest,
          "contract", {"record" => 1_i64})
      end
    ensure
      reader.close
      File.delete(path) if File.exists?(path)
    end
  end

  it "rejects same-inode record mutation after file authentication even when mtime is restored" do
    path = File.tempname("hy-nomas-qbit-range-mutation", ".bundle")
    expected = Bytes.new(70_003) { |index| (index % 251).to_u8 }
    digest = hy_nomas_qbit_checkpoint_write(path, expected)
    reader = ML::Video::HyNomasResumeBundle.open_bounded(path, digest,
      "contract", {"record" => 80_000_i64})
    begin
      before = File.info(path, false)
      File.open(path, "r+b") do |io|
        io.seek(ML::Video::HyNomasResumeBundle::PAYLOAD_OFFSET.to_i64)
        byte = io.read_byte.not_nil!
        io.seek(ML::Video::HyNomasResumeBundle::PAYLOAD_OFFSET.to_i64)
        io.write_byte(byte ^ 0xff_u8)
      end
      File.utime(Time.utc(2020, 1, 1), before.modification_time, path)

      consumed = Bytes.new(expected.size)
      reader.read_range!("record", 0_i64, consumed)
      consumed_digest = Digest::SHA256.new
      consumed_digest.update(consumed)
      expect_raises(ArgumentError, /SHA-256/) do
        reader.verify_record_digest!("record", consumed_digest.hexfinal)
      end
    ensure
      reader.close
      File.delete(path) if File.exists?(path)
    end
  end
end

private def hy_nomas_qbit_checkpoint_stats(name : String, tier : UInt32) : String
  tier_name = case tier
              when 4_u32 then "p4"
              when 5_u32 then "p5"
              when 8_u32 then "p8"
              else            "f32"
              end
  payload_bytes = HyNomasCacheQBitCheckpoint.payload_bytes_for_tier(tier)
  logical_bytes = payload_bytes + HyNomasCacheQBit::HEADER_BYTES
  JSON.build do |json|
    json.object do
      json.field "records" do
        json.array do
          json.object do
            json.field "name", name
            json.field "count", 128
            json.field "rows", 1
            json.field "logical_packed_bytes", logical_bytes
            json.field "exact_bytes", 512
            json.field "payload_bytes", payload_bytes
            json.field "tier_counts" do
              json.object do
                {"p4", "p5", "p8", "f32"}.each do |entry|
                  json.field entry, entry == tier_name ? 1 : 0
                end
              end
            end
            json.field "invalid_rows", 0
          end
        end
      end
      json.field "logical_packed_bytes", logical_bytes
      json.field "exact_bytes", 512
      json.field "tier_counts" do
        json.object do
          {"p4", "p5", "p8", "f32"}.each do |entry|
            json.field entry, entry == tier_name ? 1 : 0
          end
        end
      end
      json.field "invalid_rows", 0
    end
  end
end

private def hy_nomas_qbit_checkpoint_fixture(path : String,
                                             row_header : Bytes,
                                             payload : Bytes,
                                             tier : UInt32,
                                             manifest_schema : String = HyNomasCacheQBitCheckpoint::MANIFEST_SCHEMA,
                                             identity_json : String = HyNomasCacheQBitCheckpoint.identity_json_for_setup) : String
  name = "history.layer0.key"
  cache = Bytes.new(HyNomasCacheQBit::HEADER_BYTES + payload.size)
  cache[0, row_header.size].copy_from(row_header)
  cache[HyNomasCacheQBit::HEADER_BYTES, payload.size].copy_from(payload)
  stats = hy_nomas_qbit_checkpoint_stats(name, tier)
  manifest = JSON.build do |json|
    json.object do
      json.field "schema", manifest_schema
      json.field "identity_json", identity_json
      json.field "timeline_bytes", 4_i64
      json.field "stats_json", stats
    end
  end
  records = {
    "timeline"                                => Bytes[1_u8, 2_u8, 3_u8, 4_u8],
    name                                      => cache,
    HyNomasCacheQBitCheckpoint::MANIFEST_NAME => manifest.to_slice,
  }
  sizes = records.transform_values { |bytes| bytes.size.to_i64 }
  ML::Video::HyNomasResumeBundle.write(path,
    HyNomasCacheQBitCheckpoint.contract_for("base"), sizes) do |record, io|
    io.write(records[record])
  end
end

private def hy_nomas_qbit_row_header(mean : Float32, sigma : Float32,
                                     payload_offset : UInt32, tier : UInt32) : Bytes
  bytes = Bytes.new(HyNomasCacheQBit::HEADER_BYTES)
  HyNomasCacheQBitCheckpoint.write_f32(bytes, 0, mean)
  HyNomasCacheQBitCheckpoint.write_f32(bytes, 4, sigma)
  HyNomasCacheQBitCheckpoint.write_u32(bytes, 8, payload_offset)
  HyNomasCacheQBitCheckpoint.write_u32(bytes, 12, tier)
  bytes
end

describe HyNomasCacheQBitCheckpoint::Preflight do
  it "budgets only per-row headers, counters, and the decoder codebook" do
    counts = [128_i32, 256_i32]
    expected = (3_i64 * HyNomasCacheQBit::HEADER_BYTES) +
               (2_i64 * HyNomasCacheQBit::STATS_BYTES) + HyNomasCacheQBit::CODEBOOK_BYTES
    HyNomasCacheQBitCheckpoint::PackPlan.named_bytes(counts).should eq(expected)
    HyNomasCacheQBitCheckpoint::RestorePlan.named_bytes(counts).should eq(
      expected - 2_i64 * HyNomasCacheQBit::STATS_BYTES
    )
    expect_raises(ArgumentError, /divisible by 128/) do
      HyNomasCacheQBitCheckpoint::PackPlan.named_bytes([127_i32])
    end
    expect_raises(ArgumentError, /bounded bundle record size/) do
      HyNomasCacheQBitCheckpoint::PackPlan.named_bytes([1_073_741_824_i32])
    end
  end

  it "authenticates and validates a compact record before restore setup" do
    path = File.tempname("hy-nomas-qbit-preflight", ".bundle")
    header = hy_nomas_qbit_row_header(1.0_f32, 0.0_f32, 0_u32, 4_u32)
    digest = hy_nomas_qbit_checkpoint_fixture(path, header, Bytes.new(64, 0_u8), 4_u32)
    preflight = HyNomasCacheQBitCheckpoint::Preflight.open(path, digest,
      "base", {"timeline" => 4_i64, "history.layer0.key" => 512_i64})
    begin
      preflight.identity["disk_layout"].as_s.should eq(HyNomasCacheQBitCheckpoint::LAYOUT)
      preflight.stats["invalid_rows"].as_i.should eq(0)
      preflight.timeline_bytes.should eq(4_i64)
      preflight.disk_bytes.should eq(File.info(path).size)
      preflight.validate_contract!("base", {"timeline" => 4_i64, "history.layer0.key" => 512_i64})
      expect_raises(ArgumentError, /contract/) do
        preflight.validate_contract!("changed", {"timeline" => 4_i64, "history.layer0.key" => 512_i64})
      end
    ensure
      preflight.close
      File.delete(path) if File.exists?(path)
    end
  end

  it "rejects malformed tiers, noncanonical offsets, nonfinite moments, and nonfinite exact escapes" do
    cases = [
      {"tier", hy_nomas_qbit_row_header(1.0_f32, 0.0_f32, 0_u32, 7_u32), Bytes.new(64, 0_u8), 4_u32, /tier/},
      {"offset", hy_nomas_qbit_row_header(1.0_f32, 0.0_f32, 1_u32, 4_u32), Bytes.new(64, 0_u8), 4_u32, /offset/},
      {"moment", hy_nomas_qbit_row_header(Float32::NAN, 0.0_f32, 0_u32, 4_u32), Bytes.new(64, 0_u8), 4_u32, /moments/},
    ]
    cases.each do |label, header, payload, tier, error|
      path = File.tempname("hy-nomas-qbit-invalid-#{label}", ".bundle")
      digest = hy_nomas_qbit_checkpoint_fixture(path, header, payload, tier)
      begin
        expect_raises(ArgumentError, error) do
          HyNomasCacheQBitCheckpoint::Preflight.open(path, digest,
            "base", {"timeline" => 4_i64, "history.layer0.key" => 512_i64})
        end
      ensure
        File.delete(path) if File.exists?(path)
      end
    end

    path = File.tempname("hy-nomas-qbit-nonfinite-escape", ".bundle")
    header = hy_nomas_qbit_row_header(0.0_f32, 0.0_f32, 0_u32, 32_u32)
    payload = Bytes.new(512, 0_u8)
    HyNomasCacheQBitCheckpoint.write_u32(payload, 0, 0x7fc0_0000_u32)
    digest = hy_nomas_qbit_checkpoint_fixture(path, header, payload, 32_u32)
    begin
      expect_raises(ArgumentError, /nonfinite value/) do
        HyNomasCacheQBitCheckpoint::Preflight.open(path, digest,
          "base", {"timeline" => 4_i64, "history.layer0.key" => 512_i64})
      end
    ensure
      File.delete(path) if File.exists?(path)
    end
  end

  it "rejects oversized logical inventory and a modified bundle digest" do
    expect_raises(ArgumentError, /whole 128-value rows/) do
      HyNomasCacheQBitCheckpoint.maximum_inventory({"timeline" => 4_i64, "history" => 511_i64})
    end

    path = File.tempname("hy-nomas-qbit-modified", ".bundle")
    digest = hy_nomas_qbit_checkpoint_fixture(path,
      hy_nomas_qbit_row_header(1.0_f32, 0.0_f32, 0_u32, 4_u32),
      Bytes.new(64, 0_u8), 4_u32)
    begin
      expect_raises(ArgumentError, /inventory mismatch/) do
        HyNomasCacheQBitCheckpoint::Preflight.open(path, digest,
          "base", {"timeline" => 4_i64, "history.layer0.key" => 512_i64, "history.layer0.value" => 512_i64})
      end
      File.open(path, "r+") do |io|
        io.seek(io.info.size - 1, IO::Seek::Set)
        io.write_byte(0_u8)
      end
      expect_raises(ArgumentError, /full-file SHA-256/) do
        HyNomasCacheQBitCheckpoint::Preflight.open(path, digest,
          "base", {"timeline" => 4_i64, "history.layer0.key" => 512_i64})
      end
    ensure
      File.delete(path) if File.exists?(path)
    end
  end

  it "rejects manifest schema, codebook identity, and timeline inventory mismatches" do
    fixtures = [
      {"schema", HyNomasCacheQBitCheckpoint::MANIFEST_SCHEMA + "-old", HyNomasCacheQBitCheckpoint.identity_json_for_setup, /schema mismatch/},
      {"identity", HyNomasCacheQBitCheckpoint::MANIFEST_SCHEMA, "{}", /codec identity mismatch/},
    ]
    fixtures.each do |label, schema, identity, error|
      path = File.tempname("hy-nomas-qbit-manifest-#{label}", ".bundle")
      digest = hy_nomas_qbit_checkpoint_fixture(path,
        hy_nomas_qbit_row_header(1.0_f32, 0.0_f32, 0_u32, 4_u32),
        Bytes.new(64, 0_u8), 4_u32, schema, identity)
      begin
        expect_raises(ArgumentError, error) do
          HyNomasCacheQBitCheckpoint::Preflight.open(path, digest,
            "base", {"timeline" => 4_i64, "history.layer0.key" => 512_i64})
        end
      ensure
        File.delete(path) if File.exists?(path)
      end
    end

    path = File.tempname("hy-nomas-qbit-timeline-inventory", ".bundle")
    digest = hy_nomas_qbit_checkpoint_fixture(path,
      hy_nomas_qbit_row_header(1.0_f32, 0.0_f32, 0_u32, 4_u32),
      Bytes.new(64, 0_u8), 4_u32)
    begin
      expect_raises(ArgumentError, /timeline byte count/) do
        HyNomasCacheQBitCheckpoint::Preflight.open(path, digest,
          "base", {"timeline" => 8_i64, "history.layer0.key" => 512_i64})
      end
    ensure
      File.delete(path) if File.exists?(path)
    end
  end

  it "rejects repeated and partial restore setup" do
    path = File.tempname("hy-nomas-qbit-partial-restore", ".bundle")
    digest = hy_nomas_qbit_checkpoint_fixture(path,
      hy_nomas_qbit_row_header(1.0_f32, 0.0_f32, 0_u32, 4_u32),
      Bytes.new(64, 0_u8), 4_u32)
    preflight = HyNomasCacheQBitCheckpoint::Preflight.open(path, digest,
      "base", {"timeline" => 4_i64, "history.layer0.key" => 512_i64})
    begin
      preflight.claim_restore!
      expect_raises(ArgumentError, /already been claimed/) do
        preflight.claim_restore!
      end
      timeline = Bytes.new(4)
      preflight.copy_timeline_to!(timeline)
      timeline.should eq(Bytes[1_u8, 2_u8, 3_u8, 4_u8])
      expect_raises(ArgumentError, /partial/) do
        preflight.finish_restore!
      end
      expect_raises(ArgumentError, /not writable/) do
        preflight.copy_timeline_to!(Bytes.new(4))
      end
    ensure
      preflight.close
      File.delete(path) if File.exists?(path)
    end
  end
end
