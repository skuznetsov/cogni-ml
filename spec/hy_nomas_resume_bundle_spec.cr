require "spec"
require "digest/sha256"
require "file/tempfile"
require "../src/ml/video/hy_nomas_resume_bundle"

private def hy_nomas_resume_sha256(path : String) : String
  digest = Digest::SHA256.new
  File.open(path, "rb") do |io|
    chunk = Bytes.new(64 * 1024)
    while (count = io.read(chunk)) > 0
      digest.update(chunk[0, count])
    end
  end
  digest.hexfinal
end

private def hy_nomas_resume_write(path : String, contract : String,
                                  records : Hash(String, Bytes)) : String
  sizes = records.transform_values { |bytes| bytes.size.to_i64 }
  ML::Video::HyNomasResumeBundle.write(path, contract, sizes) do |name, io|
    io.write(records[name])
  end
end

describe ML::Video::HyNomasResumeBundle do
  it "streams an exact inventory and restores each record byte-for-byte" do
    path = File.tempname("hy-nomas-resume-roundtrip", ".bundle")
    records = {
      "latent"  => Bytes[0_u8, 1_u8, 2_u8, 255_u8],
      "history" => Bytes[9_u8, 8_u8, 7_u8],
    }
    digest = hy_nomas_resume_write(path, "run:opaque:identity", records)
    digest.should eq(hy_nomas_resume_sha256(path))

    reader = ML::Video::HyNomasResumeBundle.open(path, digest,
      "run:opaque:identity", {"latent" => 4_i64, "history" => 3_i64})
    begin
      reader.validate_contract!("run:opaque:identity",
        {"latent" => 4_i64, "history" => 3_i64})
      records.each do |name, expected|
        destination = Bytes.new(expected.size, 0_u8)
        reader.copy_to!(name, destination)
        destination.should eq(expected)
      end
      expect_raises(ArgumentError, /unknown record/) do
        reader.copy_to!("missing", Bytes.empty)
      end
    ensure
      reader.close
      File.delete(path) if File.exists?(path)
    end
  end

  it "rejects a changed full-file digest, contract, or inventory before copying" do
    path = File.tempname("hy-nomas-resume-identity", ".bundle")
    digest = hy_nomas_resume_write(path, "contract-A", {"state" => Bytes[1_u8, 2_u8]})
    begin
      expect_raises(ArgumentError, /SHA-256/) do
        ML::Video::HyNomasResumeBundle.open(path, "0" * 64,
          "contract-A", {"state" => 2_i64})
      end
      expect_raises(ArgumentError, /contract/) do
        ML::Video::HyNomasResumeBundle.open(path, digest,
          "contract-B", {"state" => 2_i64})
      end
      expect_raises(ArgumentError, /inventory/) do
        ML::Video::HyNomasResumeBundle.open(path, digest,
          "contract-A", {"other" => 2_i64})
      end
      reader = ML::Video::HyNomasResumeBundle.open(path, digest,
        "contract-A", {"state" => 2_i64})
      begin
        expect_raises(ArgumentError, /contract or inventory/) do
          reader.validate_contract!("contract-B", {"state" => 2_i64})
        end
      ensure
        reader.close
      end
    ensure
      File.delete(path) if File.exists?(path)
    end
  end

  it "checks the record digest during direct destination copies" do
    path = File.tempname("hy-nomas-resume-record-digest", ".bundle")
    digest = hy_nomas_resume_write(path, "contract", {"state" => Bytes[1_u8, 2_u8, 3_u8]})
    begin
      File.open(path, "r+") do |io|
        io.seek(io.info.size - 1, IO::Seek::Set)
        io.write_byte(4_u8)
      end
      tampered_digest = hy_nomas_resume_sha256(path)
      tampered_digest.should_not eq(digest)

      reader = ML::Video::HyNomasResumeBundle.open(path, tampered_digest,
        "contract", {"state" => 3_i64})
      begin
        expect_raises(ArgumentError, /record SHA-256/) do
          reader.copy_to!("state", Bytes.new(3, 0_u8))
        end
      ensure
        reader.close
      end
    ensure
      File.delete(path) if File.exists?(path)
    end
  end

  it "rejects short or long callback output without publishing a partial bundle" do
    [{"state" => 2_i64}, {"state" => 4_i64}].each_with_index do |descriptors, index|
      path = File.tempname("hy-nomas-resume-length-#{index}", ".bundle")
      begin
        expect_raises(ArgumentError, /record byte count/) do
          ML::Video::HyNomasResumeBundle.write(path, "contract", descriptors) do |_name, io|
            io.write(Bytes[1_u8, 2_u8, 3_u8])
          end
        end
        File.exists?(path).should be_false
      ensure
        File.delete(path) if File.exists?(path)
      end
    end
  end

  it "never clobbers an existing path and rejects symlinks and nonregular files" do
    target = File.tempname("hy-nomas-resume-existing", ".bundle")
    File.write(target, "preserve")
    begin
      expect_raises(ArgumentError, /already exists/) do
        ML::Video::HyNomasResumeBundle.write(target, "contract", {"state" => 1_i64}) do |_name, io|
          io.write(Bytes[7_u8])
        end
      end
      File.read(target).should eq("preserve")
    ensure
      File.delete(target) if File.exists?(target)
    end

    link_target = File.tempname("hy-nomas-resume-link-target", ".bundle")
    File.write(link_target, "not a bundle")
    link = File.tempname("hy-nomas-resume-link", ".bundle")
    File.symlink(link_target, link)
    begin
      expect_raises(ArgumentError, /regular file/) do
        ML::Video::HyNomasResumeBundle.open(link, "0" * 64,
          "contract", {"state" => 1_i64})
      end
    ensure
      File.delete(link) if File.exists?(link) || File.info?(link, false)
      File.delete(link_target) if File.exists?(link_target)
    end

    directory = File.tempname("hy-nomas-resume-directory")
    Dir.mkdir(directory)
    begin
      expect_raises(ArgumentError, /regular file/) do
        ML::Video::HyNomasResumeBundle.open(directory, "0" * 64,
          "contract", {"state" => 1_i64})
      end
    ensure
      Dir.delete(directory) if Dir.exists?(directory)
    end
  end

  it "rejects an oversized header before parsing it" do
    path = File.tempname("hy-nomas-resume-oversized-header", ".bundle")
    File.open(path, "wb") do |io|
      io.write("HNRSB01\0".to_slice)
      io.write_bytes((1024 * 1024 + 1).to_u32, IO::ByteFormat::LittleEndian)
      io.write(Bytes.new(1024 * 1024 + 1, 0_u8))
    end
    digest = hy_nomas_resume_sha256(path)
    begin
      expect_raises(ArgumentError, /header size/) do
        ML::Video::HyNomasResumeBundle.open(path, digest,
          "contract", {"state" => 1_i64})
      end
    ensure
      File.delete(path) if File.exists?(path)
    end
  end

  it "rejects truncated files and files changed after reader admission" do
    truncated = File.tempname("hy-nomas-resume-truncated", ".bundle")
    hy_nomas_resume_write(truncated, "contract", {"state" => Bytes[1_u8, 2_u8, 3_u8]})
    begin
      File.open(truncated, "r+") do |io|
        io.truncate(io.info.size - 1)
      end
      truncated_sha = hy_nomas_resume_sha256(truncated)
      expect_raises(ArgumentError, /file length/) do
        ML::Video::HyNomasResumeBundle.open(truncated, truncated_sha,
          "contract", {"state" => 3_i64})
      end
    ensure
      File.delete(truncated) if File.exists?(truncated)
    end

    changed = File.tempname("hy-nomas-resume-changed", ".bundle")
    digest = hy_nomas_resume_write(changed, "contract", {"state" => Bytes[4_u8, 5_u8]})
    reader = ML::Video::HyNomasResumeBundle.open(changed, digest,
      "contract", {"state" => 2_i64})
    begin
      File.open(changed, "ab") { |io| io.write_byte(6_u8) }
      expect_raises(ArgumentError, /changed after admission/) do
        reader.copy_to!("state", Bytes.new(2, 0_u8))
      end
    ensure
      reader.close
      File.delete(changed) if File.exists?(changed)
    end
  end

  it "bounds generated header memory and does not publish oversized headers" do
    path = File.tempname("hy-nomas-resume-writer-header-limit", ".bundle")
    begin
      expect_raises(ArgumentError, /header size/) do
        ML::Video::HyNomasResumeBundle.write(path, "c" * (1024 * 1024 + 1),
          {"state" => 0_i64}) do |_name, _io|
        end
      end
      File.exists?(path).should be_false
    ensure
      File.delete(path) if File.exists?(path)
    end
  end
end
