require "spec"
require "digest/sha256"
require "json"
require "../examples/support/hy_nomas_reference_image"

private class HyNomasReferenceImageFixture
  getter root : String
  getter manifest_path : String
  getter payload_path : String
  getter encoder_sha : String
  getter payload_bytes : Bytes

  def initialize
    @root = File.tempname("hy-nomas-reference-image")
    Dir.mkdir(@root)
    @root = File.realpath(@root)
    @manifest_path = File.join(@root, "reference.manifest.json")
    @payload_path = File.join(@root, "reference.rgb8.bin")
    @encoder_sha = "a" * 64
    @payload_bytes = Bytes.new(ML::Video::HyNomasReferenceImage::PIXEL_BYTES.to_i, 0x5a_u8)
    File.write(@payload_path, @payload_bytes)
    write_manifest
  end

  def manifest : Hash(String, JSON::Any)
    {
      "schema"                     => JSON::Any.new("cogni_ml.hy_nomas_reference_image/v1"),
      "schema_version"             => JSON::Any.new(1_i64),
      "upstream_revision"          => JSON::Any.new("a31c0fe0bcc493be188fd37ddba4ce2f674becb4"),
      "image_path"                 => JSON::Any.new("/prepared/source.png"),
      "image_sha256"               => JSON::Any.new("b" * 64),
      "width"                      => JSON::Any.new(512_i64),
      "height"                     => JSON::Any.new(288_i64),
      "frames"                     => JSON::Any.new(120_i64),
      "fit_policy"                 => JSON::Any.new("exif-rgb-lanczos-pad-black-center"),
      "payload"                    => JSON::Any.new("reference.rgb8.bin"),
      "payload_bytes"              => JSON::Any.new(@payload_bytes.size.to_i64),
      "payload_sha256"             => JSON::Any.new(Digest::SHA256.hexdigest(@payload_bytes)),
      "checkpoint_sha256"          => JSON::Any.new("ebbee483c80890c7795a2babc06e1cb4d42564174cf3bae2b698c0589d95cc3b"),
      "encoder_safetensors_sha256" => JSON::Any.new(@encoder_sha),
    }
  end

  def write_manifest(document = manifest) : Nil
    File.write(@manifest_path, document.to_json)
  end

  def inputs : ML::Video::HyNomasReferenceImage::Inputs
    ML::Video::HyNomasReferenceImage::Inputs.new(@manifest_path,
      Digest::SHA256.hexdigest(File.read(@manifest_path)), @encoder_sha)
  end

  def cleanup : Nil
    File.delete(@manifest_path) if File.exists?(@manifest_path)
    File.delete(@payload_path) if File.exists?(@payload_path)
    Dir.delete(@root) if Dir.exists?(@root)
  end
end

describe ML::Video::HyNomasReferenceImage::Inputs do
  it "admits only the pinned one-frame RGB payload and reports its encoder identity" do
    fixture = HyNomasReferenceImageFixture.new
    begin
      inputs = fixture.inputs
      inputs.pixel_bytes.should eq(512_i64 * 288 * 3)
      inputs.encoder_sha.should eq(fixture.encoder_sha)
      inputs.image_sha256.should eq("b" * 64)
      inputs.payload_sha256.should eq(Digest::SHA256.hexdigest(fixture.payload_bytes))
      inputs.verify_payload!
    ensure
      fixture.cleanup
    end
  end

  it "rejects a false trusted manifest hash" do
    fixture = HyNomasReferenceImageFixture.new
    begin
      expect_raises(ArgumentError) do
        ML::Video::HyNomasReferenceImage::Inputs.new(fixture.manifest_path,
          "c" * 64, fixture.encoder_sha)
      end
    ensure
      fixture.cleanup
    end
  end

  it "rejects manifest mutation or symlink substitution after admission" do
    fixture = HyNomasReferenceImageFixture.new
    link = File.join(fixture.root, "manifest-link.json")
    begin
      inputs = fixture.inputs
      changed = fixture.manifest
      changed["image_sha256"] = JSON::Any.new("e" * 64)
      fixture.write_manifest(changed)
      expect_raises(ArgumentError, /manifest changed/) { inputs.verify_payload! }

      File.symlink(fixture.manifest_path, link)
      expect_raises(ArgumentError) do
        ML::Video::HyNomasReferenceImage::Inputs.new(link,
          Digest::SHA256.hexdigest(File.read(fixture.manifest_path)), fixture.encoder_sha)
      end
    ensure
      File.delete(link) if File.symlink?(link)
      fixture.cleanup
    end
  end

  it "rejects a CLI manifest symlink before path canonicalization" do
    fixture = HyNomasReferenceImageFixture.new
    link = File.join(fixture.root, "cli-manifest-link.json")
    begin
      File.symlink(fixture.manifest_path, link)
      expect_raises(ArgumentError, /must not be a symlink/) do
        ML::Video::HyNomasReferenceImage.canonical_manifest_argument!(link)
      end
      ML::Video::HyNomasReferenceImage.canonical_manifest_argument!(fixture.manifest_path)
        .should eq(fixture.manifest_path)
      ML::Video::HyNomasReferenceImage.canonical_manifest_argument!(File.join(fixture.root, ".", "reference.manifest.json"))
        .should eq(fixture.manifest_path)
    ensure
      File.delete(link) if File.symlink?(link)
      fixture.cleanup
    end
  end

  it "rejects changed schema, geometry, encoder, checkpoint, and extra keys" do
    fixture = HyNomasReferenceImageFixture.new
    begin
      {
        ->(doc : Hash(String, JSON::Any)) { doc["schema"] = JSON::Any.new("wrong/v1") },
        ->(doc : Hash(String, JSON::Any)) { doc["width"] = JSON::Any.new(256_i64) },
        ->(doc : Hash(String, JSON::Any)) { doc["encoder_safetensors_sha256"] = JSON::Any.new("c" * 64) },
        ->(doc : Hash(String, JSON::Any)) { doc["checkpoint_sha256"] = JSON::Any.new("d" * 64) },
        ->(doc : Hash(String, JSON::Any)) { doc["untrusted"] = JSON::Any.new(true) },
      }.each do |mutate|
        document = fixture.manifest
        mutate.call(document)
        fixture.write_manifest(document)
        expect_raises(ArgumentError) { fixture.inputs }
      end
    ensure
      fixture.cleanup
    end
  end

  it "rejects payload traversal and payload tampering after admission" do
    fixture = HyNomasReferenceImageFixture.new
    begin
      document = fixture.manifest
      document["payload"] = JSON::Any.new("../outside.rgb8.bin")
      fixture.write_manifest(document)
      expect_raises(ArgumentError) { fixture.inputs }

      document = fixture.manifest
      fixture.write_manifest(document)
      inputs = fixture.inputs
      File.open(fixture.payload_path, "r+") do |io|
        io.seek(0)
        io.write(Bytes[0x78_u8])
      end
      expect_raises(ArgumentError) { inputs.verify_payload! }

      File.delete(fixture.payload_path)
      File.symlink(File.join(fixture.root, "outside.rgb8.bin"), fixture.payload_path)
      expect_raises(ArgumentError) do
        ML::Video::HyNomasReferenceImage::Inputs.new(fixture.manifest_path,
          Digest::SHA256.hexdigest(File.read(fixture.manifest_path)), fixture.encoder_sha)
      end
    ensure
      File.delete(fixture.payload_path) if File.symlink?(fixture.payload_path)
      fixture.cleanup
    end
  end

  it "rejects malformed manifest JSON even with a matching trusted digest" do
    fixture = HyNomasReferenceImageFixture.new
    begin
      malformed = "{\"schema\":"
      File.write(fixture.manifest_path, malformed)
      expect_raises(ArgumentError, /malformed/) do
        ML::Video::HyNomasReferenceImage::Inputs.new(fixture.manifest_path,
          Digest::SHA256.hexdigest(malformed), fixture.encoder_sha)
      end
    ensure
      fixture.cleanup
    end
  end
end
