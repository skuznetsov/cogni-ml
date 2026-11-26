require "spec"
require "../examples/support/hy_nomas_cinematic"

private def cinematic_document : JSON::Any
  JSON.parse({schema: "cogni_ml.hy_nomas_cinematic_bf16_cache/v1",
              upstream_revision: "911cea6458a89c2663513ff7099561ec190dd491",
              base_index_sha256: "a235ca592b47161472c7fd94cd3d886fce164480cd0571179c8860002063cbb0",
              base_source_sha256: "73819bcf3b82b090116dd1a1748795de95eef703c304e7443963d00ea8aa4d70",
              adapter: {repo: "tencent/Hy-Nomas", revision: "bb6a8ca451bfe84947f0ea1c6d51957f983ffffe",
                        path: "DiT/lora_weights/cinematic/adapter.safetensors",
                        sha256: "765426a6c417eed70a57d72f67cfe002076cfc92b6b35fe185091012ba80c7c2",
                        bytes: 6_122_591_632_i64, rank: 256, alpha: 512.0,
                        config_sha256: "f656c2866953e7a90a785a5c51ce127a79ea61c4217ef30e3cd2d69afcd3d998"},
              merge: {formula: "bf16(base_f32 + (B_f32 @ A_f32) * 2)", active_targets: 756,
                      skipped_action_targets: 162, action_conditioning: false},
              index: {basename: "index.json", sha256: "a" * 64}, producer: {test: true}}.to_json)
end

describe HyNomasCinematic do
  it "admits only the pinned active-block merge and keeps its actual index identity" do
    HyNomasCinematic.validate_document!(cinematic_document).should eq("a" * 64)
  end

  it "rejects a changed base, rank, config, scale, target count or action scope" do
    [
      {"base_source_sha256", nil, JSON::Any.new("a" * 64)},
      {"base_index_sha256", nil, JSON::Any.new("b" * 64)},
      {"adapter", "rank", JSON::Any.new(128_i64)},
      {"adapter", "config_sha256", JSON::Any.new("a" * 64)},
      {"adapter", "alpha", JSON::Any.new(256.0)},
      {"merge", "active_targets", JSON::Any.new(918_i64)},
      {"merge", "skipped_action_targets", JSON::Any.new(0_i64)},
      {"merge", "action_conditioning", JSON::Any.new(true)},
      {"merge", "formula", JSON::Any.new("bf16(base_bf16 + delta)")},
      {"index", "basename", JSON::Any.new("../index.json")},
    ].each do |outer, inner, value|
      document = cinematic_document
      if inner
        document[outer].as_h[inner] = value
      else
        document.as_h[outer] = value
      end
      expect_raises(ArgumentError) { HyNomasCinematic.validate_document!(document) }
    end
  end

  it "rejects incomplete selection and Q4 before touching files or Metal" do
    HyNomasCinematic.load!("", "", bf16: true, camera_conditioning: false, action_conditioning: false).should be_nil
    expect_raises(ArgumentError, /both/) { HyNomasCinematic.load!("/missing", "", bf16: true, camera_conditioning: false, action_conditioning: false) }
    expect_raises(ArgumentError, /BF16/) { HyNomasCinematic.load!("/missing", "a" * 64, bf16: false, camera_conditioning: false, action_conditioning: false) }
    expect_raises(ArgumentError, /SHA/) { HyNomasCinematic.load!("/missing", "bad", bf16: true, camera_conditioning: false, action_conditioning: false) }
    expect_raises(ArgumentError, /conditioning/) do
      HyNomasCinematic.load!("/missing", "a" * 64, bf16: true, camera_conditioning: true, action_conditioning: false)
    end
    expect_raises(ArgumentError, /conditioning/) do
      HyNomasCinematic.load!("/missing", "a" * 64, bf16: true, camera_conditioning: false, action_conditioning: true)
    end
  end

  it "binds the selected sibling index and refuses a changed lineage file" do
    directory = File.tempname("hy-nomas-cinematic")
    Dir.mkdir(directory)
    path = File.join(directory, "cinematic-lineage.json")
    index = File.join(directory, "index.json")
    begin
      contents = cinematic_document.to_json
      File.write(path, contents)
      File.write(index, "{}")
      sha = Digest::SHA256.hexdigest(contents)
      identity = HyNomasCinematic.load!(path, sha, bf16: true,
        camera_conditioning: false, action_conditioning: false).not_nil!
      identity.path.should eq(File.realpath(index))
      identity.sha256.should eq("a" * 64)
      identity.lineage_sha256.should eq(sha)
      File.write(path, contents + "\n")
      expect_raises(ArgumentError, /SHA or file changed/) do
        HyNomasCinematic.load!(path, sha, bf16: true, camera_conditioning: false, action_conditioning: false)
      end
    ensure
      File.delete?(path)
      File.delete?(index)
      Dir.delete(directory)
    end
  end
end
