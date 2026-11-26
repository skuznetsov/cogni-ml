require "spec"
require "../examples/support/hy_nomas_reference_cache_probe"

describe HyNomasReferenceCacheProbe do
  it "uses the authenticated valid text prefix length without a fixed token count" do
    {1, 23, 1280}.each do |tokens|
      dtype, shape = HyNomasReferenceCacheProbe.tensor_spec("block.txt_valid_f32", tokens)
      dtype.should eq("F32")
      shape.should eq([1, tokens, 2048])
    end

    expect_raises(ArgumentError) do
      HyNomasReferenceCacheProbe.tensor_spec("block.txt_valid_f32", 0)
    end
    expect_raises(ArgumentError) do
      HyNomasReferenceCacheProbe.tensor_spec("block.txt_valid_f32", 1281)
    end
  end

  it "preserves a file created after the freshness check wins a race" do
    root = File.tempname("hy-nomas-cache-race")
    Dir.mkdir(root)
    path = File.join(root, "fixture.bin")
    begin
      HyNomasReferenceCacheProbe.ensure_fresh_outputs!(path, "#{path}.manifest")
      # Model another process winning between preflight and O_EXCL creation.
      File.write(path, "other owner's bytes")

      expect_raises(IO::Error) do
        HyNomasReferenceCacheProbe.open_fresh_output!(path)
      end
      File.read(path).should eq("other owner's bytes")
    ensure
      File.delete(path) if File.exists?(path)
      Dir.delete(root) if Dir.exists?(root)
    end
  end

  it "refuses a pre-existing sidecar or manifest without replacing either" do
    root = File.tempname("hy-nomas-cache-existing")
    Dir.mkdir(root)
    manifest = File.join(root, "fixture.json")
    payload = File.join(root, "fixture.tensors.bin")
    begin
      File.write(manifest, "manifest owner")
      File.write(payload, "payload owner")
      expect_raises(ArgumentError) do
        HyNomasReferenceCacheProbe.ensure_fresh_outputs!(manifest, payload)
      end
      File.read(manifest).should eq("manifest owner")
      File.read(payload).should eq("payload owner")
    ensure
      File.delete(manifest) if File.exists?(manifest)
      File.delete(payload) if File.exists?(payload)
      Dir.delete(root) if Dir.exists?(root)
    end
  end
end
