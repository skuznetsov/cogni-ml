require "./spec_helper"
require "../src/ml/gguf/gemma4_state_snapshot"
require "../src/ml/gguf/qwen35_metal"

private def gemma4_snapshot_fill(state : ML::GGUF::Gemma4Metal::ResidentState) : Nil
  state.layers.each_with_index do |layer, il|
    count = layer.max_seq * layer.kv_dim
    k = Array(Float32).new(count) { |i| (1000 * (il + 1) + i).to_f32 }
    v = Array(Float32).new(count) { |i| (-1000 * (il + 1) - i).to_f32 }
    layer.k_cache_buf.write(k)
    layer.v_cache_buf.write(v)
  end
end

private def gemma4_snapshot_expect_prefix(restored : ML::GGUF::Gemma4Metal::ResidentState,
                                          source : ML::GGUF::Gemma4Metal::ResidentState,
                                          prefix_len : Int32) : Nil
  restored.layers.each_with_index do |layer, il|
    src = source.layers[il]
    live = prefix_len * layer.kv_dim
    layer.k_cache_buf.read(layer.max_seq * layer.kv_dim)[0, live].should eq(src.k_cache_buf.read(src.max_seq * src.kv_dim)[0, live])
    layer.v_cache_buf.read(layer.max_seq * layer.kv_dim)[0, live].should eq(src.v_cache_buf.read(src.max_seq * src.kv_dim)[0, live])
  end
end

describe "Gemma4StateSnapshot" do
  pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

  it "captures and restores exact resident K/V prefix rows" do
    kv_dims = [3, 5]
    source = ML::GGUF::Gemma4Metal::ResidentState.new(kv_dims, 6)
    target = ML::GGUF::Gemma4Metal::ResidentState.new(kv_dims, 6)
    gemma4_snapshot_fill(source)

    snapshot = ML::GGUF::Gemma4StateSnapshot.capture(source, prefix_len: 4)
    snapshot.max_seq.should eq(6)
    snapshot.prefix_len.should eq(4)
    snapshot.layer_count.should eq(2)
    snapshot.records.size.should eq(4)
    snapshot.byte_size.should eq((4 * (3 + 3 + 5 + 5) * sizeof(Float32)).to_i64)

    ML::GGUF::Gemma4StateSnapshot.restore_into(snapshot, target)

    gemma4_snapshot_expect_prefix(target, source, 4)
  end

  it "round-trips a raw Gemma K/V artifact with checksum validation" do
    kv_dims = [2, 4]
    source = ML::GGUF::Gemma4Metal::ResidentState.new(kv_dims, 5)
    gemma4_snapshot_fill(source)
    snapshot = ML::GGUF::Gemma4StateSnapshot.capture(source, prefix_len: 3)
    path = File.tempname("gemma4-state", ".gkv")

    begin
      info = ML::GGUF::Gemma4StateSnapshot.write_artifact(snapshot, path)
      loaded = ML::GGUF::Gemma4StateSnapshot.read_artifact(path, expected_sha256: info.sha256)
      restored = ML::GGUF::Gemma4StateSnapshot.restore(loaded, kv_dims)

      loaded.max_seq.should eq(snapshot.max_seq)
      loaded.prefix_len.should eq(snapshot.prefix_len)
      loaded.byte_size.should eq(snapshot.byte_size)
      gemma4_snapshot_expect_prefix(restored, source, 3)

      expect_raises(ArgumentError, /checksum mismatch/) do
        ML::GGUF::Gemma4StateSnapshot.read_artifact(path, expected_sha256: "0" * 64)
      end
    ensure
      File.delete(path) if File.exists?(path)
    end
  end

  it "streams the same artifact bytes as the in-memory encoder" do
    kv_dims = [2]
    source = ML::GGUF::Gemma4Metal::ResidentState.new(kv_dims, 4)
    gemma4_snapshot_fill(source)
    snapshot = ML::GGUF::Gemma4StateSnapshot.capture(source, prefix_len: 2)
    expected = ML::GGUF::Gemma4StateSnapshot.encode_artifact_bytes(snapshot)
    path = File.tempname("gemma4-state-stream", ".gkv")

    begin
      info = ML::GGUF::Gemma4StateSnapshot.write_artifact(snapshot, path)
      File.read(path).to_slice.should eq(expected)
      info.byte_size.should eq(expected.size)
      info.sha256.should eq(Digest::SHA256.hexdigest(expected))
    ensure
      File.delete(path) if File.exists?(path)
    end
  end

  it "rejects restore into an incompatible resident state" do
    source = ML::GGUF::Gemma4Metal::ResidentState.new([3, 5], 6)
    target = ML::GGUF::Gemma4Metal::ResidentState.new([3, 4], 6)
    gemma4_snapshot_fill(source)

    snapshot = ML::GGUF::Gemma4StateSnapshot.capture(source, prefix_len: 4)

    expect_raises(ArgumentError, /kv_dim mismatch/) do
      ML::GGUF::Gemma4StateSnapshot.restore_into(snapshot, target)
    end
  end
end
