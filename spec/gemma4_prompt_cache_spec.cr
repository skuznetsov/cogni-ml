require "./spec_helper"
require "file_utils"
require "../src/ml/gguf/gemma4_prompt_cache"
require "../src/ml/gguf/qwen35_metal"

private def gemma4_prompt_cache_fill(state : ML::GGUF::Gemma4Metal::ResidentState) : Nil
  state.layers.each_with_index do |layer, il|
    count = layer.max_seq * layer.kv_dim
    layer.k_cache_buf.write(Array(Float32).new(count) { |i| (10_000 * (il + 1) + i).to_f32 })
    layer.v_cache_buf.write(Array(Float32).new(count) { |i| (-10_000 * (il + 1) - i).to_f32 })
  end
end

private def gemma4_prompt_cache_prefix_equal?(a : ML::GGUF::Gemma4Metal::ResidentState,
                                              b : ML::GGUF::Gemma4Metal::ResidentState,
                                              prefix_len : Int32) : Bool
  a.layers.each_with_index.all? do |layer, il|
    other = b.layers[il]
    live = prefix_len * layer.kv_dim
    layer.k_cache_buf.read(layer.max_seq * layer.kv_dim)[0, live] == other.k_cache_buf.read(other.max_seq * other.kv_dim)[0, live] &&
      layer.v_cache_buf.read(layer.max_seq * layer.kv_dim)[0, live] == other.v_cache_buf.read(other.max_seq * other.kv_dim)[0, live]
  end
end

describe "Gemma4PromptCache" do
  pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

  it "saves, looks up, and restores an exact resident prompt prefix" do
    dir = File.tempname("gemma4-prompt-cache")
    FileUtils.mkdir_p(dir)
    begin
      store = ML::GGUF::Gemma4PromptCache::Store.new(dir)
      source = ML::GGUF::Gemma4Metal::ResidentState.new([2, 3], 8)
      gemma4_prompt_cache_fill(source)
      token_ids = [2_i32, 10_i32, 20_i32, 30_i32]

      saved = store.save_resident_state(
        source,
        token_ids,
        model_id: "gemma4-test",
        tokenizer_id: "tok-test",
        prompt_text: "hello",
        session_id: "s1",
        turn_id: "t1",
      )
      saved.runtime_id.should eq(ML::GGUF::Gemma4PromptCache::RUNTIME_ID)
      saved.prefix_len.should eq(token_ids.size)
      saved.token_hash.should eq(ML::GGUF::Gemma4PromptCache.token_hash(token_ids))

      hit = store.lookup_prompt("gemma4-test", "tok-test", "hello", token_ids).not_nil!
      hit.artifact_sha256.should eq(saved.artifact_sha256)
      restored = store.restore(hit)

      gemma4_prompt_cache_prefix_equal?(restored, source, token_ids.size).should be_true
    ensure
      FileUtils.rm_rf(dir) if File.exists?(dir)
    end
  end

  it "finds longest compatible token prefix and ignores model/tokenizer mismatches" do
    dir = File.tempname("gemma4-prompt-cache")
    FileUtils.mkdir_p(dir)
    begin
      store = ML::GGUF::Gemma4PromptCache::Store.new(dir)
      source = ML::GGUF::Gemma4Metal::ResidentState.new([2], 8)
      gemma4_prompt_cache_fill(source)
      prefix = [2_i32, 10_i32, 20_i32]

      store.save_resident_state(source, prefix, model_id: "m", tokenizer_id: "t", prompt_text: "p")

      store.lookup_longest_prefix("m", "t", prefix + [30_i32, 40_i32]).not_nil!.prefix_len.should eq(prefix.size)
      store.lookup_longest_prefix("other", "t", prefix + [30_i32]).should be_nil
      store.lookup_longest_prefix("m", "other", prefix + [30_i32]).should be_nil
    ensure
      FileUtils.rm_rf(dir) if File.exists?(dir)
    end
  end

  it "fails closed for corrupt manifest lines and mutated artifacts" do
    dir = File.tempname("gemma4-prompt-cache")
    FileUtils.mkdir_p(dir)
    begin
      store = ML::GGUF::Gemma4PromptCache::Store.new(dir)
      source = ML::GGUF::Gemma4Metal::ResidentState.new([2], 8)
      gemma4_prompt_cache_fill(source)
      token_ids = [2_i32, 10_i32]

      saved = store.save_resident_state(source, token_ids, model_id: "m", tokenizer_id: "t")
      File.open(File.join(dir, "manifest.jsonl"), "a") { |file| file.puts("{not-json") }
      store.entries.size.should eq(1)

      File.open(saved.artifact_path, "a") { |file| file.write_byte(0_u8) }
      hit = store.lookup_prompt("m", "t", "", token_ids).not_nil!
      expect_raises(ArgumentError, /byte-size mismatch|checksum mismatch|changed/) do
        store.restore(hit)
      end
    ensure
      FileUtils.rm_rf(dir) if File.exists?(dir)
    end
  end

  it "restores into a reusable resident state without replacing the object" do
    dir = File.tempname("gemma4-prompt-cache")
    FileUtils.mkdir_p(dir)
    begin
      store = ML::GGUF::Gemma4PromptCache::Store.new(dir)
      source = ML::GGUF::Gemma4Metal::ResidentState.new([2, 3], 8)
      target = ML::GGUF::Gemma4Metal::ResidentState.new([2, 3], 8)
      gemma4_prompt_cache_fill(source)
      token_ids = [2_i32, 10_i32, 20_i32]

      saved = store.save_resident_state(source, token_ids, model_id: "m", tokenizer_id: "t")
      restored = store.restore(saved, reuse_state: target)

      restored.object_id.should eq(target.object_id)
      gemma4_prompt_cache_prefix_equal?(target, source, token_ids.size).should be_true
    ensure
      FileUtils.rm_rf(dir) if File.exists?(dir)
    end
  end

  it "reuses validated snapshots in-process and invalidates them when the artifact changes" do
    dir = File.tempname("gemma4-prompt-cache")
    FileUtils.mkdir_p(dir)
    begin
      store = ML::GGUF::Gemma4PromptCache::Store.new(
        dir,
        snapshot_cache_byte_limit: 1_000_000_i64,
        snapshot_cache_entry_limit: 1,
      )
      source = ML::GGUF::Gemma4Metal::ResidentState.new([2, 3], 8)
      gemma4_prompt_cache_fill(source)
      token_ids = [2_i32, 10_i32, 20_i32]

      saved = store.save_resident_state(source, token_ids, model_id: "m", tokenizer_id: "t")
      target1 = ML::GGUF::Gemma4Metal::ResidentState.new([2, 3], 8)
      store.restore(saved, reuse_state: target1)
      store.snapshot_cache_misses.should eq(1)
      store.snapshot_cache_hits.should eq(0)
      store.snapshot_cache_bytes.should eq(saved.state_byte_size)

      target2 = ML::GGUF::Gemma4Metal::ResidentState.new([2, 3], 8)
      store.restore(saved, reuse_state: target2)
      store.snapshot_cache_hits.should eq(1)
      gemma4_prompt_cache_prefix_equal?(target2, source, token_ids.size).should be_true

      sleep 20.milliseconds
      File.open(saved.artifact_path, "r+") do |file|
        file.seek(saved.artifact_byte_size - 1)
        byte = file.read_byte.not_nil!
        file.seek(saved.artifact_byte_size - 1)
        file.write_byte(byte ^ 0xff_u8)
      end

      target3 = ML::GGUF::Gemma4Metal::ResidentState.new([2, 3], 8)
      expect_raises(ArgumentError, /checksum mismatch/) do
        store.restore(saved, reuse_state: target3)
      end
    ensure
      FileUtils.rm_rf(dir) if File.exists?(dir)
    end
  end
end
