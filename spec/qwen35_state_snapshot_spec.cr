require "./spec_helper"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_state_snapshot"
require "../src/ml/gguf/qwen35_weights"

QWEN_9B_SNAPSHOT = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"

describe ML::GGUF::Qwen35StateSnapshot do
  it "keeps raw v1 artifacts as the default durable format" do
    snapshot = synthetic_snapshot([1.0_f32, -2.5_f32, 0.0_f32, 128.0_f32])
    path = File.tempname("qwen35-state-raw", ".qkv")
    begin
      info = ML::GGUF::Qwen35StateSnapshot.write_artifact(snapshot, path)
      File.read(path)[4, 4].bytes.should eq([1_u8, 0_u8, 0_u8, 0_u8])

      loaded = ML::GGUF::Qwen35StateSnapshot.read_artifact(path, expected_sha256: info.sha256)
      floats_from(loaded.records[0].bytes).should eq(floats_from(snapshot.records[0].bytes))
    ensure
      File.delete(path) if File.exists?(path)
    end
  end

  it "writes and reads BF16 v2 artifacts with explicit codec checking" do
    values = [1.0_f32, -2.5_f32, 0.125_f32, 128.0_f32]
    snapshot = synthetic_snapshot(values)
    path = File.tempname("qwen35-state-bf16", ".qkv")
    begin
      info = ML::GGUF::Qwen35StateSnapshot.write_artifact(snapshot, path, artifact_codec: "recurrent-bf16")
      File.read(path)[4, 4].bytes.should eq([2_u8, 0_u8, 0_u8, 0_u8])

      expect_raises(ArgumentError, /requires explicit codec metadata/) do
        ML::GGUF::Qwen35StateSnapshot.read_artifact(path, expected_sha256: info.sha256)
      end

      loaded = ML::GGUF::Qwen35StateSnapshot.read_artifact(path, expected_sha256: info.sha256, expected_codec: "recurrent-bf16")
      floats_from(loaded.records[0].bytes).each_with_index do |actual, i|
        actual.should be_close(values[i], 0.01_f32)
      end

      expect_raises(ArgumentError, /codec mismatch/) do
        ML::GGUF::Qwen35StateSnapshot.read_artifact(path, expected_sha256: info.sha256, expected_codec: "recurrent-int8")
      end
    ensure
      File.delete(path) if File.exists?(path)
    end
  end

  it "writes and reads block INT8 v2 artifacts with block-size checking" do
    values = [1.0_f32, -2.5_f32, 0.125_f32, 128.0_f32, -64.0_f32]
    snapshot = synthetic_snapshot(values)
    path = File.tempname("qwen35-state-i8", ".qkv")
    begin
      info = ML::GGUF::Qwen35StateSnapshot.write_artifact(
        snapshot,
        path,
        artifact_codec: "recurrent-int8",
        artifact_codec_block: 2,
      )

      loaded = ML::GGUF::Qwen35StateSnapshot.read_artifact(
        path,
        expected_sha256: info.sha256,
        expected_codec: "recurrent-int8",
        expected_codec_block: 2,
      )
      floats_from(loaded.records[0].bytes).each_with_index do |actual, i|
        actual.should be_close(values[i], 0.6_f32)
      end

      expect_raises(ArgumentError, /block mismatch/) do
        ML::GGUF::Qwen35StateSnapshot.read_artifact(
          path,
          expected_sha256: info.sha256,
          expected_codec: "recurrent-int8",
          expected_codec_block: 8,
        )
      end
    ensure
      File.delete(path) if File.exists?(path)
    end
  end

  pending!("9B model not present") unless File.exists?(QWEN_9B_SNAPSHOT)

  it "round-trips a Metal-backed prompt state and preserves next-token top1" do
    w = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_SNAPSHOT)
    hp = w.hparams
    prompt = [760_i32, 6511_i32, 314_i32, 9338_i32, 369_i32] # "The capital of France is"

    live = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
    prompt.each_with_index do |token_id, pos|
      ML::GGUF::Qwen35CPU.forward_top1(w, token_id, pos.to_i32, live)
    end

    snapshot = ML::GGUF::Qwen35StateSnapshot.capture(live)
    snapshot.max_seq.should eq(32)
    snapshot.layer_count.should eq(hp.n_layer)
    snapshot.records.size.should be > 0
    snapshot.byte_size.should be > 0

    restored = ML::GGUF::Qwen35StateSnapshot.restore(snapshot, hp)
    live_logits_state = live.fork
    restored_logits_state = restored.fork

    live_top, live_logit = ML::GGUF::Qwen35CPU.forward_top1(w, 11751_i32, prompt.size.to_i32, live)
    restored_top, restored_logit = ML::GGUF::Qwen35CPU.forward_top1(w, 11751_i32, prompt.size.to_i32, restored)

    restored_top.should eq(live_top)
    restored_logit.should be_close(live_logit, 1e-4_f32)

    live_logits = ML::GGUF::Qwen35CPU.forward(w, 11751_i32, prompt.size.to_i32, live_logits_state)
    restored_logits = ML::GGUF::Qwen35CPU.forward(w, 11751_i32, prompt.size.to_i32, restored_logits_state)
    cosine(live_logits, restored_logits).should be_close(1.0, 1e-6)
    max_delta(live_logits, restored_logits).should be < 1e-4

    checked = 0
    live.layers.each_with_index do |src_layer, i|
      restored_layer = restored.layers[i]
      {
        {src_layer.k_cache_buf, restored_layer.k_cache_buf},
        {src_layer.v_cache_buf, restored_layer.v_cache_buf},
        {src_layer.conv_state_buf, restored_layer.conv_state_buf},
        {src_layer.ssm_state_buf, restored_layer.ssm_state_buf},
      }.each do |src_opt, restored_opt|
        next unless src = src_opt

        restored_buf = restored_opt.not_nil!
        restored_buf.handle.should_not eq(src.handle)
        restored_buf.size.should eq(src.size)
        checked += 1
      end
    end
    checked.should be > 0
  end

  it "can restore into CPU arrays when Metal buffers are not requested" do
    w = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_SNAPSHOT)
    hp = w.hparams

    state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 16)
    ML::GGUF::Qwen35CPU.forward_top1(w, 0_i32, 0_i32, state)

    snapshot = ML::GGUF::Qwen35StateSnapshot.capture(state)
    restored = ML::GGUF::Qwen35StateSnapshot.restore(snapshot, hp, prefer_metal: false)

    restored.layers.any? { |layer| layer.k_cache || layer.v_cache || layer.conv_state || layer.ssm_state }.should be_true
    restored.layers.all? { |layer| layer.k_cache_buf.nil? && layer.v_cache_buf.nil? && layer.conv_state_buf.nil? && layer.ssm_state_buf.nil? }.should be_true
  end

  it "writes and reads a fail-closed .qkv artifact" do
    w = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_SNAPSHOT)
    hp = w.hparams

    state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 16)
    ML::GGUF::Qwen35CPU.forward_top1(w, 0_i32, 0_i32, state)

    snapshot = ML::GGUF::Qwen35StateSnapshot.capture(state)
    path = File.tempname("qwen35-state", ".qkv")
    begin
      info = ML::GGUF::Qwen35StateSnapshot.write_artifact(snapshot, path)
      info.byte_size.should be > 0
      info.sha256.size.should eq(64)

      loaded = ML::GGUF::Qwen35StateSnapshot.read_artifact(path, expected_sha256: info.sha256)
      loaded.max_seq.should eq(snapshot.max_seq)
      loaded.layer_count.should eq(snapshot.layer_count)
      loaded.records.size.should eq(snapshot.records.size)
      loaded.byte_size.should eq(snapshot.byte_size)

      File.open(path, "r+") do |file|
        file.seek(-1, IO::Seek::End)
        byte = file.read_byte.not_nil!
        file.seek(-1, IO::Seek::End)
        file.write_byte(byte ^ 0xff)
      end

      expect_raises(ArgumentError, /sha256 mismatch/) do
        ML::GGUF::Qwen35StateSnapshot.read_artifact(path, expected_sha256: info.sha256)
      end
    ensure
      File.delete(path) if File.exists?(path)
    end
  end
end

private def cosine(a : Array(Float32), b : Array(Float32)) : Float64
  dot = 0.0_f64
  aa = 0.0_f64
  bb = 0.0_f64
  a.each_with_index do |av, i|
    bv = b[i]
    dot += av.to_f64 * bv.to_f64
    aa += av.to_f64 * av.to_f64
    bb += bv.to_f64 * bv.to_f64
  end
  dot / Math.sqrt(aa * bb)
end

private def max_delta(a : Array(Float32), b : Array(Float32)) : Float32
  max = 0.0_f32
  a.each_with_index do |av, i|
    delta = (av - b[i]).abs
    max = delta if delta > max
  end
  max
end

private def synthetic_snapshot(values : Array(Float32)) : ML::GGUF::Qwen35StateSnapshot::Snapshot
  ML::GGUF::Qwen35StateSnapshot::Snapshot.new(
    max_seq: 8,
    layer_count: 1,
    positions: [1_i32],
    records: [
      ML::GGUF::Qwen35StateSnapshot::Record.new(
        0,
        ML::GGUF::Qwen35StateSnapshot::RecordKind::SsmState,
        bytes_from(values),
        ML::StorageMode::Shared,
      ),
    ],
  )
end

private def bytes_from(values : Array(Float32)) : Bytes
  bytes = Bytes.new(values.size * sizeof(Float32))
  values.each_with_index do |value, i|
    bits = value.unsafe_as(UInt32)
    offset = i * sizeof(Float32)
    bytes[offset] = (bits & 0xff).to_u8
    bytes[offset + 1] = ((bits >> 8) & 0xff).to_u8
    bytes[offset + 2] = ((bits >> 16) & 0xff).to_u8
    bytes[offset + 3] = ((bits >> 24) & 0xff).to_u8
  end
  bytes
end

private def floats_from(bytes : Bytes) : Array(Float32)
  values = [] of Float32
  (bytes.size // sizeof(Float32)).times do |i|
    offset = i * sizeof(Float32)
    bits = bytes[offset].to_u32 |
           (bytes[offset + 1].to_u32 << 8) |
           (bytes[offset + 2].to_u32 << 16) |
           (bytes[offset + 3].to_u32 << 24)
    values << bits.unsafe_as(Float32)
  end
  values
end
