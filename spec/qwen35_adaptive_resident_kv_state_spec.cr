require "./spec_helper"
require "../src/ml/gguf/reader"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_state_snapshot"
require "../src/ml/gguf/qwen_qbit_state_snapshot"

QWEN_38_ADAPTIVE_STATE    = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf"
QWEN_35_9B_ADAPTIVE_STATE = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"

private def with_adaptive_state_env(layer : String?, tier : String?, map : String? = nil, &)
  keys = [
    "QWEN35_ADAPTIVE_RESIDENT_KV_LAYER",
    "QWEN35_ADAPTIVE_RESIDENT_KV_TIER",
    "QWEN35_ADAPTIVE_RESIDENT_KV_MAP",
    "QWEN35_PREFILL_APPEND_CMD_OFF",
    "QWEN35_PREFILL_RESIDENT_BOUNDARY_OFF",
    "QWEN35_PREFILL_FUSE_FULL_REC_OFF",
    "QWEN35_FULL_PREFILL_CHUNK_OFF",
    "QWEN35_PREFILL_REC_RUN_OFF",
    "QWEN35_PREFILL_CHUNK_OFF",
  ]
  old = keys.to_h { |key| {key, ENV[key]?} }
  keys.each { |key| ENV.delete(key) }
  if layer
    ENV["QWEN35_ADAPTIVE_RESIDENT_KV_LAYER"] = layer
  else
    ENV.delete("QWEN35_ADAPTIVE_RESIDENT_KV_LAYER")
  end
  if tier
    ENV["QWEN35_ADAPTIVE_RESIDENT_KV_TIER"] = tier
  else
    ENV.delete("QWEN35_ADAPTIVE_RESIDENT_KV_TIER")
  end
  if map
    ENV["QWEN35_ADAPTIVE_RESIDENT_KV_MAP"] = map
  else
    ENV.delete("QWEN35_ADAPTIVE_RESIDENT_KV_MAP")
  end
  yield
ensure
  old.try &.each do |key, value|
    if value
      ENV[key] = value
    else
      ENV.delete(key)
    end
  end
end

private def release_adaptive_state!(state : ML::GGUF::Qwen35CPU::State) : Nil
  ML::Metal::Device.synchronize
  state.layers.each do |layer|
    layer.k_cache_buf.try(&.release)
    layer.v_cache_buf.try(&.release)
    layer.conv_state_buf.try(&.release)
    layer.ssm_state_buf.try(&.release)
    layer.adaptive_kv.try(&.release)
  end
end

describe ML::GGUF::Qwen35CPU do
  it "keeps the ordinary F32 KV owner when adaptive resident KV is not configured" do
    pending!("Qwen3.8 27B model not present") unless File.exists?(QWEN_38_ADAPTIVE_STATE)
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    gguf = ML::GGUF::GGUFFile.new(QWEN_38_ADAPTIVE_STATE)
    hp = ML::GGUF::Qwen35Hparams.new(gguf)
    state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 8)
    with_adaptive_state_env(nil, nil) do
      ML::GGUF::Qwen35CPU.prepare_state_metal!(state, hp)
    end

    raw_buffer_bytes = (state.max_seq * hp.n_head_kv * hp.head_dim * sizeof(Float32)).to_i64
    hp.full_attention_layers.each do |layer_index|
      layer = state.layers[layer_index]
      layer.adaptive_kv.should be_nil
      layer.k_cache_buf.not_nil!.size.should eq(raw_buffer_bytes)
      layer.v_cache_buf.not_nil!.size.should eq(raw_buffer_bytes)
    end
  ensure
    release_adaptive_state!(state) if state
    gguf.try(&.close)
  end

  it "can explicitly retain the F32 miss fallback when an adaptive map is configured" do
    pending!("Qwen3.8 27B model not present") unless File.exists?(QWEN_38_ADAPTIVE_STATE)
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    gguf = ML::GGUF::GGUFFile.new(QWEN_38_ADAPTIVE_STATE)
    hp = ML::GGUF::Qwen35Hparams.new(gguf)
    state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 8)
    with_adaptive_state_env(nil, nil, "p4;27=bf16,43=bf16,47=bf16,51=bf16") do
      ML::GGUF::Qwen35CPU.prepare_state_metal!(state, hp, admit_adaptive_resident_kv: false)
    end

    state.adaptive_kv_layer_indices.should be_empty
    hp.full_attention_layers.each do |layer_index|
      layer = state.layers[layer_index]
      layer.k_cache_buf.should_not be_nil
      layer.v_cache_buf.should_not be_nil
    end
  ensure
    release_adaptive_state!(state) if state
    gguf.try(&.close)
  end

  it "gives one selected GQA6 layer a sole compact KV owner" do
    pending!("Qwen3.8 27B model not present") unless File.exists?(QWEN_38_ADAPTIVE_STATE)
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    gguf = ML::GGUF::GGUFFile.new(QWEN_38_ADAPTIVE_STATE)
    hp = ML::GGUF::Qwen35Hparams.new(gguf)
    selected = hp.full_attention_layers.first
    state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 8)
    with_adaptive_state_env(selected.to_s, "bf16") do
      ML::GGUF::Qwen35CPU.prepare_state_metal!(state, hp)
    end

    layer = state.layers[selected]
    layer.k_cache.should be_nil
    layer.v_cache.should be_nil
    layer.k_cache_buf.should be_nil
    layer.v_cache_buf.should be_nil
    cache = layer.adaptive_kv.not_nil!
    cache.cache_len.should eq(0)
    raw_kv_bytes = 2_i64 * state.max_seq * hp.n_head_kv * hp.head_dim * sizeof(Float32)
    cache.compressed_bytes.should be < raw_kv_bytes

    hp.full_attention_layers.reject { |layer_index| layer_index == selected }.each do |layer_index|
      ordinary = state.layers[layer_index]
      ordinary.adaptive_kv.should be_nil
      ordinary.k_cache_buf.should_not be_nil
      ordinary.v_cache_buf.should_not be_nil
    end

    expect_raises(ArgumentError, /fork.*unsupported/) { state.fork }
    expect_raises(ArgumentError, /snapshot.*unsupported/) do
      ML::GGUF::Qwen35StateSnapshot.capture(state)
    end
    expect_raises(ArgumentError, /tail clearing.*unsupported/) do
      ML::GGUF::Qwen35CPU.clear_kv_tail_metal!(state, hp, 0)
    end
    expect_raises(ArgumentError, /fork\/copy.*unsupported/) do
      ML::GGUF::Qwen35CPU.copy_state_metal_used!(state, state, hp)
    end
    expect_raises(ArgumentError, /checkpoint swapping.*unsupported/) do
      ML::GGUF::Qwen35CPU.swap_recurrent_state_metal_buffers!(state, state, hp)
    end
  ensure
    release_adaptive_state!(state) if state
    gguf.try(&.close)
  end

  it "gives every Qwen3.8 full-attention layer a sole mixed-tier compact KV owner" do
    pending!("Qwen3.8 27B model not present") unless File.exists?(QWEN_38_ADAPTIVE_STATE)
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    gguf = ML::GGUF::GGUFFile.new(QWEN_38_ADAPTIVE_STATE)
    hp = ML::GGUF::Qwen35Hparams.new(gguf)
    state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 8)
    with_adaptive_state_env(nil, nil, "p4;27=bf16,43=bf16,47=bf16,51=bf16") do
      ML::GGUF::Qwen35CPU.prepare_state_metal!(state, hp)
    end

    state.adaptive_kv_layer_indices.should eq(hp.full_attention_layers)
    adaptive_bytes = 0_i64
    hp.full_attention_layers.each do |layer_index|
      layer = state.layers[layer_index]
      layer.k_cache.should be_nil
      layer.v_cache.should be_nil
      layer.k_cache_buf.should be_nil
      layer.v_cache_buf.should be_nil
      adaptive_bytes += layer.adaptive_kv.not_nil!.compressed_bytes
    end
    raw_kv_bytes = hp.full_attention_layers.size.to_i64 * 2_i64 *
                   state.max_seq * hp.n_head_kv * hp.head_dim * sizeof(Float32)
    adaptive_bytes.should be < raw_kv_bytes
    state.layers[27].adaptive_kv.not_nil!.compressed_bytes.should be >
                                                                  state.layers[3].adaptive_kv.not_nil!.compressed_bytes
  ensure
    release_adaptive_state!(state) if state
    gguf.try(&.close)
  end

  it "rejects an incomplete cold adaptive restore before publishing KV rows" do
    pending!("Qwen3.8 27B model not present") unless File.exists?(QWEN_38_ADAPTIVE_STATE)
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    gguf = ML::GGUF::GGUFFile.new(QWEN_38_ADAPTIVE_STATE)
    hp = ML::GGUF::Qwen35Hparams.new(gguf)
    state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 8)
    with_adaptive_state_env(nil, nil, "p4;27=bf16,43=bf16,47=bf16,51=bf16") do
      ML::GGUF::Qwen35CPU.prepare_state_metal!(state, hp)
    end

    encoded = ML::GGUF::QwenQBitGaussianCodec.encode([0.0_f32], 8, 7)
    native_bytes = ML::GGUF::QwenQBitNativeWriter.encode([
      ML::GGUF::QwenQBitNativeWriter::Record.new(
        7_u64,
        hp.recurrent_layers.first,
        ML::GGUF::Qwen35StateSnapshot::RecordKind::ConvState.value,
        encoded,
      ),
    ])
    stream = ML::GGUF::QwenQBitNativeBlock.parse_stream(native_bytes)
    exact_records = [] of ML::GGUF::Qwen35StateSnapshot::EncodedRecord
    exact = ML::GGUF::Qwen35StateSnapshot::EncodedSnapshot.new(
      state.max_seq,
      hp.n_layer,
      Array(Int32).new(hp.n_layer, 1_i32),
      exact_records,
      ML::GGUF::Qwen35StateSnapshot::RecordCodec::RawF32,
      0_i32,
    )

    expect_raises(ArgumentError, /record set mismatch/) do
      ML::GGUF::QwenQBitStateSnapshot.restore_admitted_native_stream_into_adaptive(
        stream,
        exact,
        7_u64,
        hp,
        state,
      )
    end
    hp.full_attention_layers.each do |layer_index|
      state.layers[layer_index].adaptive_kv.not_nil!.cache_len.should eq(0)
    end
  ensure
    release_adaptive_state!(state) if state
    gguf.try(&.close)
  end

  it "rejects changing an allocated all-layer tier map" do
    pending!("Qwen3.8 27B model not present") unless File.exists?(QWEN_38_ADAPTIVE_STATE)
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    gguf = ML::GGUF::GGUFFile.new(QWEN_38_ADAPTIVE_STATE)
    hp = ML::GGUF::Qwen35Hparams.new(gguf)
    state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 8)
    with_adaptive_state_env(nil, nil, "p4") do
      ML::GGUF::Qwen35CPU.prepare_state_metal!(state, hp)
    end
    with_adaptive_state_env(nil, nil, "p5") do
      expect_raises(ArgumentError, /tier map cannot change/) do
        ML::GGUF::Qwen35CPU.prepare_state_metal!(state, hp)
      end
    end
  ensure
    release_adaptive_state!(state) if state
    gguf.try(&.close)
  end

  it "rejects ambiguous and non-attention all-layer maps before allocation" do
    pending!("Qwen3.8 27B model not present") unless File.exists?(QWEN_38_ADAPTIVE_STATE)
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    gguf = ML::GGUF::GGUFFile.new(QWEN_38_ADAPTIVE_STATE)
    hp = ML::GGUF::Qwen35Hparams.new(gguf)

    with_adaptive_state_env("3", "p4", "p4") do
      state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 8)
      expect_raises(ArgumentError, /cannot be combined/) do
        ML::GGUF::Qwen35CPU.prepare_state_metal!(state, hp)
      end
      state.layers.all?(&.adaptive_kv.nil?).should be_true
    end

    with_adaptive_state_env(nil, nil, "p4;26=bf16") do
      state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 8)
      expect_raises(ArgumentError, /full-attention/) do
        ML::GGUF::Qwen35CPU.prepare_state_metal!(state, hp)
      end
      state.layers.all?(&.adaptive_kv.nil?).should be_true
    end

    with_adaptive_state_env(nil, nil, "p4;3=p5,3=bf16") do
      state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 8)
      expect_raises(ArgumentError, /duplicate layer/) do
        ML::GGUF::Qwen35CPU.prepare_state_metal!(state, hp)
      end
      state.layers.all?(&.adaptive_kv.nil?).should be_true
    end
  ensure
    gguf.try(&.close)
  end

  it "rejects unsupported GQA4 ownership before allocating adaptive KV" do
    pending!("Qwen3.5 9B model not present") unless File.exists?(QWEN_35_9B_ADAPTIVE_STATE)
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    gguf = ML::GGUF::GGUFFile.new(QWEN_35_9B_ADAPTIVE_STATE)
    hp = ML::GGUF::Qwen35Hparams.new(gguf)
    state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 8)
    selected = hp.full_attention_layers.first
    with_adaptive_state_env(selected.to_s, "p4") do
      expect_raises(ArgumentError, /GQA6/) do
        ML::GGUF::Qwen35CPU.prepare_state_metal!(state, hp)
      end
    end
    state.layers.all?(&.adaptive_kv.nil?).should be_true
  ensure
    release_adaptive_state!(state) if state
    gguf.try(&.close)
  end

  it "requires both selectors and the shared fused prefill corridor" do
    pending!("Qwen3.8 27B model not present") unless File.exists?(QWEN_38_ADAPTIVE_STATE)
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    gguf = ML::GGUF::GGUFFile.new(QWEN_38_ADAPTIVE_STATE)
    hp = ML::GGUF::Qwen35Hparams.new(gguf)
    selected = hp.full_attention_layers.first

    with_adaptive_state_env(selected.to_s, nil) do
      state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 8)
      expect_raises(ArgumentError, /both.*LAYER.*TIER/) do
        ML::GGUF::Qwen35CPU.prepare_state_metal!(state, hp)
      end
    end

    with_adaptive_state_env(selected.to_s, "p4") do
      ENV["QWEN35_PREFILL_APPEND_CMD_OFF"] = "1"
      state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 8)
      expect_raises(ArgumentError, /shared.*command/) do
        ML::GGUF::Qwen35CPU.prepare_state_metal!(state, hp)
      end
    end
  ensure
    gguf.try(&.close)
  end
end
