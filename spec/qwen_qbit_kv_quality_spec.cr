require "./spec_helper"
require "../src/ml/gguf/qwen_qbit_kv_quality"

describe ML::GGUF::QwenQBitKVQuality do
  it "roundtrips an appended CPU KV row without touching adjacent rows" do
    head_dim = 256
    n_head_kv = 2
    kv_dim = n_head_kv * head_dim
    layer = ML::GGUF::Qwen35CPU::LayerState.new
    original_k = Array(Float32).new(3 * kv_dim) { |i| Math.sin(i.to_f64 * 0.021).to_f32 }
    original_v = Array(Float32).new(3 * kv_dim) { |i| Math.cos(i.to_f64 * 0.019).to_f32 }
    layer.k_cache = original_k.dup
    layer.v_cache = original_v.dup
    excluded_layer = ML::GGUF::Qwen35CPU::LayerState.new
    excluded_layer.k_cache = original_k.dup
    excluded_layer.v_cache = original_v.dup

    stats = ML::GGUF::QwenQBitKVQuality.roundtrip_layers_span!(
      [excluded_layer, layer], [1], 3, n_head_kv, head_dim, 1, 1, 5,
    )

    stats.raw_bytes.should eq((2 * kv_dim * sizeof(Float32)).to_i64)
    layer.k_cache.not_nil![0, kv_dim].should eq(original_k[0, kv_dim])
    layer.k_cache.not_nil![2 * kv_dim, kv_dim].should eq(original_k[2 * kv_dim, kv_dim])
    layer.k_cache.not_nil![kv_dim, kv_dim].should_not eq(original_k[kv_dim, kv_dim])
    excluded_layer.k_cache.should eq(original_k)
    excluded_layer.v_cache.should eq(original_v)
  end

  it "produces the same rows when an aligned live prefix retires in chunks" do
    head_dim = 256
    n_head_kv = 2
    kv_dim = n_head_kv * head_dim
    original_k = Array(Float32).new(4 * kv_dim) { |i| Math.sin(i.to_f64 * 0.007).to_f32 }
    original_v = Array(Float32).new(4 * kv_dim) { |i| Math.cos(i.to_f64 * 0.011).to_f32 }
    whole = ML::GGUF::Qwen35CPU::LayerState.new
    chunked = ML::GGUF::Qwen35CPU::LayerState.new
    whole.k_cache = original_k.dup
    whole.v_cache = original_v.dup
    chunked.k_cache = original_k.dup
    chunked.v_cache = original_v.dup

    whole_stats = ML::GGUF::QwenQBitKVQuality.roundtrip_layers_span!(
      [whole], [0], 4, n_head_kv, head_dim, 0, 4, 4,
    )
    first_stats = ML::GGUF::QwenQBitKVQuality.roundtrip_layers_span!(
      [chunked], [0], 4, n_head_kv, head_dim, 0, 2, 4,
    )
    second_stats = ML::GGUF::QwenQBitKVQuality.roundtrip_layers_span!(
      [chunked], [0], 4, n_head_kv, head_dim, 2, 2, 4,
    )

    chunked.k_cache.should eq(whole.k_cache)
    chunked.v_cache.should eq(whole.v_cache)
    (first_stats.raw_bytes + second_stats.raw_bytes).should eq(whole_stats.raw_bytes)
    (first_stats.payload_bytes + second_stats.payload_bytes).should eq(whole_stats.payload_bytes)
  end

  it "applies an externally calibrated adaptive tier per full-attention layer" do
    head_dim = 256
    n_head_kv = 1
    kv_dim = n_head_kv * head_dim
    excluded = ML::GGUF::Qwen35CPU::LayerState.new
    selected = ML::GGUF::Qwen35CPU::LayerState.new
    original_k = Array(Float32).new(2 * kv_dim) { |i| Math.sin(i.to_f64 * 0.013).to_f32 }
    original_v = Array(Float32).new(2 * kv_dim) { |i| Math.cos(i.to_f64 * 0.017).to_f32 }
    excluded.k_cache = original_k.dup
    excluded.v_cache = original_v.dup
    selected.k_cache = original_k.dup
    selected.v_cache = original_v.dup
    stats = ML::GGUF::QwenQBitKVQuality.roundtrip_layers_span_adaptive!(
      [excluded, selected], [1], 2, n_head_kv, head_dim, 0, 2,
      ML::GGUF::QwenQBitAdaptiveKV::Tier::P4,
      {1 => ML::GGUF::QwenQBitAdaptiveKV::Tier::F32},
    )

    selected.k_cache.should eq(original_k)
    selected.v_cache.should eq(original_v)
    excluded.k_cache.should eq(original_k)
    excluded.v_cache.should eq(original_v)
    stats.raw_bytes.should eq((2 * 2 * kv_dim * sizeof(Float32)).to_i64)
    per_owner = 2 * (ML::GGUF::QwenQBitAdaptiveKV::BASE_ROW_BYTES +
                     ML::GGUF::QwenQBitAdaptiveKV::METADATA_BYTES +
                     ML::GGUF::QwenQBitAdaptiveKV::F32_SIDECAR_BYTES)
    stats.payload_bytes.should eq((2 * per_owner).to_i64)
  end

  it "roundtrips the requested span in shared Metal storage" do
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    head_dim = 256
    n_head_kv = 1
    kv_dim = n_head_kv * head_dim
    original_k = Array(Float32).new(3 * kv_dim) { |i| Math.sin(i.to_f64 * 0.009).to_f32 }
    original_v = Array(Float32).new(3 * kv_dim) { |i| Math.cos(i.to_f64 * 0.015).to_f32 }
    expected_k = ML::GGUF::QwenQBitGaussianCodec.decode(
      ML::GGUF::QwenQBitGaussianCodec.encode(original_k[kv_dim, kv_dim], block_size: head_dim, precision: 4),
    )
    layer = ML::GGUF::Qwen35CPU::LayerState.new
    layer.k_cache_buf = ML::MetalBuffer.from_array(original_k)
    layer.v_cache_buf = ML::MetalBuffer.from_array(original_v)

    begin
      ML::GGUF::QwenQBitKVQuality.roundtrip_layers_span!(
        [layer], [0], 3, n_head_kv, head_dim, 1, 1, 4,
      )
      actual_k = layer.k_cache_buf.not_nil!.read(original_k.size)
      actual_k[0, kv_dim].should eq(original_k[0, kv_dim])
      actual_k[kv_dim, kv_dim].should eq(expected_k)
      actual_k[2 * kv_dim, kv_dim].should eq(original_k[2 * kv_dim, kv_dim])
    ensure
      layer.k_cache_buf.try(&.release)
      layer.v_cache_buf.try(&.release)
      layer.k_cache_buf = nil
      layer.v_cache_buf = nil
    end
  end

  it "roundtrips an adaptive tier in shared Metal storage" do
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    head_dim = 256
    n_head_kv = 1
    original_k = Array(Float32).new(2 * head_dim) { |i| Math.sin(i.to_f64 * 0.023).to_f32 }
    original_v = Array(Float32).new(2 * head_dim) { |i| Math.cos(i.to_f64 * 0.029).to_f32 }
    tiers = [ML::GGUF::QwenQBitAdaptiveKV::Tier::BF16]
    expected_k = ML::GGUF::QwenQBitAdaptiveKV.decode(
      ML::GGUF::QwenQBitAdaptiveKV.encode(original_k[head_dim, head_dim], tiers),
    )
    layer = ML::GGUF::Qwen35CPU::LayerState.new
    layer.k_cache_buf = ML::MetalBuffer.from_array(original_k)
    layer.v_cache_buf = ML::MetalBuffer.from_array(original_v)

    begin
      ML::GGUF::QwenQBitKVQuality.roundtrip_layers_span_adaptive!(
        [layer], [0], 2, n_head_kv, head_dim, 1, 1,
        ML::GGUF::QwenQBitAdaptiveKV::Tier::BF16,
        {} of Int32 => ML::GGUF::QwenQBitAdaptiveKV::Tier,
      )
      actual_k = layer.k_cache_buf.not_nil!.read(original_k.size)
      actual_k[0, head_dim].should eq(original_k[0, head_dim])
      actual_k[head_dim, head_dim].should eq(expected_k)
    ensure
      layer.k_cache_buf.try(&.release)
      layer.v_cache_buf.try(&.release)
      layer.k_cache_buf = nil
      layer.v_cache_buf = nil
    end
  end

  it "rejects non-Qwen3.8 head geometry and out-of-range spans" do
    expect_raises(ArgumentError, /head dimension 256/) do
      ML::GGUF::QwenQBitKVQuality.roundtrip_layers_span!(
        [] of ML::GGUF::Qwen35CPU::LayerState, [] of Int32,
        1, 1, 128, 0, 1, 4,
      )
    end

    expect_raises(ArgumentError, /outside cache capacity/) do
      ML::GGUF::QwenQBitKVQuality.roundtrip_layers_span!(
        [] of ML::GGUF::Qwen35CPU::LayerState, [] of Int32,
        1, 1, 256, 1, 1, 4,
      )
    end

    expect_raises(ArgumentError, /not a full-attention layer/) do
      ML::GGUF::QwenQBitKVQuality.roundtrip_layers_span_adaptive!(
        [] of ML::GGUF::Qwen35CPU::LayerState, [] of Int32,
        1, 1, 256, 0, 1,
        ML::GGUF::QwenQBitAdaptiveKV::Tier::P4,
        {0 => ML::GGUF::QwenQBitAdaptiveKV::Tier::BF16},
      )
    end
  end
end
