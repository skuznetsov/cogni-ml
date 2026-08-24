require "./qwen35_cpu"
require "./qwen_qbit_adaptive_kv"
require "./qwen_qbit_gaussian_codec"

module ML::GGUF
  # Diagnostic helpers for measuring the model-quality effect of the same
  # p4/p5 values that a future resident KV path would consume. These helpers
  # deliberately roundtrip back into the existing Float32 cache; they do not
  # change production cache ownership or routing.
  module QwenQBitKVQuality
    extend self

    record Stats,
      raw_bytes : Int64,
      payload_bytes : Int64,
      blocks : Int64

    # Quantize and reconstruct only an appended token span in existing cache
    # storage. This models retire-then-pack semantics: the current attention
    # has completed, and only future attention observes the reconstructed row.
    def roundtrip_layers_span!(layers : Array(Qwen35CPU::LayerState),
                               full_attention_layers : Array(Int32),
                               max_seq : Int32,
                               n_head_kv : Int32,
                               head_dim : Int32,
                               start_pos : Int32,
                               token_count : Int32,
                               precision : Int32) : Stats
      validate_shape(max_seq, n_head_kv, head_dim, start_pos, token_count, precision)
      kv_dim = n_head_kv * head_dim
      value_offset = start_pos * kv_dim
      value_count = token_count * kv_dim
      stats = Stats.new(0_i64, 0_i64, 0_i64)

      full_attention_layers.each do |layer_index|
        if layer_index < 0 || layer_index >= layers.size
          raise ArgumentError.new("full-attention layer index outside state")
        end
        layer = layers[layer_index]
        stats = roundtrip_owner!(layer.k_cache_buf, layer.k_cache, value_offset, value_count, head_dim, precision, stats)
        stats = roundtrip_owner!(layer.v_cache_buf, layer.v_cache, value_offset, value_count, head_dim, precision, stats)
      end
      stats
    end

    # Diagnostic policy hook for model-calibrated adaptive rows. This first
    # bounded selector assigns one tier per full-attention layer; the wire
    # format itself remains row-addressable and can admit a finer map later.
    def roundtrip_layers_span_adaptive!(
      layers : Array(Qwen35CPU::LayerState),
      full_attention_layers : Array(Int32),
      max_seq : Int32,
      n_head_kv : Int32,
      head_dim : Int32,
      start_pos : Int32,
      token_count : Int32,
      default_tier : QwenQBitAdaptiveKV::Tier,
      layer_tiers : Hash(Int32, QwenQBitAdaptiveKV::Tier),
    ) : Stats
      validate_adaptive_shape(max_seq, n_head_kv, head_dim, start_pos, token_count)
      layer_tiers.each_key do |layer_index|
        unless full_attention_layers.includes?(layer_index)
          raise ArgumentError.new("adaptive KV tier override is not a full-attention layer")
        end
      end

      kv_dim = n_head_kv * head_dim
      value_offset = start_pos * kv_dim
      value_count = token_count * kv_dim
      stats = Stats.new(0_i64, 0_i64, 0_i64)
      full_attention_layers.each do |layer_index|
        if layer_index < 0 || layer_index >= layers.size
          raise ArgumentError.new("full-attention layer index outside state")
        end
        selected_tier = layer_tiers[layer_index]? || default_tier
        layer = layers[layer_index]
        stats = roundtrip_owner_adaptive!(
          layer.k_cache_buf, layer.k_cache, value_offset, value_count,
          head_dim, selected_tier, stats,
        )
        stats = roundtrip_owner_adaptive!(
          layer.v_cache_buf, layer.v_cache, value_offset, value_count,
          head_dim, selected_tier, stats,
        )
      end
      stats
    end

    private def roundtrip_owner!(buffer : ML::MetalBuffer?,
                                 values : Array(Float32)?,
                                 value_offset : Int32,
                                 value_count : Int32,
                                 head_dim : Int32,
                                 precision : Int32,
                                 stats : Stats) : Stats
      live = Array(Float32).new(value_count, 0.0_f32)
      if owner = buffer
        unless owner.storage_mode == ML::StorageMode::Shared
          raise ArgumentError.new("KV quality roundtrip requires shared Metal storage")
        end
        byte_end = (value_offset.to_i64 + value_count) * sizeof(Float32)
        raise ArgumentError.new("KV buffer is smaller than the requested span") if byte_end > owner.size
        Slice.new(live.to_unsafe, value_count).copy_from(
          Slice.new(owner.contents.as(Pointer(Float32)) + value_offset, value_count),
        )
        encoded = QwenQBitGaussianCodec.encode(live, block_size: head_dim, precision: precision)
        decoded = QwenQBitGaussianCodec.decode(encoded)
        Slice.new(owner.contents.as(Pointer(Float32)) + value_offset, value_count).copy_from(
          Slice.new(decoded.to_unsafe, value_count),
        )
        add_stats(stats, value_count, encoded)
      elsif owner = values
        raise ArgumentError.new("KV array is smaller than the requested span") if value_offset + value_count > owner.size
        Slice.new(live.to_unsafe, value_count).copy_from(
          Slice.new(owner.to_unsafe + value_offset, value_count),
        )
        encoded = QwenQBitGaussianCodec.encode(live, block_size: head_dim, precision: precision)
        decoded = QwenQBitGaussianCodec.decode(encoded)
        Slice.new(owner.to_unsafe + value_offset, value_count).copy_from(
          Slice.new(decoded.to_unsafe, value_count),
        )
        add_stats(stats, value_count, encoded)
      else
        raise ArgumentError.new("full-attention layer has no KV storage")
      end
    end

    private def roundtrip_owner_adaptive!(
      buffer : ML::MetalBuffer?,
      values : Array(Float32)?,
      value_offset : Int32,
      value_count : Int32,
      head_dim : Int32,
      tier : QwenQBitAdaptiveKV::Tier,
      stats : Stats,
    ) : Stats
      live = Array(Float32).new(value_count, 0.0_f32)
      if owner = buffer
        unless owner.storage_mode == ML::StorageMode::Shared
          raise ArgumentError.new("KV quality roundtrip requires shared Metal storage")
        end
        byte_end = (value_offset.to_i64 + value_count) * sizeof(Float32)
        raise ArgumentError.new("KV buffer is smaller than the requested span") if byte_end > owner.size
        Slice.new(live.to_unsafe, value_count).copy_from(
          Slice.new(owner.contents.as(Pointer(Float32)) + value_offset, value_count),
        )
        encoded = encode_adaptive(live, head_dim, tier)
        decoded = QwenQBitAdaptiveKV.decode(encoded)
        Slice.new(owner.contents.as(Pointer(Float32)) + value_offset, value_count).copy_from(
          Slice.new(decoded.to_unsafe, value_count),
        )
        add_adaptive_stats(stats, value_count, encoded)
      elsif owner = values
        raise ArgumentError.new("KV array is smaller than the requested span") if value_offset + value_count > owner.size
        Slice.new(live.to_unsafe, value_count).copy_from(
          Slice.new(owner.to_unsafe + value_offset, value_count),
        )
        encoded = encode_adaptive(live, head_dim, tier)
        decoded = QwenQBitAdaptiveKV.decode(encoded)
        Slice.new(owner.to_unsafe + value_offset, value_count).copy_from(
          Slice.new(decoded.to_unsafe, value_count),
        )
        add_adaptive_stats(stats, value_count, encoded)
      else
        raise ArgumentError.new("full-attention layer has no KV storage")
      end
    end

    private def encode_adaptive(values : Array(Float32),
                                head_dim : Int32,
                                tier : QwenQBitAdaptiveKV::Tier) : QwenQBitAdaptiveKV::Encoded
      rows = values.size // head_dim
      tiers = Array(QwenQBitAdaptiveKV::Tier).new(rows, tier)
      QwenQBitAdaptiveKV.encode(values, tiers, block_size: head_dim)
    end

    private def add_stats(stats : Stats,
                          value_count : Int32,
                          encoded : QwenQBitGaussianCodec::Encoded) : Stats
      Stats.new(
        stats.raw_bytes + value_count.to_i64 * sizeof(Float32),
        stats.payload_bytes + encoded.payload.size,
        stats.blocks + QwenQBitGaussianCodec.tile_count(encoded),
      )
    end

    private def add_adaptive_stats(stats : Stats,
                                   value_count : Int32,
                                   encoded : QwenQBitAdaptiveKV::Encoded) : Stats
      Stats.new(
        stats.raw_bytes + value_count.to_i64 * sizeof(Float32),
        stats.payload_bytes + encoded.payload_bytes,
        stats.blocks + value_count // encoded.block_size,
      )
    end

    private def validate_shape(max_seq : Int32,
                               n_head_kv : Int32,
                               head_dim : Int32,
                               start_pos : Int32,
                               token_count : Int32,
                               precision : Int32) : Nil
      raise ArgumentError.new("KV cache capacity must be positive") unless max_seq > 0
      raise ArgumentError.new("KV head count must be positive") unless n_head_kv > 0
      raise ArgumentError.new("KV quality probe requires Qwen3.8 head dimension 256") unless head_dim == 256
      raise ArgumentError.new("KV quality precision must be p4 or p5") unless precision == 4 || precision == 5
      unless start_pos >= 0 && token_count > 0 && start_pos.to_i64 + token_count <= max_seq
        raise ArgumentError.new("KV roundtrip span outside cache capacity")
      end
    end

    private def validate_adaptive_shape(max_seq : Int32,
                                        n_head_kv : Int32,
                                        head_dim : Int32,
                                        start_pos : Int32,
                                        token_count : Int32) : Nil
      raise ArgumentError.new("KV cache capacity must be positive") unless max_seq > 0
      raise ArgumentError.new("KV head count must be positive") unless n_head_kv > 0
      raise ArgumentError.new("KV quality probe requires Qwen3.8 head dimension 256") unless head_dim == 256
      unless start_pos >= 0 && token_count > 0 && start_pos.to_i64 + token_count <= max_seq
        raise ArgumentError.new("KV roundtrip span outside cache capacity")
      end
    end
  end
end
