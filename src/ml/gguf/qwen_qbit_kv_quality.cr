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

    # Mutable only because a quality run retires many independent spans. The
    # production wire format remains immutable and carries no selector state.
    class TierCounts
      getter p4 = 0_i64
      getter p5 = 0_i64
      getter bf16 = 0_i64
      getter f32 = 0_i64

      def observe!(tier : QwenQBitAdaptiveKV::Tier) : Nil
        case tier
        when QwenQBitAdaptiveKV::Tier::P4   then @p4 += 1
        when QwenQBitAdaptiveKV::Tier::P5   then @p5 += 1
        when QwenQBitAdaptiveKV::Tier::BF16 then @bf16 += 1
        when QwenQBitAdaptiveKV::Tier::F32  then @f32 += 1
        end
      end

      def total : Int64
        @p4 + @p5 + @bf16 + @f32
      end
    end

    # Local calibration proxy: the worst reconstruction residual in a
    # semantic row, normalized by that row's standard deviation. This metric
    # is scale-invariant and outlier-sensitive, but it is not a model-quality
    # gate; top-1/top-2/ECS/free text remain authoritative.
    def max_normalized_error(row : Array(Float32), precision : Int32) : Float64
      unless row.size == QwenQBitAdaptiveKV::ROW_VALUES
        raise ArgumentError.new("adaptive selector requires one 256-value semantic row")
      end
      unless precision == 4 || precision == 5
        raise ArgumentError.new("adaptive selector precision must be p4 or p5")
      end
      raise ArgumentError.new("adaptive selector row must be finite") unless row.all?(&.finite?)

      mean = row.sum(0.0_f64) { |value| value.to_f64 } / row.size
      variance = row.sum(0.0_f64) do |value|
        delta = value.to_f64 - mean
        delta * delta
      end / row.size
      scale = Math.sqrt(variance)
      decoded = QwenQBitGaussianCodec.decode(
        QwenQBitGaussianCodec.encode(row, block_size: row.size.to_i32, precision: precision),
      )
      maximum = row.each_with_index.max_of do |value, index|
        (value.to_f64 - decoded[index].to_f64).abs
      end
      return 0.0_f64 if maximum == 0.0_f64
      return Float64::INFINITY if scale == 0.0_f64
      maximum / scale
    end

    # Choose the cheapest row representation that satisfies the local proxy.
    # BF16 is the bounded escape tier; F32 remains reserved for explicit
    # calibration maps and corruption/debugging controls.
    def select_tier(row : Array(Float32), max_error : Float64) : QwenQBitAdaptiveKV::Tier
      validate_normalized_error_bound(max_error)
      return QwenQBitAdaptiveKV::Tier::P4 if max_normalized_error(row, 4) <= max_error
      return QwenQBitAdaptiveKV::Tier::P5 if max_normalized_error(row, 5) <= max_error
      QwenQBitAdaptiveKV::Tier::BF16
    end

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

    # Diagnostic row-local selector. It evaluates completed rows only, then
    # roundtrips the selected canonical adaptive representation into the old
    # Float32 quality cache. This does not alter resident GPU allocation.
    def roundtrip_layers_span_selected!(
      layers : Array(Qwen35CPU::LayerState),
      full_attention_layers : Array(Int32),
      max_seq : Int32,
      n_head_kv : Int32,
      head_dim : Int32,
      start_pos : Int32,
      token_count : Int32,
      max_error : Float64,
      counts : TierCounts = TierCounts.new,
    ) : Stats
      validate_adaptive_shape(max_seq, n_head_kv, head_dim, start_pos, token_count)
      validate_normalized_error_bound(max_error)

      kv_dim = n_head_kv * head_dim
      value_offset = start_pos * kv_dim
      value_count = token_count * kv_dim
      stats = Stats.new(0_i64, 0_i64, 0_i64)
      full_attention_layers.each do |layer_index|
        if layer_index < 0 || layer_index >= layers.size
          raise ArgumentError.new("full-attention layer index outside state")
        end
        layer = layers[layer_index]
        stats = roundtrip_owner_selected!(
          layer.k_cache_buf, layer.k_cache, value_offset, value_count,
          head_dim, max_error, counts, stats,
        )
        stats = roundtrip_owner_selected!(
          layer.v_cache_buf, layer.v_cache, value_offset, value_count,
          head_dim, max_error, counts, stats,
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

    private def roundtrip_owner_selected!(
      buffer : ML::MetalBuffer?,
      values : Array(Float32)?,
      value_offset : Int32,
      value_count : Int32,
      head_dim : Int32,
      max_error : Float64,
      counts : TierCounts,
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
        encoded = encode_selected(live, head_dim, max_error, counts)
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
        encoded = encode_selected(live, head_dim, max_error, counts)
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

    private def encode_selected(values : Array(Float32),
                                head_dim : Int32,
                                max_error : Float64,
                                counts : TierCounts) : QwenQBitAdaptiveKV::Encoded
      rows = values.size // head_dim
      tiers = Array(QwenQBitAdaptiveKV::Tier).new(rows)
      rows.times do |row|
        selected = select_tier(values[row * head_dim, head_dim], max_error)
        tiers << selected
        counts.observe!(selected)
      end
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

    private def validate_normalized_error_bound(max_error : Float64) : Nil
      unless max_error.finite? && max_error > 0.0_f64
        raise ArgumentError.new("adaptive normalized-error bound must be finite and positive")
      end
    end
  end
end
