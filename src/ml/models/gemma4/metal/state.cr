# Resident KV state and reusable scratch buffers for Gemma4 Metal.

module ML::GGUF
  module Gemma4Metal
    class LayerState
      property k_cache : Array(Float32)
      property v_cache : Array(Float32)

      def initialize(kv_dim : Int32, max_seq : Int32)
        @k_cache = Array(Float32).new(max_seq * kv_dim, 0.0_f32)
        @v_cache = Array(Float32).new(max_seq * kv_dim, 0.0_f32)
      end
    end

    class State
      getter layers : Array(LayerState)
      getter max_seq : Int32

      def initialize(hp : Gemma4Hparams, @max_seq : Int32 = 1024)
        @layers = Array(LayerState).new(hp.n_layer) do |il|
          kv_dim = hp.n_head_kv(il) * hp.head_dim_for_layer(il)
          LayerState.new(kv_dim, @max_seq)
        end
      end
    end


    {% unless flag?(:cpu_only) %}
      class ResidentLayerState
        getter k_cache_buf : ML::MetalBuffer
        getter v_cache_buf : ML::MetalBuffer
        getter k_cache_h16_buf : ML::MetalBuffer?
        getter v_cache_h16_buf : ML::MetalBuffer?
        getter kv_dim : Int32
        getter max_seq : Int32

        def initialize(@kv_dim : Int32, @max_seq : Int32, h16_cache : Bool = false)
          @k_cache_buf = ML::MetalBuffer.from_array(Array(Float32).new(max_seq * kv_dim, 0.0_f32))
          @v_cache_buf = ML::MetalBuffer.from_array(Array(Float32).new(max_seq * kv_dim, 0.0_f32))
          if h16_cache
            @k_cache_h16_buf = ML::MetalBuffer.new(max_seq.to_i64 * kv_dim * 2_i64)
            @v_cache_h16_buf = ML::MetalBuffer.new(max_seq.to_i64 * kv_dim * 2_i64)
          end
        end

        def k_cache : Array(Float32)
          @k_cache_buf.read(@max_seq * @kv_dim)
        end

        def v_cache : Array(Float32)
          @v_cache_buf.read(@max_seq * @kv_dim)
        end
      end

      class ResidentScratch
        @buffers = {} of String => ML::MetalBuffer
        @f32_source_keys = {} of String => UInt64

        def get(name : String, byte_size : Int64) : ML::MetalBuffer
          if buf = @buffers[name]?
            return buf if buf.size >= byte_size
          end
          buf = ML::MetalBuffer.new(byte_size)
          @buffers[name] = buf
          buf
        end

        def f32_source_cached?(name : String, data : Array(Float32)) : Bool
          key = data.to_unsafe.address.to_u64 &+ (data.size.to_u64 << 1)
          if @f32_source_keys[name]? == key
            true
          else
            @f32_source_keys[name] = key
            false
          end
        end
      end

      class ResidentState
        getter layers : Array(ResidentLayerState)
        getter max_seq : Int32
        getter scratch : ResidentScratch

        def initialize(hp : Gemma4Hparams, @max_seq : Int32 = 1024)
          h16_cache = ENV["GEMMA4_ROW_PREFILL_ATTN_KV_H16_CACHE"]? == "1" &&
            ENV["GEMMA4_ROW_PREFILL_ATTN_KV_H16_CACHE_OFF"]? != "1"
          @layers = Array(ResidentLayerState).new(hp.n_layer) do |il|
            kv_dim = hp.n_head_kv(il) * hp.head_dim_for_layer(il)
            ResidentLayerState.new(kv_dim, @max_seq, h16_cache)
          end
          @scratch = ResidentScratch.new
        end

        def initialize(kv_dims : Array(Int32), @max_seq : Int32, h16_cache : Bool = false)
          raise ArgumentError.new("Gemma4 resident KV dims must not be empty") if kv_dims.empty?

          @layers = kv_dims.map do |kv_dim|
            raise ArgumentError.new("Gemma4 resident KV dim must be positive") unless kv_dim > 0

            ResidentLayerState.new(kv_dim, @max_seq, h16_cache)
          end
          @scratch = ResidentScratch.new
        end
      end

      private def write_scratch_f32(scratch : ResidentScratch, name : String, data : Array(Float32)) : ML::MetalBuffer
        buf = scratch.get(name, data.size.to_i64 * sizeof(Float32))
        if ENV["GEMMA4_SCRATCH_F32_STATIC_CACHE_OFF"]? == "1" || !scratch.f32_source_cached?(name, data)
          buf.write(data)
        end
        buf
      end


    {% end %}
  end
end
