require "./qwen35_metal"
require "./qwen_qbit_gaussian_codec"
require "../core/buffer"

{% unless flag?(:cpu_only) %}
  require "../metal/device"
  require "../metal/dispatch"
{% end %}

module ML::GGUF
  # Default-off experimental owner for p4/p5 KV that stays compressed in
  # Metal-visible memory and is decoded inside the attention kernel.
  module QwenQBitResidentKV
    extend self

    class Cache
      getter cache_len : Int32
      getter n_head_kv : Int32
      getter head_dim : Int32
      getter precision : Int32

      @lifecycle_mutex : Mutex
      @released : Bool

      def initialize(@k_buffer : ML::MetalBuffer,
                     @v_buffer : ML::MetalBuffer,
                     @cache_len : Int32,
                     @n_head_kv : Int32,
                     @head_dim : Int32,
                     @precision : Int32)
        @lifecycle_mutex = Mutex.new
        @released = false
      end

      def compressed_bytes : Int64
        @k_buffer.size + @v_buffer.size
      end

      def release : Nil
        @lifecycle_mutex.synchronize do
          return if @released
          @released = true
          @k_buffer.release
          @v_buffer.release
        end
      end

      # Keeps the compressed buffers owned until the synchronous command using
      # them has completed, so release cannot invalidate an in-flight dispatch.
      def with_live_buffers(&)
        @lifecycle_mutex.synchronize do
          raise ArgumentError.new("resident QBit cache has been released") if @released
          yield @k_buffer, @v_buffer
        end
      end
    end

    {% unless flag?(:cpu_only) %}
      SOURCE = {{ read_file("#{__DIR__}/kernels/qbit_attn_decode_qwen35.metal") }}
      @@gqa6_pipeline : ML::Metal::ComputePipeline?
      @@gqa6_pipeline_mutex = Mutex.new
    {% end %}

    def prepare(k : QwenQBitGaussianCodec::Encoded,
                v : QwenQBitGaussianCodec::Encoded,
                cache_len : Int32,
                n_head_kv : Int32,
                head_dim : Int32) : Cache
      validate_layout(k, v, cache_len, n_head_kv, head_dim)

      {% if flag?(:cpu_only) %}
        raise "Metal disabled (cpu_only)"
      {% else %}
        raise "Metal not available" unless Qwen35Metal.available?
        k_buffer = ML::MetalBuffer.new(k.payload.size.to_i64, ML::StorageMode::Shared)
        begin
          v_buffer = ML::MetalBuffer.new(v.payload.size.to_i64, ML::StorageMode::Shared)
          begin
            k_buffer.write_bytes(k.payload.to_unsafe, k.payload.size)
            v_buffer.write_bytes(v.payload.to_unsafe, v.payload.size)
            Cache.new(k_buffer, v_buffer, cache_len, n_head_kv, head_dim, k.precision)
          rescue ex
            v_buffer.release
            raise ex
          end
        rescue ex
          k_buffer.release
          raise ex
        end
      {% end %}
    end

    def attn_decode(q : Array(Float32),
                    gate : Array(Float32),
                    cache : Cache,
                    n_head : Int32,
                    heads_per_group : Int32,
                    scale : Float32) : Array(Float32)
      validate_attention(q, gate, cache, n_head, heads_per_group, scale)

      {% if flag?(:cpu_only) %}
        raise "Metal disabled (cpu_only)"
      {% else %}
        raise "Metal not available" unless Qwen35Metal.available?
        cache.with_live_buffers do |k_buffer, v_buffer|
          buffers = [] of ML::MetalBuffer
          begin
            q_buffer = ML::MetalBuffer.from_array(q)
            buffers << q_buffer
            gate_buffer = ML::MetalBuffer.from_array(gate)
            buffers << gate_buffer
            out_buffer = ML::MetalBuffer.new(q.size.to_i64 * sizeof(Float32))
            buffers << out_buffer
            command = ML::Metal::CommandBuffer.new
            encoder = ML::Metal::ComputeEncoder.new(command)
            encoder.set_pipeline(gqa6_pipeline)
            encoder.set_buffer(q_buffer, 0)
            encoder.set_buffer(gate_buffer, 1)
            encoder.set_buffer(k_buffer, 2)
            encoder.set_buffer(v_buffer, 3)
            encoder.set_buffer(out_buffer, 4, ML::Metal::BufferAccess::Write)
            encoder.set_value(cache.cache_len.to_u32, 5)
            encoder.set_value(n_head.to_u32, 6)
            encoder.set_value(cache.n_head_kv.to_u32, 7)
            encoder.set_value(cache.head_dim.to_u32, 8)
            encoder.set_value(heads_per_group.to_u32, 9)
            encoder.set_value(cache.precision.to_u32, 10)
            encoder.set_value(scale, 11)
            encoder.dispatch_threadgroups({cache.n_head_kv, 1, 1}, {192, 1, 1})
            encoder.end_encoding
            command.commit_and_wait
            out_buffer.read(q.size.to_i32)
          ensure
            buffers.each(&.release)
          end
        end
      {% end %}
    end

    private def validate_layout(k : QwenQBitGaussianCodec::Encoded,
                                v : QwenQBitGaussianCodec::Encoded,
                                cache_len : Int32,
                                n_head_kv : Int32,
                                head_dim : Int32) : Nil
      QwenQBitGaussianCodec.validate(k)
      QwenQBitGaussianCodec.validate(v)
      raise ArgumentError.new("resident QBit cache length must be positive") unless cache_len > 0
      raise ArgumentError.new("resident QBit KV head count must be positive") unless n_head_kv > 0
      unless head_dim == 256
        raise ArgumentError.new("resident QBit probe requires Qwen3.8 head dimension 256")
      end
      unless k.precision == v.precision && (k.precision == 4 || k.precision == 5)
        raise ArgumentError.new("resident QBit K/V precision must match and be p4 or p5")
      end
      unless k.block_size == head_dim && v.block_size == head_dim
        raise ArgumentError.new("resident QBit block size must equal one KV head dimension")
      end
      expected = cache_len.to_i64 * n_head_kv * head_dim
      unless k.value_count == expected && v.value_count == expected
        raise ArgumentError.new("resident QBit K/V value count does not match the declared shape")
      end
    end

    private def validate_attention(q : Array(Float32),
                                   gate : Array(Float32),
                                   cache : Cache,
                                   n_head : Int32,
                                   heads_per_group : Int32,
                                   scale : Float32) : Nil
      raise ArgumentError.new("resident QBit query head count must be positive") unless n_head > 0
      unless heads_per_group == 6 && n_head == cache.n_head_kv * heads_per_group
        raise ArgumentError.new("resident QBit probe requires Qwen3.8 GQA6 shape")
      end
      expected = n_head * cache.head_dim
      unless q.size == expected && gate.size == expected
        raise ArgumentError.new("resident QBit query/gate shape mismatch")
      end
      raise ArgumentError.new("resident QBit attention scale must be finite") unless scale.finite?
    end

    {% unless flag?(:cpu_only) %}
      private def gqa6_pipeline : ML::Metal::ComputePipeline
        @@gqa6_pipeline_mutex.synchronize do
          @@gqa6_pipeline ||= ML::Metal::PipelineCache.get("qwen35_qbit_attn_decode_gqa6") {
            ML::Metal::ComputePipeline.new("qwen35_qbit_attn_decode_gqa6", SOURCE)
          }
        end
      end
    {% end %}
  end
end
