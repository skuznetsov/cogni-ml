require "./qwen35_metal"
require "./qwen_qbit_adaptive_kv"
require "../core/buffer"

{% unless flag?(:cpu_only) %}
  require "../metal/device"
  require "../metal/dispatch"
{% end %}

module ML::GGUF
  # Default-off experimental owner for adaptive row-aligned QBit KV. The Metal
  # kernel reads the p4 base plus optional p5/BF16/F32 sidecars directly.
  module QwenQBitAdaptiveResidentKV
    extend self

    class Cache
      getter cache_len : Int32
      getter n_head_kv : Int32
      getter head_dim : Int32
      getter compressed_bytes : Int64

      @lifecycle_mutex : Mutex
      @released : Bool

      def initialize(@k_base : ML::MetalBuffer,
                     @k_metadata : ML::MetalBuffer,
                     @k_sidecar : ML::MetalBuffer,
                     @v_base : ML::MetalBuffer,
                     @v_metadata : ML::MetalBuffer,
                     @v_sidecar : ML::MetalBuffer,
                     @cache_len : Int32,
                     @n_head_kv : Int32,
                     @head_dim : Int32,
                     @compressed_bytes : Int64)
        @lifecycle_mutex = Mutex.new
        @released = false
      end

      def release : Nil
        @lifecycle_mutex.synchronize do
          return if @released
          @released = true
          buffers.each(&.release)
        end
      end

      # Metal command completion is synchronous, so retaining this lock keeps
      # all six representation buffers alive for the complete dispatch.
      def with_live_buffers(&)
        @lifecycle_mutex.synchronize do
          raise ArgumentError.new("adaptive resident QBit cache has been released") if @released
          yield @k_base, @k_metadata, @k_sidecar,
            @v_base, @v_metadata, @v_sidecar
        end
      end

      private def buffers : Array(ML::MetalBuffer)
        [@k_base, @k_metadata, @k_sidecar,
         @v_base, @v_metadata, @v_sidecar]
      end
    end

    {% unless flag?(:cpu_only) %}
      SOURCE = {{ read_file("#{__DIR__}/kernels/qbit_adaptive_attn_decode_qwen35.metal") }}
      @@gqa6_pipeline : ML::Metal::ComputePipeline?
      @@gqa6_pipeline_mutex = Mutex.new
    {% end %}

    def prepare(k : QwenQBitAdaptiveKV::Encoded,
                v : QwenQBitAdaptiveKV::Encoded,
                cache_len : Int32,
                n_head_kv : Int32,
                head_dim : Int32) : Cache
      k_regions = QwenQBitAdaptiveKV.regions(k)
      v_regions = QwenQBitAdaptiveKV.regions(v)
      validate_shape(k, v, cache_len, n_head_kv, head_dim)

      {% if flag?(:cpu_only) %}
        raise "Metal disabled (cpu_only)"
      {% else %}
        raise "Metal not available" unless Qwen35Metal.available?
        payloads = [
          k_regions.base, k_regions.metadata, k_regions.sidecar,
          v_regions.base, v_regions.metadata, v_regions.sidecar,
        ]
        buffers = [] of ML::MetalBuffer
        begin
          payloads.each { |payload| buffers << upload(payload) }
          Cache.new(
            buffers[0], buffers[1], buffers[2],
            buffers[3], buffers[4], buffers[5],
            cache_len, n_head_kv, head_dim,
            k.payload_bytes.to_i64 + v.payload_bytes.to_i64,
          )
        rescue ex
          buffers.each(&.release)
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
        cache.with_live_buffers do |k_base, k_metadata, k_sidecar, v_base, v_metadata, v_sidecar|
          transient = [] of ML::MetalBuffer
          begin
            q_buffer = ML::MetalBuffer.from_array(q)
            transient << q_buffer
            gate_buffer = ML::MetalBuffer.from_array(gate)
            transient << gate_buffer
            out_buffer = ML::MetalBuffer.new(q.size.to_i64 * sizeof(Float32))
            transient << out_buffer

            command = ML::Metal::CommandBuffer.new
            encoder = ML::Metal::ComputeEncoder.new(command)
            encoder.set_pipeline(gqa6_pipeline)
            encoder.set_buffer(q_buffer, 0)
            encoder.set_buffer(gate_buffer, 1)
            encoder.set_buffer(k_base, 2)
            encoder.set_buffer(k_metadata, 3)
            encoder.set_buffer(k_sidecar, 4)
            encoder.set_buffer(v_base, 5)
            encoder.set_buffer(v_metadata, 6)
            encoder.set_buffer(v_sidecar, 7)
            encoder.set_buffer(out_buffer, 8, ML::Metal::BufferAccess::Write)
            encoder.set_value(cache.cache_len.to_u32, 9)
            encoder.set_value(n_head.to_u32, 10)
            encoder.set_value(cache.n_head_kv.to_u32, 11)
            encoder.set_value(cache.head_dim.to_u32, 12)
            encoder.set_value(heads_per_group.to_u32, 13)
            encoder.set_value(scale, 14)
            encoder.dispatch_threadgroups({cache.n_head_kv, 1, 1}, {192, 1, 1})
            encoder.end_encoding
            command.commit_and_wait
            out_buffer.read(q.size.to_i32)
          ensure
            transient.each(&.release)
          end
        end
      {% end %}
    end

    private def validate_shape(k : QwenQBitAdaptiveKV::Encoded,
                               v : QwenQBitAdaptiveKV::Encoded,
                               cache_len : Int32,
                               n_head_kv : Int32,
                               head_dim : Int32) : Nil
      raise ArgumentError.new("adaptive resident QBit cache length must be positive") unless cache_len > 0
      raise ArgumentError.new("adaptive resident QBit KV head count must be positive") unless n_head_kv > 0
      unless head_dim == QwenQBitAdaptiveKV::ROW_VALUES
        raise ArgumentError.new("adaptive resident QBit requires Qwen3.8 head dimension 256")
      end
      unless k.block_size == head_dim && v.block_size == head_dim
        raise ArgumentError.new("adaptive resident QBit block size must equal one KV head dimension")
      end
      expected = cache_len.to_i64 * n_head_kv * head_dim
      unless k.value_count == expected && v.value_count == expected
        raise ArgumentError.new("adaptive resident QBit K/V value count does not match the declared shape")
      end
    end

    private def validate_attention(q : Array(Float32),
                                   gate : Array(Float32),
                                   cache : Cache,
                                   n_head : Int32,
                                   heads_per_group : Int32,
                                   scale : Float32) : Nil
      raise ArgumentError.new("adaptive resident QBit query head count must be positive") unless n_head > 0
      unless heads_per_group == 6 && n_head == cache.n_head_kv * heads_per_group
        raise ArgumentError.new("adaptive resident QBit requires Qwen3.8 GQA6 shape")
      end
      expected = n_head * cache.head_dim
      unless q.size == expected && gate.size == expected
        raise ArgumentError.new("adaptive resident QBit query/gate shape mismatch")
      end
      raise ArgumentError.new("adaptive resident QBit attention scale must be finite") unless scale.finite?
    end

    {% unless flag?(:cpu_only) %}
      private def upload(payload : Bytes) : ML::MetalBuffer
        # Metal rejects zero-byte allocations. A one-byte sentinel is bound for
        # an empty sidecar but is never read because validation admits only p4.
        buffer = ML::MetalBuffer.new(Math.max(payload.size, 1).to_i64, ML::StorageMode::Shared)
        begin
          buffer.write_bytes(payload.to_unsafe, payload.size) unless payload.empty?
          buffer
        rescue ex
          buffer.release
          raise ex
        end
      end

      private def gqa6_pipeline : ML::Metal::ComputePipeline
        @@gqa6_pipeline_mutex.synchronize do
          @@gqa6_pipeline ||= ML::Metal::PipelineCache.get("qwen35_qbit_adaptive_attn_decode_gqa6") {
            ML::Metal::ComputePipeline.new("qwen35_qbit_adaptive_attn_decode_gqa6", SOURCE)
          }
        end
      end
    {% end %}
  end
end
