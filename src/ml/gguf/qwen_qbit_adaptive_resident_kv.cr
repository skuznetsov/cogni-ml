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
      getter max_seq : Int32
      getter n_head_kv : Int32
      getter head_dim : Int32
      getter compressed_bytes : Int64

      @lifecycle_mutex : Mutex
      @released : Bool
      @cache_len : Int32
      @k_plan : QwenQBitAdaptiveKV::Plan?
      @v_plan : QwenQBitAdaptiveKV::Plan?

      def initialize(@k_base : ML::MetalBuffer,
                     @k_metadata : ML::MetalBuffer,
                     @k_sidecar : ML::MetalBuffer,
                     @v_base : ML::MetalBuffer,
                     @v_metadata : ML::MetalBuffer,
                     @v_sidecar : ML::MetalBuffer,
                     @cache_len : Int32,
                     @max_seq : Int32,
                     @n_head_kv : Int32,
                     @head_dim : Int32,
                     @compressed_bytes : Int64,
                     @k_plan : QwenQBitAdaptiveKV::Plan? = nil,
                     @v_plan : QwenQBitAdaptiveKV::Plan? = nil)
        @lifecycle_mutex = Mutex.new
        @released = false
      end

      def cache_len : Int32
        @lifecycle_mutex.synchronize do
          ensure_live!
          @cache_len
        end
      end

      def live_compressed_bytes : Int64
        @lifecycle_mutex.synchronize do
          ensure_live!
          if (k_plan = @k_plan) && (v_plan = @v_plan)
            rows = @cache_len * @n_head_kv
            rows.to_i64 *
              (2 * QwenQBitAdaptiveKV::BASE_ROW_BYTES +
                2 * QwenQBitAdaptiveKV::METADATA_BYTES) +
              k_plan.prefix_sidecar_bytes(rows) +
              v_plan.prefix_sidecar_bytes(rows)
          else
            @compressed_bytes
          end
        end
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
          ensure_live!
          yield @k_base, @k_metadata, @k_sidecar,
            @v_base, @v_metadata, @v_sidecar, @cache_len
        end
      end

      # Append is serialized with attention and release. The visible length is
      # advanced only after the caller's pack command has completed cleanly.
      def with_append(token_count : Int32, &)
        @lifecycle_mutex.synchronize do
          ensure_live!
          k_plan = @k_plan
          v_plan = @v_plan
          unless k_plan && v_plan
            raise ArgumentError.new("adaptive resident QBit cache is not appendable")
          end
          unless token_count > 0
            raise ArgumentError.new("adaptive resident QBit append token count must be positive")
          end
          if @cache_len.to_i64 + token_count > @max_seq
            raise ArgumentError.new("adaptive resident QBit append exceeds cache capacity")
          end

          start_token = @cache_len
          yield @k_base, @k_metadata, @k_sidecar,
            @v_base, @v_metadata, @v_sidecar,
            k_plan, v_plan, start_token
          @cache_len += token_count
        end
      end

      def with_snapshot_buffers(&)
        @lifecycle_mutex.synchronize do
          ensure_live!
          k_plan = @k_plan
          v_plan = @v_plan
          unless k_plan && v_plan
            raise ArgumentError.new("adaptive resident QBit cache is not appendable")
          end
          yield @k_base, @k_sidecar, @v_base, @v_sidecar,
            k_plan, v_plan, @cache_len
        end
      end

      private def buffers : Array(ML::MetalBuffer)
        [@k_base, @k_metadata, @k_sidecar,
         @v_base, @v_metadata, @v_sidecar]
      end

      private def ensure_live! : Nil
        raise ArgumentError.new("adaptive resident QBit cache has been released") if @released
      end
    end

    {% unless flag?(:cpu_only) %}
      SOURCE = {{ read_file("#{__DIR__}/kernels/qbit_adaptive_attn_decode_qwen35.metal") }}
      PACK_SOURCE = {{ read_file("#{__DIR__}/kernels/qbit_adaptive_pack_qwen35.metal") }}
      @@gqa6_pipeline : ML::Metal::ComputePipeline?
      @@gqa6_pipeline_mutex = Mutex.new
      @@pack_pipeline : ML::Metal::ComputePipeline?
      @@pack_pipeline_mutex = Mutex.new
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
            cache_len, cache_len, n_head_kv, head_dim,
            k.payload_bytes.to_i64 + v.payload_bytes.to_i64,
          )
        rescue ex
          buffers.each(&.release)
          raise ex
        end
      {% end %}
    end

    # Allocate the complete compressed capacity from a deterministic tier plan.
    # No Float32 KV owner is retained by this cache.
    def allocate(k_plan : QwenQBitAdaptiveKV::Plan,
                 v_plan : QwenQBitAdaptiveKV::Plan,
                 max_seq : Int32,
                 n_head_kv : Int32,
                 head_dim : Int32) : Cache
      validate_plan_shape(k_plan, v_plan, max_seq, n_head_kv, head_dim)

      {% if flag?(:cpu_only) %}
        raise "Metal disabled (cpu_only)"
      {% else %}
        raise "Metal not available" unless Qwen35Metal.available?
        buffers = [] of ML::MetalBuffer
        begin
          buffers << allocate_region(k_plan.base_bytes)
          buffers << upload(k_plan.metadata)
          buffers << allocate_region(k_plan.sidecar_bytes)
          buffers << allocate_region(v_plan.base_bytes)
          buffers << upload(v_plan.metadata)
          buffers << allocate_region(v_plan.sidecar_bytes)
          Cache.new(
            buffers[0], buffers[1], buffers[2],
            buffers[3], buffers[4], buffers[5],
            0, max_seq, n_head_kv, head_dim,
            k_plan.payload_bytes.to_i64 + v_plan.payload_bytes.to_i64,
            k_plan, v_plan,
          )
        rescue ex
          buffers.each(&.release)
          raise ex
        end
      {% end %}
    end

    # Pack a contiguous source token range directly from temporary Float32
    # Metal buffers into the next rows of the resident adaptive cache. Callers
    # must keep both source buffers alive until this synchronous call returns.
    def append_from_metal(cache : Cache,
                          k_source : ML::MetalBuffer,
                          v_source : ML::MetalBuffer,
                          token_count : Int32,
                          source_token_offset : Int32 = 0) : Nil
      raise ArgumentError.new("adaptive resident QBit source token offset must be non-negative") if source_token_offset < 0
      raise ArgumentError.new("adaptive resident QBit append token count must be positive") unless token_count > 0
      source_rows = (source_token_offset.to_i64 + token_count) * cache.n_head_kv
      source_values = source_rows * cache.head_dim
      if source_values > UInt32::MAX
        raise ArgumentError.new("adaptive resident QBit source index exceeds the device kernel limit")
      end
      required_bytes = source_values * sizeof(Float32)
      unless k_source.valid? && v_source.valid?
        raise ArgumentError.new("adaptive resident QBit source buffer has been released")
      end
      if k_source.size < required_bytes || v_source.size < required_bytes
        raise ArgumentError.new("adaptive resident QBit source buffer is too small")
      end

      {% if flag?(:cpu_only) %}
        raise "Metal disabled (cpu_only)"
      {% else %}
        raise "Metal not available" unless Qwen35Metal.available?
        cache.with_append(token_count) do |k_base, k_metadata, k_sidecar, v_base, v_metadata, v_sidecar, _k_plan, _v_plan, start_token|
          status = upload(Bytes.new(sizeof(UInt32), 0_u8))
          begin
            source_row_offset = checked_u32(source_token_offset.to_i64 * cache.n_head_kv, "source row offset")
            destination_row_offset = checked_u32(start_token.to_i64 * cache.n_head_kv, "destination row offset")
            row_count = checked_i32(token_count.to_i64 * cache.n_head_kv, "row count")
            command = ML::Metal::CommandBuffer.new
            encode_pack(command, k_source, k_base, k_metadata, k_sidecar, status,
              source_row_offset, destination_row_offset, row_count)
            encode_pack(command, v_source, v_base, v_metadata, v_sidecar, status,
              source_row_offset, destination_row_offset, row_count)
            command.commit_and_wait
            status_code = read_u32(status)
            unless status_code == 0
              raise ArgumentError.new("adaptive resident QBit device pack failed closed (status=#{status_code})")
            end
          ensure
            status.release
          end
        end
      {% end %}
      nil
    end

    # Read back only the live compressed prefix for validation or durable
    # snapshotting. Attention never calls this path.
    def snapshot(cache : Cache) : {QwenQBitAdaptiveKV::Encoded, QwenQBitAdaptiveKV::Encoded}
      {% if flag?(:cpu_only) %}
        raise "Metal disabled (cpu_only)"
      {% else %}
        result = nil
        cache.with_snapshot_buffers do |k_base, k_sidecar, v_base, v_sidecar, k_plan, v_plan, cache_len|
          rows = cache_len * cache.n_head_kv
          k_base_bytes = read_bytes(k_base, rows * QwenQBitAdaptiveKV::BASE_ROW_BYTES)
          v_base_bytes = read_bytes(v_base, rows * QwenQBitAdaptiveKV::BASE_ROW_BYTES)
          k_sidecar_bytes = read_bytes(k_sidecar, k_plan.prefix_sidecar_bytes(rows))
          v_sidecar_bytes = read_bytes(v_sidecar, v_plan.prefix_sidecar_bytes(rows))
          result = {
            QwenQBitAdaptiveKV.encoded_from_regions(k_plan, rows, k_base_bytes, k_sidecar_bytes),
            QwenQBitAdaptiveKV.encoded_from_regions(v_plan, rows, v_base_bytes, v_sidecar_bytes),
          }
        end
        result.not_nil!
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
        cache.with_live_buffers do |k_base, k_metadata, k_sidecar, v_base, v_metadata, v_sidecar, cache_len|
          raise ArgumentError.new("adaptive resident QBit cache is empty") unless cache_len > 0
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
            encoder.set_value(cache_len.to_u32, 9)
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

    private def validate_plan_shape(k_plan : QwenQBitAdaptiveKV::Plan,
                                    v_plan : QwenQBitAdaptiveKV::Plan,
                                    max_seq : Int32,
                                    n_head_kv : Int32,
                                    head_dim : Int32) : Nil
      raise ArgumentError.new("adaptive resident QBit maximum sequence must be positive") unless max_seq > 0
      raise ArgumentError.new("adaptive resident QBit KV head count must be positive") unless n_head_kv > 0
      unless head_dim == QwenQBitAdaptiveKV::ROW_VALUES
        raise ArgumentError.new("adaptive resident QBit requires Qwen3.8 head dimension 256")
      end
      expected_rows = max_seq.to_i64 * n_head_kv
      unless k_plan.row_count == expected_rows && v_plan.row_count == expected_rows
        raise ArgumentError.new("adaptive resident QBit plan row count does not match the declared capacity")
      end
    end

    private def validate_attention(q : Array(Float32),
                                   gate : Array(Float32),
                                   cache : Cache,
                                   n_head : Int32,
                                   heads_per_group : Int32,
                                   scale : Float32) : Nil
      raise ArgumentError.new("adaptive resident QBit query head count must be positive") unless n_head > 0
      expected_heads = cache.n_head_kv.to_i64 * heads_per_group
      unless heads_per_group == 6 && n_head.to_i64 == expected_heads
        raise ArgumentError.new("adaptive resident QBit requires Qwen3.8 GQA6 shape")
      end
      expected_values = n_head.to_i64 * cache.head_dim
      unless q.size.to_i64 == expected_values && gate.size.to_i64 == expected_values
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

      private def allocate_region(bytes : Int32) : ML::MetalBuffer
        ML::MetalBuffer.new(Math.max(bytes, 1).to_i64, ML::StorageMode::Shared)
      end

      private def encode_pack(command : ML::Metal::CommandBuffer,
                              source : ML::MetalBuffer,
                              base : ML::MetalBuffer,
                              metadata : ML::MetalBuffer,
                              sidecar : ML::MetalBuffer,
                              status : ML::MetalBuffer,
                              source_row_offset : UInt32,
                              destination_row_offset : UInt32,
                              row_count : Int32) : Nil
        encoder = ML::Metal::ComputeEncoder.new(command)
        encoder.set_pipeline(pack_pipeline)
        encoder.set_buffer(source, 0)
        encoder.set_buffer(base, 1, ML::Metal::BufferAccess::Write)
        encoder.set_buffer(metadata, 2)
        encoder.set_buffer(sidecar, 3, ML::Metal::BufferAccess::Write)
        encoder.set_buffer(status, 4, ML::Metal::BufferAccess::Write)
        encoder.set_value(source_row_offset, 5)
        encoder.set_value(destination_row_offset, 6)
        encoder.set_value(row_count.to_u32, 7)
        encoder.dispatch_threadgroups({row_count, 1, 1}, {32, 1, 1})
        encoder.end_encoding
      end

      private def checked_u32(value : Int64, label : String) : UInt32
        unless value >= 0 && value <= UInt32::MAX
          raise ArgumentError.new("adaptive resident QBit #{label} exceeds the device kernel limit")
        end
        value.to_u32
      end

      private def checked_i32(value : Int64, label : String) : Int32
        unless value >= 0 && value <= Int32::MAX
          raise ArgumentError.new("adaptive resident QBit #{label} exceeds the device dispatch limit")
        end
        value.to_i32
      end

      private def read_bytes(buffer : ML::MetalBuffer, byte_count : Int32) : Bytes
        return Bytes.new(0) if byte_count == 0
        bytes = Bytes.new(byte_count)
        buffer.read_bytes(bytes.to_unsafe, byte_count)
        bytes
      end

      private def read_u32(buffer : ML::MetalBuffer) : UInt32
        bytes = read_bytes(buffer, sizeof(UInt32))
        bytes[0].to_u32 |
          (bytes[1].to_u32 << 8) |
          (bytes[2].to_u32 << 16) |
          (bytes[3].to_u32 << 24)
      end

      private def gqa6_pipeline : ML::Metal::ComputePipeline
        @@gqa6_pipeline_mutex.synchronize do
          @@gqa6_pipeline ||= ML::Metal::PipelineCache.get("qwen35_qbit_adaptive_attn_decode_gqa6") {
            ML::Metal::ComputePipeline.new("qwen35_qbit_adaptive_attn_decode_gqa6", SOURCE)
          }
        end
      end

      private def pack_pipeline : ML::Metal::ComputePipeline
        @@pack_pipeline_mutex.synchronize do
          @@pack_pipeline ||= ML::Metal::PipelineCache.get("qwen35_qbit_adaptive_pack_row") {
            ML::Metal::ComputePipeline.new("qwen35_qbit_adaptive_pack_row", PACK_SOURCE)
          }
        end
      end
    {% end %}
  end
end
