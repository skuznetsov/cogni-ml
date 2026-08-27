require "./qwen35_metal"
require "./qwen_qbit_adaptive_kv"
require "./qwen_qbit_adaptive_metal_policy"
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

    {% if flag?(:cpu_only) %}
      # Keep the resident cache type available to shared state declarations
      # without importing Metal into CPU-only builds. No CPU path constructs or
      # submits this command type.
      private class DisabledCommandBuffer
        def committed? : Bool
          false
        end

        def completed? : Bool
          false
        end

        def completed_successfully? : Bool
          false
        end
      end

      alias ResidentCommandBuffer = DisabledCommandBuffer
    {% else %}
      alias ResidentCommandBuffer = ML::Metal::CommandBuffer
    {% end %}

    DEVICE_SUCCESS                 = 0xa17ecafe_u32
    PREFILL_ATTENTION_CHUNK_TOKENS =             64

    # Only the validated module factories can construct a resident cache. The
    # private admission type prevents callers from bypassing buffer/shape
    # validation and binding arbitrary regions to unchecked Metal kernels.
    private record CacheAdmission,
      k_base : ML::MetalBuffer,
      k_metadata : ML::MetalBuffer,
      k_sidecar : ML::MetalBuffer,
      v_base : ML::MetalBuffer,
      v_metadata : ML::MetalBuffer,
      v_sidecar : ML::MetalBuffer,
      cache_len : Int32,
      max_seq : Int32,
      n_head_kv : Int32,
      head_dim : Int32,
      compressed_bytes : Int64,
      k_plan : QwenQBitAdaptiveKV::Plan?,
      v_plan : QwenQBitAdaptiveKV::Plan?

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
      @pending_status : ML::MetalBuffer?
      @pending_command : ResidentCommandBuffer?
      @pending_token_count : Int32
      @pending_start_token : Int32

      def self.from_admission(admission : CacheAdmission) : self
        new(admission)
      end

      private def initialize(admission : CacheAdmission)
        @k_base = admission.k_base
        @k_metadata = admission.k_metadata
        @k_sidecar = admission.k_sidecar
        @v_base = admission.v_base
        @v_metadata = admission.v_metadata
        @v_sidecar = admission.v_sidecar
        @cache_len = admission.cache_len
        @max_seq = admission.max_seq
        @n_head_kv = admission.n_head_kv
        @head_dim = admission.head_dim
        @compressed_bytes = admission.compressed_bytes
        @k_plan = admission.k_plan
        @v_plan = admission.v_plan
        @lifecycle_mutex = Mutex.new
        @released = false
        @pending_status = nil
        @pending_command = nil
        @pending_token_count = 0
        @pending_start_token = 0
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
          ensure_no_pending!
          @released = true
          buffers.each(&.release)
        end
      end

      # Metal command completion is synchronous, so retaining this lock keeps
      # all six representation buffers alive for the complete dispatch.
      def with_live_buffers(&)
        @lifecycle_mutex.synchronize do
          ensure_live!
          ensure_no_pending!
          yield @k_base, @k_metadata, @k_sidecar,
            @v_base, @v_metadata, @v_sidecar, @cache_len
        end
      end

      # Append is serialized with attention and release. The visible length is
      # advanced only after the caller's pack command has completed cleanly.
      def with_append(token_count : Int32, &)
        @lifecycle_mutex.synchronize do
          ensure_live!
          ensure_no_pending!
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
          ensure_no_pending!
          k_plan = @k_plan
          v_plan = @v_plan
          unless k_plan && v_plan
            raise ArgumentError.new("adaptive resident QBit cache is not appendable")
          end
          yield @k_base, @k_sidecar, @v_base, @v_sidecar,
            k_plan, v_plan, @cache_len
        end
      end

      # Restore one validated compact live prefix into an empty planned cache.
      # Publication happens only after both K and V regions have been copied;
      # a failed copy therefore leaves cache_len at zero and invisible.
      def restore_snapshot!(k : QwenQBitAdaptiveKV::Encoded,
                            v : QwenQBitAdaptiveKV::Encoded,
                            cache_len : Int32) : Nil
        k_regions = QwenQBitAdaptiveKV.regions(k)
        v_regions = QwenQBitAdaptiveKV.regions(v)
        @lifecycle_mutex.synchronize do
          ensure_live!
          ensure_no_pending!
          unless @cache_len == 0
            raise ArgumentError.new("adaptive resident QBit snapshot target is not empty")
          end
          unless cache_len > 0 && cache_len <= @max_seq
            raise ArgumentError.new("adaptive resident QBit snapshot length is outside capacity")
          end
          expected_values = cache_len.to_i64 * @n_head_kv * @head_dim
          unless expected_values <= Int32::MAX &&
                 k.value_count == expected_values && v.value_count == expected_values
            raise ArgumentError.new("adaptive resident QBit snapshot shape mismatch")
          end
          k_plan = @k_plan
          v_plan = @v_plan
          unless k_plan && v_plan
            raise ArgumentError.new("adaptive resident QBit snapshot target is not appendable")
          end
          rows = cache_len * @n_head_kv
          expected_metadata_bytes = rows * QwenQBitAdaptiveKV::METADATA_BYTES
          unless k_regions.metadata == k_plan.metadata[0, expected_metadata_bytes] &&
                 v_regions.metadata == v_plan.metadata[0, expected_metadata_bytes]
            raise ArgumentError.new("adaptive resident QBit snapshot tier plan mismatch")
          end
          unless k_regions.base.size <= @k_base.size &&
                 k_regions.sidecar.size <= @k_sidecar.size &&
                 v_regions.base.size <= @v_base.size &&
                 v_regions.sidecar.size <= @v_sidecar.size
            raise ArgumentError.new("adaptive resident QBit snapshot exceeds target buffers")
          end

          @k_base.write_bytes(k_regions.base.to_unsafe, k_regions.base.size) unless k_regions.base.empty?
          @k_sidecar.write_bytes(k_regions.sidecar.to_unsafe, k_regions.sidecar.size) unless k_regions.sidecar.empty?
          @v_base.write_bytes(v_regions.base.to_unsafe, v_regions.base.size) unless v_regions.base.empty?
          @v_sidecar.write_bytes(v_regions.sidecar.to_unsafe, v_regions.sidecar.size) unless v_regions.sidecar.empty?
          @cache_len = cache_len
        end
      end

      # Internal encoder lease: unlike normal reads this is legal only while a
      # shared-command append reservation is active. Publication is still
      # controlled exclusively by `finish_pending_append!`.
      def with_pending_buffers(&)
        @lifecycle_mutex.synchronize do
          ensure_live!
          unless @pending_status
            raise ArgumentError.new("adaptive resident QBit append is not pending")
          end
          uniform_tier = if (k_plan = @k_plan) && (v_plan = @v_plan)
                           selected = k_plan.uniform_tier
                           selected if selected && v_plan.uniform_tier == selected
                         end
          yield @k_base, @k_metadata, @k_sidecar,
            @v_base, @v_metadata, @v_sidecar, uniform_tier
        end
      end

      # Reserve one append whose encoders are owned by an external command
      # buffer. The old prefix remains the only visible prefix until the caller
      # waits for that command and explicitly publishes the completion status.
      def begin_pending_append!(token_count : Int32,
                                expected_start_token : Int32,
                                status : ML::MetalBuffer,
                                command : ResidentCommandBuffer) : Int32
        @lifecycle_mutex.synchronize do
          ensure_live!
          ensure_no_pending!
          if command.committed?
            raise ArgumentError.new("adaptive resident QBit append requires an uncommitted command")
          end
          unless @k_plan && @v_plan
            raise ArgumentError.new("adaptive resident QBit cache is not appendable")
          end
          unless token_count > 0
            raise ArgumentError.new("adaptive resident QBit append token count must be positive")
          end
          unless expected_start_token == @cache_len
            raise ArgumentError.new("adaptive resident QBit append start does not match the live prefix")
          end
          if @cache_len.to_i64 + token_count > @max_seq
            raise ArgumentError.new("adaptive resident QBit append exceeds cache capacity")
          end
          unless status.valid? && status.size >= sizeof(UInt32)
            raise ArgumentError.new("adaptive resident QBit pending status buffer is invalid")
          end

          @pending_status = status
          @pending_command = command
          @pending_token_count = token_count
          @pending_start_token = @cache_len
          @pending_start_token
        end
      end

      # Publish only after the caller has committed and waited for the external
      # command. A failed/non-executed marker clears the reservation without
      # advancing the visible prefix, so a clean retry remains possible.
      def finish_pending_append!(command : ResidentCommandBuffer) : Nil
        @lifecycle_mutex.synchronize do
          ensure_live!
          status = @pending_status
          raise ArgumentError.new("adaptive resident QBit append is not pending") unless status
          ensure_pending_command!(command)
          unless command.completed_successfully?
            raise ArgumentError.new("adaptive resident QBit publication requires its command to complete successfully")
          end

          status_code = status.contents.as(Pointer(UInt32)).value
          token_count = @pending_token_count
          clear_pending!
          status.release
          unless status_code == DEVICE_SUCCESS
            raise ArgumentError.new("adaptive resident QBit prefill/pack failed closed (status=#{status_code})")
          end
          @cache_len += token_count
        end
      end

      # Non-mutating half of group publication. Callers with several caches on
      # one command validate every tail marker before advancing any live prefix.
      def validate_pending_append!(command : ResidentCommandBuffer) : Nil
        @lifecycle_mutex.synchronize do
          ensure_live!
          status = @pending_status
          raise ArgumentError.new("adaptive resident QBit append is not pending") unless status
          ensure_pending_command!(command)
          unless command.completed_successfully?
            raise ArgumentError.new("adaptive resident QBit publication requires its command to complete successfully")
          end
          status_code = status.contents.as(Pointer(UInt32)).value
          unless status_code == DEVICE_SUCCESS
            raise ArgumentError.new("adaptive resident QBit prefill/pack failed closed (status=#{status_code})")
          end
        end
      end

      # Cancellation may reopen the reserved destination rows only while the
      # bound command is still uncommitted or after it has completed. An
      # in-flight/unknown command deliberately leaves the cache pending.
      def cancel_pending_append!(command : ResidentCommandBuffer) : Nil
        @lifecycle_mutex.synchronize do
          return unless status = @pending_status
          ensure_pending_command!(command)
          if command.committed? && !command.completed?
            raise ArgumentError.new("adaptive resident QBit command is not completed; cancellation is unsafe")
          end
          clear_pending!
          status.release
        end
      end

      def with_pending_status(command : ResidentCommandBuffer, &)
        @lifecycle_mutex.synchronize do
          ensure_live!
          status = @pending_status
          raise ArgumentError.new("adaptive resident QBit append is not pending") unless status
          ensure_pending_command!(command)
          if command.committed?
            raise ArgumentError.new("adaptive resident QBit finalizer requires an uncommitted command")
          end
          yield status
        end
      end

      private def buffers : Array(ML::MetalBuffer)
        [@k_base, @k_metadata, @k_sidecar,
         @v_base, @v_metadata, @v_sidecar]
      end

      private def ensure_live! : Nil
        raise ArgumentError.new("adaptive resident QBit cache has been released") if @released
      end

      private def ensure_no_pending! : Nil
        if @pending_status
          raise ArgumentError.new("adaptive resident QBit append is pending shared-command completion")
        end
      end

      private def clear_pending! : Nil
        @pending_status = nil
        @pending_command = nil
        @pending_token_count = 0
        @pending_start_token = 0
      end

      private def ensure_pending_command!(command : ResidentCommandBuffer) : Nil
        pending_command = @pending_command
        unless pending_command && pending_command.same?(command)
          raise ArgumentError.new("adaptive resident QBit command does not own the pending append")
        end
      end
    end

    {% unless flag?(:cpu_only) %}
      SOURCE = {{ read_file("#{__DIR__}/kernels/qbit_adaptive_attn_decode_qwen35.metal") }}
      SOURCE_TILE15 = SOURCE.sub(
        "constant uint QQA_ADAPTIVE_GQA6_TILE = 16;",
        "constant uint QQA_ADAPTIVE_GQA6_TILE = 15;",
      )
      raise "adaptive GQA6 tile-15 source patch no longer matches" if SOURCE_TILE15 == SOURCE
      PACK_SOURCE = {{ read_file("#{__DIR__}/kernels/qbit_adaptive_pack_qwen35.metal") }}
      @@gqa6_pipelines = Hash(Int32, ML::Metal::ComputePipeline).new
      @@gqa6_pipeline_mutex = Mutex.new
      @@prefill_gqa6_pipelines = Hash(Int32, ML::Metal::ComputePipeline).new
      @@prefill_gqa6_pipeline_mutex = Mutex.new
      @@pack_pipeline : ML::Metal::ComputePipeline?
      @@pack_pipeline_mutex = Mutex.new
      @@finalize_pipeline : ML::Metal::ComputePipeline?
      @@finalize_pipeline_mutex = Mutex.new
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
          Cache.from_admission(CacheAdmission.new(
            buffers[0], buffers[1], buffers[2],
            buffers[3], buffers[4], buffers[5],
            cache_len, cache_len, n_head_kv, head_dim,
            k.payload_bytes.to_i64 + v.payload_bytes.to_i64, nil, nil,
          ))
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
          Cache.from_admission(CacheAdmission.new(
            buffers[0], buffers[1], buffers[2],
            buffers[3], buffers[4], buffers[5],
            0, max_seq, n_head_kv, head_dim,
            k_plan.payload_bytes.to_i64 + v_plan.payload_bytes.to_i64,
            k_plan, v_plan,
          ))
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
            encode_finalize(command, status)
            command.commit_and_wait
            status_code = read_u32(status)
            unless status_code == DEVICE_SUCCESS
              raise ArgumentError.new("adaptive resident QBit device pack failed closed (status=#{status_code})")
            end
          ensure
            status.release
          end
        end
      {% end %}
      nil
    end

    # Compute causal attention over the already packed prefix plus an exact
    # temporary Float32 chunk, then pack that chunk into the resident cache.
    # All encoders share one command buffer and the new prefix becomes visible
    # only after attention and both K/V packers complete without an error.
    def prefill_chunk_and_append_from_metal(cache : Cache,
                                            q_source : ML::MetalBuffer,
                                            gate_source : ML::MetalBuffer,
                                            k_source : ML::MetalBuffer,
                                            v_source : ML::MetalBuffer,
                                            output : ML::MetalBuffer,
                                            token_count : Int32,
                                            n_head : Int32,
                                            heads_per_group : Int32,
                                            scale : Float32) : Nil
      validate_prefill_buffers(
        cache, q_source, gate_source, k_source, v_source, output,
        token_count, n_head, heads_per_group, scale,
      )

      {% if flag?(:cpu_only) %}
        raise "Metal disabled (cpu_only)"
      {% else %}
        raise "Metal not available" unless Qwen35Metal.available?
        command = ML::Metal::CommandBuffer.new
        encode_prefill_chunk_and_append(
          command, cache,
          q_source, gate_source, k_source, v_source, output,
          token_count, n_head, heads_per_group, scale,
          expected_start_token: cache.cache_len,
        )
        begin
          finalize_pending_append(command, cache)
          command.commit_and_wait
          finish_pending_append!(cache, command)
        rescue ex
          if !command.committed? || command.completed?
            cancel_pending_append!(cache, command)
          end
          raise ex
        end
      {% end %}
      nil
    end

    # Encode attention over packed history plus the exact current Float32
    # chunk, followed by K/V packing, into a caller-owned command buffer. The
    # cache remains unpublished and exclusively reserved until
    # `finish_pending_append!` observes the completion marker after a wait.
    def encode_prefill_chunk_and_append(command : ResidentCommandBuffer,
                                        cache : Cache,
                                        q_source : ML::MetalBuffer,
                                        gate_source : ML::MetalBuffer,
                                        k_source : ML::MetalBuffer,
                                        v_source : ML::MetalBuffer,
                                        output : ML::MetalBuffer,
                                        token_count : Int32,
                                        n_head : Int32,
                                        heads_per_group : Int32,
                                        scale : Float32,
                                        expected_start_token : Int32) : Nil
      validate_prefill_buffers(
        cache, q_source, gate_source, k_source, v_source, output,
        token_count, n_head, heads_per_group, scale,
      )

      {% if flag?(:cpu_only) %}
        raise "Metal disabled (cpu_only)"
      {% else %}
        raise "Metal not available" unless Qwen35Metal.available?
        if command.committed?
          raise ArgumentError.new("adaptive resident QBit append requires an uncommitted command")
        end
        status = upload(Bytes.new(sizeof(UInt32), 0_u8))
        reserved = false
        begin
          start_token = cache.begin_pending_append!(token_count, expected_start_token, status, command)
          reserved = true
          cache.with_pending_buffers do |k_base, k_metadata, k_sidecar, v_base, v_metadata, v_sidecar, uniform_tier|
            qualified_uniform_tier = qualified_uniform_prefill_tier(uniform_tier)
            source_token_offset = 0_i32
            prefill_attention_chunks(token_count).each do |chunk_tokens|
              packed_len = start_token + source_token_offset
              source_row_offset = checked_u32(
                source_token_offset.to_i64 * cache.n_head_kv,
                "source row offset",
              )
              destination_row_offset = checked_u32(
                packed_len.to_i64 * cache.n_head_kv,
                "destination row offset",
              )
              row_count = checked_i32(chunk_tokens.to_i64 * cache.n_head_kv, "row count")
              encode_prefill_chunk(
                command, q_source, gate_source, k_source, v_source,
                k_base, k_metadata, k_sidecar,
                v_base, v_metadata, v_sidecar,
                output, status, packed_len, chunk_tokens, source_token_offset,
                n_head, cache.n_head_kv, cache.head_dim,
                heads_per_group, scale, qualified_uniform_tier,
              )
              encode_pack(command, k_source, k_base, k_metadata, k_sidecar, status,
                source_row_offset, destination_row_offset, row_count)
              encode_pack(command, v_source, v_base, v_metadata, v_sidecar, status,
                source_row_offset, destination_row_offset, row_count)
              source_token_offset += chunk_tokens
            end
          end
        rescue ex
          if reserved
            cache.cancel_pending_append!(command) unless command.committed?
          else
            status.release
          end
          raise ex
        end
      {% end %}
      nil
    end

    # This must be the final encoder added to the caller-owned command. Its
    # non-zero marker certifies that all earlier attention, pack, and later
    # model encoders in that command reached the tail without a device error.
    def finalize_pending_append(command : ResidentCommandBuffer,
                                cache : Cache) : Nil
      if command.committed?
        raise ArgumentError.new("adaptive resident QBit finalizer requires an uncommitted command")
      end
      {% if flag?(:cpu_only) %}
        raise "Metal disabled (cpu_only)"
      {% else %}
        cache.with_pending_status(command) do |status|
          encode_finalize(command, status)
        end
      {% end %}
      nil
    end

    def finish_pending_append!(cache : Cache,
                               command : ResidentCommandBuffer) : Nil
      cache.finish_pending_append!(command)
    end

    # All caches share one completed command. Validate the complete set first,
    # then publish; a bad layer therefore cannot expose a partial cache prefix.
    def finish_pending_appends!(caches : Array(Cache),
                                command : ResidentCommandBuffer) : Nil
      caches.each { |cache| cache.validate_pending_append!(command) }
      caches.each { |cache| cache.finish_pending_append!(command) }
    end

    def cancel_pending_append!(cache : Cache,
                               command : ResidentCommandBuffer) : Nil
      cache.cancel_pending_append!(command)
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

    # Bounded durable-writeback helpers: materialize only one canonical K/V
    # payload so the caller can release it before requesting the next record.
    def snapshot_k(cache : Cache) : QwenQBitAdaptiveKV::Encoded
      snapshot_one(cache, true)
    end

    def snapshot_v(cache : Cache) : QwenQBitAdaptiveKV::Encoded
      snapshot_one(cache, false)
    end

    private def snapshot_one(cache : Cache, key : Bool) : QwenQBitAdaptiveKV::Encoded
      {% if flag?(:cpu_only) %}
        raise "Metal disabled (cpu_only)"
      {% else %}
        result = nil
        cache.with_snapshot_buffers do |k_base, k_sidecar, v_base, v_sidecar, k_plan, v_plan, cache_len|
          rows = cache_len * cache.n_head_kv
          if key
            base = read_bytes(k_base, rows * QwenQBitAdaptiveKV::BASE_ROW_BYTES)
            sidecar = read_bytes(k_sidecar, k_plan.prefix_sidecar_bytes(rows))
            result = QwenQBitAdaptiveKV.encoded_from_regions(k_plan, rows, base, sidecar)
          else
            base = read_bytes(v_base, rows * QwenQBitAdaptiveKV::BASE_ROW_BYTES)
            sidecar = read_bytes(v_sidecar, v_plan.prefix_sidecar_bytes(rows))
            result = QwenQBitAdaptiveKV.encoded_from_regions(v_plan, rows, base, sidecar)
          end
        end
        result.not_nil!
      {% end %}
    end

    def restore_snapshot!(cache : Cache,
                          k : QwenQBitAdaptiveKV::Encoded,
                          v : QwenQBitAdaptiveKV::Encoded,
                          cache_len : Int32) : Nil
      cache.restore_snapshot!(k, v, cache_len)
    end

    def attn_decode(q : Array(Float32),
                    gate : Array(Float32),
                    cache : Cache,
                    n_head : Int32,
                    heads_per_group : Int32,
                    scale : Float32,
                    gpu_elapsed_seconds : Pointer(Float64) = Pointer(Float64).null) : Array(Float32)
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
            if gpu_elapsed_seconds.null?
              command.commit_and_wait
            else
              gpu_elapsed_seconds.value = command.commit_and_wait_gpu_elapsed_seconds
            end
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

    private def validate_prefill_buffers(cache : Cache,
                                         q_source : ML::MetalBuffer,
                                         gate_source : ML::MetalBuffer,
                                         k_source : ML::MetalBuffer,
                                         v_source : ML::MetalBuffer,
                                         output : ML::MetalBuffer,
                                         token_count : Int32,
                                         n_head : Int32,
                                         heads_per_group : Int32,
                                         scale : Float32) : Nil
      raise ArgumentError.new("adaptive resident QBit prefill token count must be positive") unless token_count > 0
      raise ArgumentError.new("adaptive resident QBit query head count must be positive") unless n_head > 0
      expected_heads = cache.n_head_kv.to_i64 * heads_per_group
      unless heads_per_group == 6 && n_head.to_i64 == expected_heads
        raise ArgumentError.new("adaptive resident QBit prefill requires Qwen3.8 GQA6 shape")
      end
      raise ArgumentError.new("adaptive resident QBit attention scale must be finite") unless scale.finite?

      buffers = [q_source, gate_source, k_source, v_source, output]
      unless buffers.all?(&.valid?)
        raise ArgumentError.new("adaptive resident QBit prefill buffer has been released")
      end
      q_values = token_count.to_i64 * n_head * cache.head_dim
      kv_values = token_count.to_i64 * cache.n_head_kv * cache.head_dim
      max_kv_values = cache.max_seq.to_i64 * cache.n_head_kv * cache.head_dim
      if q_values > UInt32::MAX || kv_values > UInt32::MAX || max_kv_values > UInt32::MAX
        raise ArgumentError.new("adaptive resident QBit prefill index exceeds the device kernel limit")
      end
      q_bytes = q_values * sizeof(Float32)
      kv_bytes = kv_values * sizeof(Float32)
      unless q_source.size >= q_bytes && gate_source.size >= q_bytes && output.size >= q_bytes
        raise ArgumentError.new("adaptive resident QBit prefill query/gate/output buffer is too small")
      end
      unless k_source.size >= kv_bytes && v_source.size >= kv_bytes
        raise ArgumentError.new("adaptive resident QBit prefill K/V buffer is too small")
      end
    end

    # Keep adaptive attention dispatches inside the qualified 64-token width
    # while the surrounding model prefill remains a single command. A
    # one-token remainder is split across the first 65 tokens so no dispatch
    # falls onto a one-token shape.
    private def prefill_attention_chunks(token_count : Int32) : Array(Int32)
      first = token_count % PREFILL_ATTENTION_CHUNK_TOKENS
      first = PREFILL_ATTENTION_CHUNK_TOKENS if first == 0
      chunks = if first == 1 && token_count > PREFILL_ATTENTION_CHUNK_TOKENS
                 left = (PREFILL_ATTENTION_CHUNK_TOKENS + 1) // 2
                 [left.to_i32, (PREFILL_ATTENTION_CHUNK_TOKENS + 1 - left).to_i32]
               else
                 [first.to_i32]
               end
      remaining = token_count - first
      remaining -= PREFILL_ATTENTION_CHUNK_TOKENS if chunks.size == 2
      while remaining > 0
        chunks << Math.min(remaining, PREFILL_ATTENTION_CHUNK_TOKENS).to_i32
        remaining -= PREFILL_ATTENTION_CHUNK_TOKENS
      end
      chunks
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

      private def encode_prefill_chunk(command : ML::Metal::CommandBuffer,
                                       q_source : ML::MetalBuffer,
                                       gate_source : ML::MetalBuffer,
                                       k_source : ML::MetalBuffer,
                                       v_source : ML::MetalBuffer,
                                       k_base : ML::MetalBuffer,
                                       k_metadata : ML::MetalBuffer,
                                       k_sidecar : ML::MetalBuffer,
                                       v_base : ML::MetalBuffer,
                                       v_metadata : ML::MetalBuffer,
                                       v_sidecar : ML::MetalBuffer,
                                       output : ML::MetalBuffer,
                                       status : ML::MetalBuffer,
                                       packed_len : Int32,
                                       token_count : Int32,
                                       source_token_offset : Int32,
                                       n_head : Int32,
                                       n_head_kv : Int32,
                                       head_dim : Int32,
                                       heads_per_group : Int32,
                                       scale : Float32,
                                       uniform_tier : QwenQBitAdaptiveKV::Tier?) : Nil
        encoder = ML::Metal::ComputeEncoder.new(command)
        encoder.set_pipeline(prefill_gqa6_pipeline)
        encoder.set_buffer(q_source, 0)
        encoder.set_buffer(gate_source, 1)
        encoder.set_buffer(k_source, 2)
        encoder.set_buffer(v_source, 3)
        encoder.set_buffer(k_base, 4)
        encoder.set_buffer(k_metadata, 5)
        encoder.set_buffer(k_sidecar, 6)
        encoder.set_buffer(v_base, 7)
        encoder.set_buffer(v_metadata, 8)
        encoder.set_buffer(v_sidecar, 9)
        encoder.set_buffer(output, 10, ML::Metal::BufferAccess::Write)
        encoder.set_buffer(status, 11, ML::Metal::BufferAccess::Write)
        encoder.set_value(packed_len.to_u32, 12)
        encoder.set_value(token_count.to_u32, 13)
        encoder.set_value(n_head.to_u32, 14)
        encoder.set_value(n_head_kv.to_u32, 15)
        encoder.set_value(head_dim.to_u32, 16)
        encoder.set_value(heads_per_group.to_u32, 17)
        encoder.set_value(scale, 18)
        encoder.set_value(source_token_offset.to_u32, 19)
        encoder.set_value(uniform_tier ? uniform_tier.value.to_u32 : UInt32::MAX, 20)
        encoder.dispatch_threadgroups({n_head_kv, token_count, 1}, {192, 1, 1})
        encoder.end_encoding
      end

      private def qualified_uniform_prefill_tier(tier : QwenQBitAdaptiveKV::Tier?) : QwenQBitAdaptiveKV::Tier?
        return nil if ENV["QWEN35_ADAPTIVE_UNIFORM_PREFILL_OFF"]? == "1"
        case tier
        when QwenQBitAdaptiveKV::Tier::P4, QwenQBitAdaptiveKV::Tier::BF16
          tier
        else
          nil
        end
      end

      # A zero-initialized status alone cannot distinguish success from a
      # command buffer that never executed. This final encoder publishes a
      # non-zero marker only after all earlier encoders completed cleanly.
      private def encode_finalize(command : ML::Metal::CommandBuffer,
                                  status : ML::MetalBuffer) : Nil
        encoder = ML::Metal::ComputeEncoder.new(command)
        encoder.set_pipeline(finalize_pipeline)
        encoder.set_buffer(status, 0, ML::Metal::BufferAccess::ReadWrite)
        encoder.dispatch_1d(1, 1)
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
        tile = gqa6_tile
        @@gqa6_pipeline_mutex.synchronize do
          @@gqa6_pipelines[tile] ||= ML::Metal::PipelineCache.get("qwen35_qbit_adaptive_attn_decode_gqa6_tile#{tile}") {
            ML::Metal::ComputePipeline.new(
              "qwen35_qbit_adaptive_attn_decode_gqa6_tile#{tile}",
              gqa6_source(tile),
              "qwen35_qbit_adaptive_attn_decode_gqa6",
            )
          }
        end
      end

      private def prefill_gqa6_pipeline : ML::Metal::ComputePipeline
        tile = gqa6_tile
        @@prefill_gqa6_pipeline_mutex.synchronize do
          @@prefill_gqa6_pipelines[tile] ||= ML::Metal::PipelineCache.get("qwen35_qbit_adaptive_prefill_chunk_gqa6_tile#{tile}") {
            ML::Metal::ComputePipeline.new(
              "qwen35_qbit_adaptive_prefill_chunk_gqa6_tile#{tile}",
              gqa6_source(tile),
              "qwen35_qbit_adaptive_prefill_chunk_gqa6",
            )
          }
        end
      end

      private def gqa6_tile : Int32
        QwenQBitAdaptiveMetalPolicy.gqa6_tile(
          ML::Metal::Device.instance.name,
          ENV["QWEN35_ADAPTIVE_GQA6_TILE"]?,
        )
      end

      private def gqa6_source(tile : Int32) : String
        tile == 15 ? SOURCE_TILE15 : SOURCE
      end

      private def pack_pipeline : ML::Metal::ComputePipeline
        @@pack_pipeline_mutex.synchronize do
          @@pack_pipeline ||= ML::Metal::PipelineCache.get("qwen35_qbit_adaptive_pack_row") {
            ML::Metal::ComputePipeline.new("qwen35_qbit_adaptive_pack_row", PACK_SOURCE)
          }
        end
      end

      private def finalize_pipeline : ML::Metal::ComputePipeline
        @@finalize_pipeline_mutex.synchronize do
          @@finalize_pipeline ||= ML::Metal::PipelineCache.get("qwen35_qbit_adaptive_finalize_status") {
            ML::Metal::ComputePipeline.new("qwen35_qbit_adaptive_finalize_status", PACK_SOURCE)
          }
        end
      end
    {% end %}
  end
end
