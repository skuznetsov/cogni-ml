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
        def transport_identity : UInt64
          0_u64
        end

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
      private class PendingAppend
        getter status : ML::MetalBuffer
        getter command : ResidentCommandBuffer
        getter token_count : Int32
        getter start_token : Int32

        def initialize(@status : ML::MetalBuffer,
                       @command : ResidentCommandBuffer,
                       @token_count : Int32,
                       @start_token : Int32)
        end
      end

      getter max_seq : Int32
      getter n_head_kv : Int32
      getter head_dim : Int32
      getter compressed_bytes : Int64

      @lifecycle_mutex : Mutex
      @released : Bool
      @cache_len : Int32
      @k_plan : QwenQBitAdaptiveKV::Plan?
      @v_plan : QwenQBitAdaptiveKV::Plan?
      @pending_appends : Array(PendingAppend)
      @pending_transport_identity : UInt64?

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
        @pending_appends = [] of PendingAppend
        @pending_transport_identity = nil
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
      def with_pending_buffers(command : ResidentCommandBuffer, &)
        @lifecycle_mutex.synchronize do
          ensure_live!
          find_pending!(command)
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
          if command.committed?
            raise ArgumentError.new("adaptive resident QBit append requires an uncommitted command")
          end
          if @pending_appends.any? { |pending| pending.command.same?(command) }
            raise ArgumentError.new("adaptive resident QBit command already owns a pending append")
          end
          if owner = @pending_transport_identity
            unless owner == command.transport_identity
              raise ArgumentError.new("adaptive resident QBit pending appends must use one Metal queue")
            end
          end
          unless @k_plan && @v_plan
            raise ArgumentError.new("adaptive resident QBit cache is not appendable")
          end
          unless token_count > 0
            raise ArgumentError.new("adaptive resident QBit append token count must be positive")
          end
          reserved_tail = pending_tail
          unless expected_start_token == reserved_tail
            raise ArgumentError.new("adaptive resident QBit append start does not match the reserved tail")
          end
          if reserved_tail.to_i64 + token_count > @max_seq
            raise ArgumentError.new("adaptive resident QBit append exceeds cache capacity")
          end
          unless status.valid? && status.size >= sizeof(UInt32)
            raise ArgumentError.new("adaptive resident QBit pending status buffer is invalid")
          end

          @pending_appends << PendingAppend.new(status, command, token_count, reserved_tail)
          @pending_transport_identity ||= command.transport_identity
          reserved_tail
        end
      end

      # Publish only after the caller has committed and waited for the external
      # command. A lone failed marker clears its reservation; a failed FIFO head
      # with a queued suffix remains pending until the owner drains every writer
      # and discards that suffix without advancing the visible prefix.
      def finish_pending_append!(command : ResidentCommandBuffer) : Nil
        @lifecycle_mutex.synchronize do
          ensure_live!
          pending = find_pending!(command)
          ensure_fifo_pending!(pending)
          unless command.completed_successfully?
            raise ArgumentError.new("adaptive resident QBit publication requires its command to complete successfully")
          end

          status_code = pending.status.contents.as(Pointer(UInt32)).value
          unless status_code == DEVICE_SUCCESS
            if @pending_appends.last?.same?(pending)
              @pending_appends.shift
              pending.status.release
              clear_pending_transport_if_empty!
            end
            raise ArgumentError.new("adaptive resident QBit prefill/pack failed closed (status=#{status_code})")
          end
          @pending_appends.shift
          pending.status.release
          @cache_len += pending.token_count
          clear_pending_transport_if_empty!
        end
      end

      # Non-mutating half of group publication. Callers with several caches on
      # one command validate every tail marker before advancing any live prefix.
      def validate_pending_append!(command : ResidentCommandBuffer) : Nil
        @lifecycle_mutex.synchronize do
          ensure_live!
          pending = find_pending!(command)
          ensure_fifo_pending!(pending)
          unless command.completed_successfully?
            raise ArgumentError.new("adaptive resident QBit publication requires its command to complete successfully")
          end
          status_code = pending.status.contents.as(Pointer(UInt32)).value
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
          return if @pending_appends.empty?
          pending = find_pending!(command)
          if command.committed? && !command.completed?
            raise ArgumentError.new("adaptive resident QBit command is not completed; cancellation is unsafe")
          end
          unless @pending_appends.last?.same?(pending)
            raise ArgumentError.new("adaptive resident QBit cancellation must remove the reserved tail")
          end
          @pending_appends.pop
          pending.status.release
          clear_pending_transport_if_empty!
        end
      end

      # Error-corridor cleanup after the owner has cancelled or drained every
      # command that can write the unpublished suffix. The visible prefix never
      # advances; releasing in reverse reservation order removes the complete
      # suffix without creating a publishable gap.
      def discard_pending_appends! : Nil
        @lifecycle_mutex.synchronize do
          ensure_live!
          @pending_appends.each do |pending|
            if pending.command.committed? && !pending.command.completed?
              raise ArgumentError.new("adaptive resident QBit pending suffix still has in-flight GPU work")
            end
          end
          @pending_appends.reverse_each { |pending| pending.status.release }
          @pending_appends.clear
          @pending_transport_identity = nil
        end
      end

      def with_pending_status(command : ResidentCommandBuffer, &)
        @lifecycle_mutex.synchronize do
          ensure_live!
          pending = find_pending!(command)
          if command.committed?
            raise ArgumentError.new("adaptive resident QBit finalizer requires an uncommitted command")
          end
          yield pending.status
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
        if @pending_appends.any?
          raise ArgumentError.new("adaptive resident QBit append is pending shared-command completion")
        end
      end

      private def pending_tail : Int32
        if pending = @pending_appends.last?
          pending.start_token + pending.token_count
        else
          @cache_len
        end
      end

      private def find_pending!(command : ResidentCommandBuffer) : PendingAppend
        @pending_appends.find { |pending| pending.command.same?(command) } ||
          raise ArgumentError.new("adaptive resident QBit command does not own a pending append")
      end

      private def ensure_fifo_pending!(pending : PendingAppend) : Nil
        unless @pending_appends.first?.same?(pending)
          raise ArgumentError.new("adaptive resident QBit publication must follow FIFO reservation order")
        end
      end

      private def clear_pending_transport_if_empty! : Nil
        @pending_transport_identity = nil if @pending_appends.empty?
      end
    end

    {% unless flag?(:cpu_only) %}
      SOURCE = {{ read_file("#{__DIR__}/kernels/qbit_adaptive_attn_decode_qwen35.metal") }}
      SOURCE_TILE15 = SOURCE.sub(
        "constant uint QQA_ADAPTIVE_GQA6_TILE = 16;",
        "constant uint QQA_ADAPTIVE_GQA6_TILE = 15;",
      )
      raise "adaptive GQA6 tile-15 source patch no longer matches" if SOURCE_TILE15 == SOURCE
      SOURCE_DEQUANT_T4 = SOURCE.sub(
        "constant bool QQA_ADAPTIVE_DEQUANT_T4 = false;",
        "constant bool QQA_ADAPTIVE_DEQUANT_T4 = true;",
      )
      raise "adaptive t4 dequant source patch no longer matches" if SOURCE_DEQUANT_T4 == SOURCE
      SOURCE_TILE15_DEQUANT_T4 = SOURCE_DEQUANT_T4.sub(
        "constant uint QQA_ADAPTIVE_GQA6_TILE = 16;",
        "constant uint QQA_ADAPTIVE_GQA6_TILE = 15;",
      )
      if SOURCE_TILE15_DEQUANT_T4 == SOURCE_DEQUANT_T4
        raise "adaptive t4 dequant tile-15 source patch no longer matches"
      end
      SOURCE_P4_SPLITK_T8 = SOURCE.sub(
        "constant bool QQA_ADAPTIVE_P4_SPLITK_T8 = false;",
        "constant bool QQA_ADAPTIVE_P4_SPLITK_T8 = true;",
      )
      if SOURCE_P4_SPLITK_T8 == SOURCE
        raise "adaptive P4 split-K t8 source patch no longer matches"
      end
      SOURCE_TILE15_P4_SPLITK_T8 = SOURCE_P4_SPLITK_T8.sub(
        "constant uint QQA_ADAPTIVE_GQA6_TILE = 16;",
        "constant uint QQA_ADAPTIVE_GQA6_TILE = 15;",
      )
      if SOURCE_TILE15_P4_SPLITK_T8 == SOURCE_P4_SPLITK_T8
        raise "adaptive P4 split-K t8 tile-15 source patch no longer matches"
      end
      SOURCE_P4_SPLITK_DIRECT_QK = SOURCE_P4_SPLITK_T8.sub(
        "constant bool QQA_ADAPTIVE_P4_SPLITK_DIRECT_QK = false;",
        "constant bool QQA_ADAPTIVE_P4_SPLITK_DIRECT_QK = true;",
      )
      if SOURCE_P4_SPLITK_DIRECT_QK == SOURCE_P4_SPLITK_T8
        raise "adaptive P4 split-K direct-QK source patch no longer matches"
      end
      SOURCE_TILE15_P4_SPLITK_DIRECT_QK = SOURCE_TILE15_P4_SPLITK_T8.sub(
        "constant bool QQA_ADAPTIVE_P4_SPLITK_DIRECT_QK = false;",
        "constant bool QQA_ADAPTIVE_P4_SPLITK_DIRECT_QK = true;",
      )
      if SOURCE_TILE15_P4_SPLITK_DIRECT_QK == SOURCE_TILE15_P4_SPLITK_T8
        raise "adaptive P4 split-K direct-QK tile-15 source patch no longer matches"
      end
      SOURCE_P4_SPLITK_V_CONTIGUOUS = SOURCE_P4_SPLITK_DIRECT_QK.sub(
        "constant bool QQA_ADAPTIVE_P4_SPLITK_V_CONTIGUOUS = false;",
        "constant bool QQA_ADAPTIVE_P4_SPLITK_V_CONTIGUOUS = true;",
      )
      if SOURCE_P4_SPLITK_V_CONTIGUOUS == SOURCE_P4_SPLITK_DIRECT_QK
        raise "adaptive P4 split-K contiguous-V source patch no longer matches"
      end
      SOURCE_TILE15_P4_SPLITK_V_CONTIGUOUS = SOURCE_TILE15_P4_SPLITK_DIRECT_QK.sub(
        "constant bool QQA_ADAPTIVE_P4_SPLITK_V_CONTIGUOUS = false;",
        "constant bool QQA_ADAPTIVE_P4_SPLITK_V_CONTIGUOUS = true;",
      )
      if SOURCE_TILE15_P4_SPLITK_V_CONTIGUOUS == SOURCE_TILE15_P4_SPLITK_DIRECT_QK
        raise "adaptive P4 split-K contiguous-V tile-15 source patch no longer matches"
      end
      SOURCE_BF16_SPLITK_T8 = SOURCE.sub(
        "constant bool QQA_ADAPTIVE_BF16_SPLITK_T8 = false;",
        "constant bool QQA_ADAPTIVE_BF16_SPLITK_T8 = true;",
      )
      if SOURCE_BF16_SPLITK_T8 == SOURCE
        raise "adaptive BF16 split-K t8 source patch no longer matches"
      end
      SOURCE_TILE15_BF16_SPLITK_T8 = SOURCE_BF16_SPLITK_T8.sub(
        "constant uint QQA_ADAPTIVE_GQA6_TILE = 16;",
        "constant uint QQA_ADAPTIVE_GQA6_TILE = 15;",
      )
      if SOURCE_TILE15_BF16_SPLITK_T8 == SOURCE_BF16_SPLITK_T8
        raise "adaptive BF16 split-K t8 tile-15 source patch no longer matches"
      end
      SOURCE_SPLITK_STAGE2_FUSED = SOURCE.sub(
        "constant bool QQA_ADAPTIVE_SPLITK_STAGE2_FUSED = false;",
        "constant bool QQA_ADAPTIVE_SPLITK_STAGE2_FUSED = true;",
      )
      if SOURCE_SPLITK_STAGE2_FUSED == SOURCE
        raise "adaptive fused split-K stage2 source patch no longer matches"
      end
      PACK_SOURCE = {{ read_file("#{__DIR__}/kernels/qbit_adaptive_pack_qwen35.metal") }}
      PACK_SOURCE_PREFIX_QUANT = PACK_SOURCE.sub(
        "constant bool QQP_PREFIX_QUANT = false;",
        "constant bool QQP_PREFIX_QUANT = true;",
      )
      if PACK_SOURCE_PREFIX_QUANT == PACK_SOURCE
        raise "adaptive prefix-only pack quantizer source patch no longer matches"
      end
      @@gqa6_pipelines = Hash(Int32, ML::Metal::ComputePipeline).new
      @@gqa6_pipeline_mutex = Mutex.new
      @@prefill_gqa6_pipelines = Hash(Tuple(Int32, Bool), ML::Metal::ComputePipeline).new
      @@prefill_gqa6_pipeline_mutex = Mutex.new
      @@decode_splitk_stage1_pipelines = Hash(Tuple(Int32, Bool, Bool, Bool, Bool, Bool), ML::Metal::ComputePipeline).new
      @@decode_splitk_stage1_pipeline_mutex = Mutex.new
      @@decode_splitk_stage2_pipelines = Hash(Bool, ML::Metal::ComputePipeline).new
      @@decode_splitk_stage2_pipeline_mutex = Mutex.new
      @@pack_pipelines = Hash(Bool, ML::Metal::ComputePipeline).new
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
    # An optional timing pointer receives the completed K/V pack command's GPU
    # interval without changing its synchronous publication boundary.
    def append_from_metal(cache : Cache,
                          k_source : ML::MetalBuffer,
                          v_source : ML::MetalBuffer,
                          token_count : Int32,
                          source_token_offset : Int32 = 0,
                          gpu_elapsed_seconds : Pointer(Float64) = Pointer(Float64).null) : Nil
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
        cache.with_append(token_count) do |k_base, k_metadata, k_sidecar, v_base, v_metadata, v_sidecar, k_plan, v_plan, start_token|
          status = upload(Bytes.new(sizeof(UInt32), 0_u8))
          begin
            source_row_offset = checked_u32(source_token_offset.to_i64 * cache.n_head_kv, "source row offset")
            destination_row_offset = checked_u32(start_token.to_i64 * cache.n_head_kv, "destination row offset")
            row_count = checked_i32(token_count.to_i64 * cache.n_head_kv, "row count")
            command = ML::Metal::CommandBuffer.new
            encode_pack(command, k_source, k_base, k_metadata, k_sidecar, status,
              source_row_offset, destination_row_offset, row_count,
              k_plan.uniform_tier == QwenQBitAdaptiveKV::Tier::BF16)
            encode_pack(command, v_source, v_base, v_metadata, v_sidecar, status,
              source_row_offset, destination_row_offset, row_count,
              v_plan.uniform_tier == QwenQBitAdaptiveKV::Tier::BF16)
            encode_finalize(command, status)
            if gpu_elapsed_seconds.null?
              command.commit_and_wait
            else
              gpu_elapsed_seconds.value = command.commit_and_wait_gpu_elapsed_seconds
            end
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
          cache.with_pending_buffers(command) do |k_base, k_metadata, k_sidecar, v_base, v_metadata, v_sidecar, uniform_tier|
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
                output, status, cache.max_seq, packed_len, chunk_tokens, source_token_offset,
                n_head, cache.n_head_kv, cache.head_dim,
                heads_per_group, scale, qualified_uniform_tier,
              )
              encode_pack(command, k_source, k_base, k_metadata, k_sidecar, status,
                source_row_offset, destination_row_offset, row_count,
                uniform_tier == QwenQBitAdaptiveKV::Tier::BF16)
              encode_pack(command, v_source, v_base, v_metadata, v_sidecar, status,
                source_row_offset, destination_row_offset, row_count,
                uniform_tier == QwenQBitAdaptiveKV::Tier::BF16)
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
      unique_caches = [] of Cache
      caches.each do |cache|
        if unique_caches.any? { |seen| seen.same?(cache) }
          raise ArgumentError.new("adaptive resident QBit publication contains a duplicate cache")
        end
        unique_caches << cache
      end
      caches.each { |cache| cache.validate_pending_append!(command) }
      caches.each { |cache| cache.finish_pending_append!(command) }
    end

    def discard_pending_appends!(cache : Cache) : Nil
      cache.discard_pending_appends!
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
                              row_count : Int32,
                              automatic_prefix_quant : Bool) : Nil
        encoder = ML::Metal::ComputeEncoder.new(command)
        encoder.set_pipeline(pack_pipeline(automatic: automatic_prefix_quant))
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
                                       max_seq : Int32,
                                       packed_len : Int32,
                                       token_count : Int32,
                                       source_token_offset : Int32,
                                       n_head : Int32,
                                       n_head_kv : Int32,
                                       head_dim : Int32,
                                       heads_per_group : Int32,
                                       scale : Float32,
                                       uniform_tier : QwenQBitAdaptiveKV::Tier?) : Nil
        if QwenQBitAdaptiveMetalPolicy.decode_splitk?(
             packed_len, token_count, !uniform_tier.nil?,
             ENV["QWEN35_ADAPTIVE_SPLITK"]?,
             ENV["QWEN35_ADAPTIVE_SPLITK_MIN_CTX"]?,
           )
          encode_decode_splitk(
            command, q_source, gate_source, k_source, v_source,
            k_base, k_metadata, k_sidecar,
            v_base, v_metadata, v_sidecar,
            output, status, max_seq, packed_len, n_head, n_head_kv, head_dim,
            heads_per_group, scale, uniform_tier.not_nil!,
          )
          return
        end

        automatic_t4 = QwenQBitAdaptiveMetalPolicy.automatic_dequant_t4?(
          token_count, false,
          uniform_tier == QwenQBitAdaptiveKV::Tier::P4,
          uniform_tier == QwenQBitAdaptiveKV::Tier::BF16,
        )
        encoder = ML::Metal::ComputeEncoder.new(command)
        encoder.set_pipeline(prefill_gqa6_pipeline(automatic_t4: automatic_t4))
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

      private def encode_decode_splitk(command : ML::Metal::CommandBuffer,
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
                                       max_seq : Int32,
                                       packed_len : Int32,
                                       n_head : Int32,
                                       n_head_kv : Int32,
                                       head_dim : Int32,
                                       heads_per_group : Int32,
                                       scale : Float32,
                                       uniform_tier : QwenQBitAdaptiveKV::Tier) : Nil
        chunk_size = adaptive_splitk_chunk_size
        block_count = checked_i32(
          (packed_len.to_i64 + 1_i64 + chunk_size.to_i64 - 1_i64) // chunk_size,
          "adaptive split-K block count",
        )
        # Scratch.get keys by both tag and byte size. Size from the immutable
        # cache capacity so growing decode does not retain one pool entry per
        # 64-token block-count boundary.
        scratch_block_count = checked_i32(
          (max_seq.to_i64 + chunk_size.to_i64 - 1_i64) // chunk_size,
          "adaptive split-K scratch block count",
        )
        partial_o = Qwen35Metal::Scratch.get(
          :adaptive_qbit_splitk_o,
          n_head.to_i64 * scratch_block_count * head_dim * sizeof(Float32),
        )
        partial_m = Qwen35Metal::Scratch.get(
          :adaptive_qbit_splitk_m,
          n_head.to_i64 * scratch_block_count * sizeof(Float32),
        )
        partial_l = Qwen35Metal::Scratch.get(
          :adaptive_qbit_splitk_l,
          n_head.to_i64 * scratch_block_count * sizeof(Float32),
        )
        stage1_pipeline = decode_splitk_stage1_pipeline(
          uniform_tier, k_sidecar, v_sidecar, packed_len,
        )
        stage2_pipeline = decode_splitk_stage2_pipeline(uniform_tier, packed_len)

        stage1 = ML::Metal::ComputeEncoder.new(command)
        stage1.set_pipeline(stage1_pipeline)
        stage1.set_buffer(q_source, 0)
        stage1.set_buffer(k_source, 1)
        stage1.set_buffer(v_source, 2)
        stage1.set_buffer(k_base, 3)
        stage1.set_buffer(k_metadata, 4)
        stage1.set_buffer(k_sidecar, 5)
        stage1.set_buffer(v_base, 6)
        stage1.set_buffer(v_metadata, 7)
        stage1.set_buffer(v_sidecar, 8)
        stage1.set_buffer(partial_o, 9, ML::Metal::BufferAccess::Write)
        stage1.set_buffer(partial_m, 10, ML::Metal::BufferAccess::Write)
        stage1.set_buffer(partial_l, 11, ML::Metal::BufferAccess::Write)
        stage1.set_buffer(status, 12, ML::Metal::BufferAccess::Write)
        stage1.set_value(packed_len.to_u32, 13)
        stage1.set_value(n_head.to_u32, 14)
        stage1.set_value(n_head_kv.to_u32, 15)
        stage1.set_value(head_dim.to_u32, 16)
        stage1.set_value(heads_per_group.to_u32, 17)
        stage1.set_value(scale, 18)
        stage1.set_value(chunk_size.to_u32, 19)
        stage1.set_value(block_count.to_u32, 20)
        stage1.set_value(uniform_tier.value.to_u32, 21)
        stage1.dispatch_threadgroups({n_head_kv, block_count, 1}, {192, 1, 1})
        stage1.end_encoding

        stage2 = ML::Metal::ComputeEncoder.new(command)
        stage2.set_pipeline(stage2_pipeline)
        stage2.set_buffer(gate_source, 0)
        stage2.set_buffer(partial_o, 1)
        stage2.set_buffer(partial_m, 2)
        stage2.set_buffer(partial_l, 3)
        stage2.set_buffer(output, 4, ML::Metal::BufferAccess::Write)
        stage2.set_buffer(status, 5, ML::Metal::BufferAccess::Write)
        stage2.set_value(n_head.to_u32, 6)
        stage2.set_value(head_dim.to_u32, 7)
        stage2.set_value(block_count.to_u32, 8)
        stage2.dispatch_threadgroups({n_head, 1, 1}, {32, 1, 1})
        stage2.end_encoding
      end

      private def adaptive_splitk_chunk_size : Int32
        raw = ENV["QWEN35_ADAPTIVE_SPLITK_CHUNK"]?
        return 64 unless raw
        value = raw.strip.to_i?
        unless value && value > 0
          raise ArgumentError.new("QWEN35_ADAPTIVE_SPLITK_CHUNK must be a positive integer")
        end
        value
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

      private def prefill_gqa6_pipeline(automatic_t4 : Bool) : ML::Metal::ComputePipeline
        tile = gqa6_tile
        dequant_t4 = dequant_t4?(automatic: automatic_t4)
        key = {tile, dequant_t4}
        suffix = dequant_t4 ? "_dequant_t4" : ""
        @@prefill_gqa6_pipeline_mutex.synchronize do
          @@prefill_gqa6_pipelines[key] ||= ML::Metal::PipelineCache.get("qwen35_qbit_adaptive_prefill_chunk_gqa6_tile#{tile}#{suffix}") {
            ML::Metal::ComputePipeline.new(
              "qwen35_qbit_adaptive_prefill_chunk_gqa6_tile#{tile}#{suffix}",
              gqa6_source(tile, dequant_t4),
              "qwen35_qbit_adaptive_prefill_chunk_gqa6",
            )
          }
        end
      end

      private def decode_splitk_stage1_pipeline(
        uniform_tier : QwenQBitAdaptiveKV::Tier,
        k_sidecar : ML::MetalBuffer,
        v_sidecar : ML::MetalBuffer,
        packed_len : Int32,
      ) : ML::Metal::ComputePipeline
        tile = gqa6_tile
        p4_t8 = uniform_tier == QwenQBitAdaptiveKV::Tier::P4 &&
                QwenQBitAdaptiveMetalPolicy.p4_splitk_t8?(
                  ML::Metal::Device.instance.name,
                  packed_len,
                  ENV["QWEN35_ADAPTIVE_P4_SPLITK_T8"]?,
                )
        bf16_t8 = uniform_tier == QwenQBitAdaptiveKV::Tier::BF16 &&
                  vector_load_aligned?(k_sidecar) &&
                  vector_load_aligned?(v_sidecar) &&
                  QwenQBitAdaptiveMetalPolicy.bf16_splitk_t8?(
                    ML::Metal::Device.instance.name,
                    packed_len,
                    ENV["QWEN35_ADAPTIVE_BF16_SPLITK_T8"]?,
                  )
        automatic_t4 = QwenQBitAdaptiveMetalPolicy.automatic_dequant_t4?(
          1, true,
          uniform_tier == QwenQBitAdaptiveKV::Tier::P4,
          uniform_tier == QwenQBitAdaptiveKV::Tier::BF16,
        )
        # T8 is a complete tier-specific loader, not an additive T4 modifier.
        # Keep the pipeline identity canonical when both explicit knobs are set.
        dequant_t4 = (p4_t8 || bf16_t8) ? false : dequant_t4?(automatic: automatic_t4)
        direct_qk = QwenQBitAdaptiveMetalPolicy.p4_splitk_direct_qk?(
          uniform_tier == QwenQBitAdaptiveKV::Tier::P4,
          p4_t8,
          ENV["QWEN35_ADAPTIVE_P4_SPLITK_DIRECT_QK"]?,
        )
        v_contiguous = QwenQBitAdaptiveMetalPolicy.p4_splitk_v_contiguous?(
          uniform_tier == QwenQBitAdaptiveKV::Tier::P4,
          p4_t8,
          direct_qk,
          ENV["QWEN35_ADAPTIVE_P4_SPLITK_V_CONTIGUOUS"]?,
        )
        key = {tile, dequant_t4, p4_t8, bf16_t8, direct_qk, v_contiguous}
        suffix = dequant_t4 ? "_dequant_t4" : ""
        suffix += "_p4_t8" if p4_t8
        suffix += "_bf16_t8" if bf16_t8
        suffix += "_direct_qk" if direct_qk
        suffix += "_v_contiguous" if v_contiguous
        @@decode_splitk_stage1_pipeline_mutex.synchronize do
          @@decode_splitk_stage1_pipelines[key] ||= ML::Metal::PipelineCache.get("qwen35_qbit_adaptive_decode_splitk_stage1_gqa6_tile#{tile}#{suffix}") {
            ML::Metal::ComputePipeline.new(
              "qwen35_qbit_adaptive_decode_splitk_stage1_gqa6_tile#{tile}#{suffix}",
              gqa6_source(tile, dequant_t4, p4_t8, bf16_t8, direct_qk, v_contiguous),
              "qwen35_qbit_adaptive_decode_splitk_stage1_gqa6",
            )
          }
        end
      end

      private def vector_load_aligned?(buffer : ML::MetalBuffer) : Bool
        pointer = buffer.contents
        !pointer.null? && pointer.address % 16_u64 == 0_u64
      end

      private def decode_splitk_stage2_pipeline(uniform_tier : QwenQBitAdaptiveKV::Tier,
                                                packed_len : Int32) : ML::Metal::ComputePipeline
        fused = QwenQBitAdaptiveMetalPolicy.splitk_stage2_fused?(
          ML::Metal::Device.instance.name,
          uniform_tier == QwenQBitAdaptiveKV::Tier::P4,
          uniform_tier == QwenQBitAdaptiveKV::Tier::BF16,
          packed_len,
          ENV["QWEN35_ADAPTIVE_P4_SPLITK_T8"]?,
          ENV["QWEN35_ADAPTIVE_SPLITK_STAGE2_FUSED"]?,
        )
        suffix = fused ? "_fused" : ""
        @@decode_splitk_stage2_pipeline_mutex.synchronize do
          @@decode_splitk_stage2_pipelines[fused] ||= ML::Metal::PipelineCache.get("qwen35_qbit_adaptive_decode_splitk_stage2#{suffix}") {
            ML::Metal::ComputePipeline.new(
              "qwen35_qbit_adaptive_decode_splitk_stage2#{suffix}",
              fused ? SOURCE_SPLITK_STAGE2_FUSED : SOURCE,
              "qwen35_qbit_adaptive_decode_splitk_stage2",
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

      private def dequant_t4?(automatic : Bool) : Bool
        QwenQBitAdaptiveMetalPolicy.dequant_t4?(
          ML::Metal::Device.instance.name,
          automatic,
          ENV["QWEN35_ADAPTIVE_DEQUANT_T4"]?,
        )
      end

      private def gqa6_source(tile : Int32,
                              dequant_t4 : Bool = false,
                              p4_t8 : Bool = false,
                              bf16_t8 : Bool = false,
                              direct_qk : Bool = false,
                              v_contiguous : Bool = false) : String
        if bf16_t8
          return tile == 15 ? SOURCE_TILE15_BF16_SPLITK_T8 : SOURCE_BF16_SPLITK_T8
        end
        if p4_t8
          if v_contiguous
            return tile == 15 ? SOURCE_TILE15_P4_SPLITK_V_CONTIGUOUS : SOURCE_P4_SPLITK_V_CONTIGUOUS
          end
          if direct_qk
            return tile == 15 ? SOURCE_TILE15_P4_SPLITK_DIRECT_QK : SOURCE_P4_SPLITK_DIRECT_QK
          end
          return tile == 15 ? SOURCE_TILE15_P4_SPLITK_T8 : SOURCE_P4_SPLITK_T8
        end
        if dequant_t4
          tile == 15 ? SOURCE_TILE15_DEQUANT_T4 : SOURCE_DEQUANT_T4
        else
          tile == 15 ? SOURCE_TILE15 : SOURCE
        end
      end

      private def pack_pipeline(automatic : Bool) : ML::Metal::ComputePipeline
        prefix_quant = QwenQBitAdaptiveMetalPolicy.pack_prefix_quant?(
          ML::Metal::Device.instance.name,
          automatic,
          ENV["QWEN35_ADAPTIVE_PACK_PREFIX_QUANT"]?,
        )
        suffix = prefix_quant ? "_prefix_quant" : ""
        @@pack_pipeline_mutex.synchronize do
          @@pack_pipelines[prefix_quant] ||= ML::Metal::PipelineCache.get("qwen35_qbit_adaptive_pack_row#{suffix}") {
            ML::Metal::ComputePipeline.new(
              "qwen35_qbit_adaptive_pack_row#{suffix}",
              prefix_quant ? PACK_SOURCE_PREFIX_QUANT : PACK_SOURCE,
              "qwen35_qbit_adaptive_pack_row",
            )
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
