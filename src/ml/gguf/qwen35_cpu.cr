require "./qwen35_meta"
require "./qwen35_weights"
require "./quant_matmul"
require "./qwen35_metal"
require "./qwen_qbit_adaptive_resident_kv"

# Qwen 3.5 / 3.6 CPU reference forward pass.
#
# Purpose: ground-truth correctness path. Not performance-tuned.
# The Metal port (Phases 2+) must match this output to cosine ≥ 0.9999.
#
# Layout conventions:
#   - All activations are Array(Float32), flattened row-major.
#   - Per-head tensors: [n_head, head_dim] flattened as [n_head * head_dim],
#     head h at offset h*head_dim.
#   - KV cache stored as [pos, n_kv_heads, head_dim] flattened.
#   - Autoregressive decode only (one token at a time). Prefill iterates decode.

module ML::GGUF
  module Qwen35CPU
    {% if flag?(:cpu_only) %}
      alias PrefillScratchArena = Nil
    {% else %}
      alias PrefillScratchArena = Qwen35Metal::Scratch::Arena
    {% end %}

    extend self
    record AllowedTokenScore,
      token_id : Int32,
      logit : Float32
    record PrefillResidentTop1Append,
      id_buf : ML::MetalBuffer,
      value_buf : ML::MetalBuffer,
      encoded : Array(Bool),
      allowed_ids : Array(Int32)? = nil
    {% if flag?(:cpu_only) %}
      alias PrefillCommandBuffer = Nil
    {% else %}
      alias PrefillCommandBuffer = ML::Metal::CommandBuffer
      alias AdaptivePrefillEncoder = Proc(
        ML::Metal::CommandBuffer,
        ML::MetalBuffer, ML::MetalBuffer, ML::MetalBuffer,
        ML::MetalBuffer, ML::MetalBuffer, Nil,
      )
    {% end %}
    # Keep prompt chunks large enough to avoid CPU-side boundary overhead while
    # preserving an env override for small-memory experiments.
    FALLBACK_PREFILL_CHUNK_SIZE          =   4096
    ADAPTIVE_RESIDENT_PREFILL_CHUNK_SIZE =   2048
    PREFILL_APPEND_ROW_GROUP_BUDGET      =   2048
    PREFILL_APPEND_COOLDOWN_MS           =     50
    QWEN38_Q4_K_M_FILE_TYPE              = 15_i64
    GIB                                  = 1024_u64 * 1024_u64 * 1024_u64
    @@default_prefill_chunk_size : Int32?
    @@prefill_gc_guard_active = false

    def prefill_chunk_size_for_memory(total_bytes : UInt64?) : Int32
      return FALLBACK_PREFILL_CHUNK_SIZE unless bytes = total_bytes
      return 8192 if bytes >= 48_u64 * GIB
      return 4096 if bytes >= 24_u64 * GIB
      2048
    end

    def default_prefill_chunk_size : Int32
      @@default_prefill_chunk_size ||= prefill_chunk_size_for_memory(physical_memory_bytes?)
    end

    # Adaptive packing adds enough work to large row tiles that repeated
    # prefill can trip the macOS GPU interactivity watchdog. Keep automatic
    # resident tiles bounded; an explicit setting remains an exact override.
    def prefill_chunk_size(resident_adaptive : Bool,
                           configured : String? = ENV["QWEN35_PREFILL_CHUNK_SIZE"]?) : Int32
      if raw = configured
        value = raw.to_i?
        unless value && value > 0
          raise ArgumentError.new("QWEN35_PREFILL_CHUNK_SIZE must be a positive integer")
        end
        return value
      end
      size = default_prefill_chunk_size
      resident_adaptive ? Math.min(size, ADAPTIVE_RESIDENT_PREFILL_CHUNK_SIZE) : size
    end

    # Bound continuous GPU occupancy without breaking the resident hidden/KV
    # corridor. Zero keeps the historical single-command behavior. The default
    # only slices large row-prefill batches and keeps roughly the same
    # token-rows x layer-groups work in each submitted command.
    def prefill_append_group_limit(n_tokens : Int32,
                                   configured : String? = ENV["QWEN35_PREFILL_APPEND_MAX_GROUPS"]?) : Int32
      raise ArgumentError.new("prefill append group limit requires positive token rows") unless n_tokens > 0
      if raw = configured
        value = raw.to_i?
        unless value && value >= 0
          raise ArgumentError.new("QWEN35_PREFILL_APPEND_MAX_GROUPS must be a non-negative integer")
        end
        return value
      end
      return 0 if n_tokens < 1024
      Math.max(1, PREFILL_APPEND_ROW_GROUP_BUDGET // n_tokens)
    end

    # Give the display compositor a bounded scheduling window between completed
    # long-prefill commands. Zero removes the idle window; the group-limit zero
    # setting disables command rotation itself.
    def prefill_append_cooldown_ms(
      configured : String? = ENV["QWEN35_PREFILL_APPEND_COOLDOWN_MS"]?,
    ) : Int32
      return PREFILL_APPEND_COOLDOWN_MS unless raw = configured
      value = raw.to_i?
      unless value && value >= 0
        raise ArgumentError.new("QWEN35_PREFILL_APPEND_COOLDOWN_MS must be a non-negative integer")
      end
      value
    end

    # Remove the compositor pause automatically only in exact measured
    # Qwen3.5-9B and Qwen3.8-27B corridors on M2 Max. Every unknown shape or
    # execution mode keeps the conservative default; the existing cooldown
    # setting is the immediate explicit override and rollback.
    def prefill_append_cooldown_policy_ms(
      cooldown_configured : String? = ENV["QWEN35_PREFILL_APPEND_COOLDOWN_MS"]?,
      group_limit_configured : String? = ENV["QWEN35_PREFILL_APPEND_MAX_GROUPS"]?,
      chunk_size_configured : String? = ENV["QWEN35_PREFILL_CHUNK_SIZE"]?,
      *,
      device_name : String,
      start_pos : Int32,
      n_tokens : Int32,
      n_layer : Int32,
      layer_limit : Int32,
      n_embd : Int32,
      n_ff : Int32,
      n_head : Int32,
      n_head_kv : Int32,
      head_dim : Int32,
      full_attention_interval : Int32,
      kv_cache_f16 : Bool,
      resident_adaptive : Bool,
      checkpoint_requested : Bool,
      boundary_profile : Bool,
      graph_depth : Int32,
      flash_d256 : Bool,
      model_capability : Q4GemvX16Capability = Q4GemvX16Capability::Unknown,
      gguf_file_type : Int64? = nil,
    ) : Int32
      return prefill_append_cooldown_ms(cooldown_configured) if cooldown_configured
      return PREFILL_APPEND_COOLDOWN_MS if group_limit_configured || chunk_size_configured
      return PREFILL_APPEND_COOLDOWN_MS unless device_name == "Apple M2 Max"
      return PREFILL_APPEND_COOLDOWN_MS unless start_pos == 0
      return PREFILL_APPEND_COOLDOWN_MS unless n_tokens == 1024 || n_tokens == 2048
      return PREFILL_APPEND_COOLDOWN_MS unless full_attention_interval == 4
      return PREFILL_APPEND_COOLDOWN_MS unless kv_cache_f16 && !resident_adaptive
      return PREFILL_APPEND_COOLDOWN_MS if checkpoint_requested || boundary_profile || graph_depth != 0

      # Full-logit/top-1 prefill handles the final full-attention layer through
      # its terminal-row kernel. Complete hidden-prefill routes stay guarded.
      measured_9b = n_layer == 32 && layer_limit == 31 &&
                    n_embd == 4096 && n_ff == 12288 &&
                    n_head == 16 && n_head_kv == 4 && head_dim == 256 &&
                    flash_d256
      measured_27b = n_layer == 64 && layer_limit == 63 &&
                     n_embd == 5120 && n_ff == 17408 &&
                     n_head == 24 && n_head_kv == 4 && head_dim == 256 &&
                     model_capability.qwen38? && gguf_file_type == QWEN38_Q4_K_M_FILE_TYPE
      measured_9b || measured_27b ? 0 : PREFILL_APPEND_COOLDOWN_MS
    end

    # The last command of a chunk is followed by the first command of the next
    # chunk, so it needs the same compositor window as an in-chunk rotation.
    def prefill_chunk_boundary_cooldown_ms(
      n_tokens : Int32,
      more_chunks : Bool,
      shared_command_completed : Bool,
      group_limit_configured : String? = ENV["QWEN35_PREFILL_APPEND_MAX_GROUPS"]?,
      cooldown_configured : String? = ENV["QWEN35_PREFILL_APPEND_COOLDOWN_MS"]?,
    ) : Int32
      return 0 unless more_chunks
      return 0 unless shared_command_completed
      return 0 if prefill_append_group_limit(n_tokens, group_limit_configured) == 0
      prefill_append_cooldown_ms(cooldown_configured)
    end

    # Default-off CogniGraph submission window. Exact and adaptive prefill use
    # the same FIFO queue; adaptive visibility is separately certified by its
    # per-command publication tickets. Checkpoints and GPU boundary profiling
    # still require synchronous completion semantics.
    def prefill_graph_max_inflight(
      adaptive_kv : Bool,
      checkpoint_requested : Bool,
      boundary_profile : Bool,
      configured : String? = ENV["QWEN35_COGNIGRAPH_PREFILL_MAX_INFLIGHT"]?,
    ) : Int32
      return 0 unless raw = configured
      value = raw.to_i?
      unless value && 0 <= value <= 2
        raise ArgumentError.new("QWEN35_COGNIGRAPH_PREFILL_MAX_INFLIGHT must be between 0 and 2")
      end
      return 0 if value == 0
      if checkpoint_requested
        raise ArgumentError.new("CogniGraph prefill enqueue does not support checkpoints")
      end
      if boundary_profile
        raise ArgumentError.new("CogniGraph prefill enqueue does not support boundary profiling")
      end
      value
    end

    def prefill_resident_top1_append_layer?(layer_index : Int32,
                                            layer_limit : Int32) : Bool
      layer_limit > 0 && layer_index == layer_limit - 1
    end

    private def with_prefill_scratch_arena(arena : PrefillScratchArena?, &)
      {% unless flag?(:cpu_only) %}
        if active = arena
          return active.with { yield }
        end
      {% end %}
      yield
    end

    {% unless flag?(:cpu_only) %}
      private def cleanup_prefill_command_setup(
        queue : ML::Metal::GraphSubmissionQueue(ML::Metal::CommandBuffer, Qwen35Metal::Scratch::Arena)?,
        command : ML::Metal::CommandBuffer?,
      ) : Nil
        if active_queue = queue
          active_queue.abort unless active_queue.failed?
        elsif active_command = command
          active_command.discard unless active_command.committed?
        end
      end
    {% end %}

    private def prefill_gc_guard_enabled? : Bool
      ENV["QWEN35_PREFILL_GC_GUARD_OFF"]? != "1"
    end

    private def with_prefill_gc_guard(&)
      return yield if @@prefill_gc_guard_active || !prefill_gc_guard_enabled?

      @@prefill_gc_guard_active = true
      GC.disable
      begin
        yield
      ensure
        GC.enable
        @@prefill_gc_guard_active = false
      end
    end

    # ─────────────────────────────────────────────────────────────────────
    # Per-sequence state: KV cache for full-attn layers + SSM state for
    # recurrent (DeltaNet) layers.
    # ─────────────────────────────────────────────────────────────────────
    class LayerState
      # Ordinary full-attention cache element type. This is authoritative:
      # buffer size alone must never select a Metal kernel ABI.
      getter kv_cache_f16 : Bool

      # Full-attn: KV cache. Populated position-by-position during decode.
      # k_cache[pos * kv_dim + h * head_dim + d], same for v.
      property k_cache : Array(Float32)?
      property v_cache : Array(Float32)?
      property position : Int32 = 0 # number of cached tokens

      # GPU-resident KV cache. Same layout as k_cache/v_cache (position-
      # major). When Metal is available the full-attn path writes K/V
      # straight into these buffers via `contents` (unified memory) and
      # dispatches the Metal attention kernel against them.
      property k_cache_buf : ML::MetalBuffer?
      property v_cache_buf : ML::MetalBuffer?
      property adaptive_kv : QwenQBitAdaptiveResidentKV::Cache?

      # DeltaNet: recurrent state
      # conv_state: [conv_kernel - 1, qkv_stream_dim] — past (kernel-1) token activations
      #   for the 1D conv (padded with zeros at start).
      # ssm_state: [num_v_heads, head_v_dim, head_k_dim] — the matrix-valued recurrent state.
      property conv_state : Array(Float32)?
      property conv_state_buf : ML::MetalBuffer?
      property ssm_state : Array(Float32)?

      # GPU-resident SSM state. Kept in parallel to the CPU `ssm_state`
      # but only one of the two is used per sequence — whichever matches
      # the backend dispatched on the first recurrent call. Persists
      # across decode steps so the DeltaNet kernel reads and writes it
      # in place.
      property ssm_state_buf : ML::MetalBuffer?

      def initialize(@kv_cache_f16 : Bool = false)
      end

      def kv_cache_element_bytes : Int64
        @kv_cache_f16 ? 2_i64 : sizeof(Float32).to_i64
      end

      def kv_cache_bytes(max_seq : Int32, kv_dim : Int32) : Int64
        max_seq.to_i64 * kv_dim.to_i64 * kv_cache_element_bytes
      end

      # Deep-copy per-layer decode state. This is the minimal primitive
      # needed by exact speculative verification: each branch must mutate
      # its own KV/SSM buffers, not alias the parent sequence.
      def fork : LayerState
        copy = LayerState.new(@kv_cache_f16)
        copy.copy_from!(self)
        copy
      end

      def copy_from!(src : LayerState) : Nil
        unless @kv_cache_f16 == src.kv_cache_f16
          raise ArgumentError.new("KV cache element type mismatch")
        end
        if @adaptive_kv || src.adaptive_kv
          raise ArgumentError.new("adaptive resident QBit KV fork/copy is unsupported")
        end
        @position = src.position
        @k_cache = src.k_cache.try(&.dup)
        @v_cache = src.v_cache.try(&.dup)
        @conv_state = src.conv_state.try(&.dup)
        @ssm_state = src.ssm_state.try(&.dup)
        @k_cache_buf = copy_buffer_from!(@k_cache_buf, src.k_cache_buf)
        @v_cache_buf = copy_buffer_from!(@v_cache_buf, src.v_cache_buf)
        @conv_state_buf = copy_buffer_from!(@conv_state_buf, src.conv_state_buf)
        @ssm_state_buf = copy_buffer_from!(@ssm_state_buf, src.ssm_state_buf)
      end

      private def copy_buffer_from!(dst : ML::MetalBuffer?, src : ML::MetalBuffer?) : ML::MetalBuffer?
        return nil unless src_buf = src

        dst_buf = dst
        if dst_buf.nil? || dst_buf.size != src_buf.size || dst_buf.storage_mode != src_buf.storage_mode
          dst_buf = ML::MetalBuffer.new(src_buf.size, src_buf.storage_mode)
        end
        dst_buf.copy_from(src_buf, src_buf.size)
        dst_buf
      end
    end

    private def physical_memory_bytes? : UInt64?
      {% if flag?(:darwin) %}
        command_u64?("sysctl", ["-n", "hw.memsize"])
      {% elsif flag?(:freebsd) %}
        command_u64?("sysctl", ["-n", "hw.physmem"]) || command_u64?("sysctl", ["-n", "hw.realmem"])
      {% elsif flag?(:linux) %}
        if File.exists?("/proc/meminfo")
          File.each_line("/proc/meminfo") do |line|
            next unless line.starts_with?("MemTotal:")
            parts = line.split
            if parts.size >= 2 && (kb = parts[1].to_u64?)
              return kb * 1024_u64
            end
          end
        end
        nil
      {% else %}
        nil
      {% end %}
    end

    private def command_u64?(cmd : String, args : Array(String)) : UInt64?
      output = IO::Memory.new
      error = IO::Memory.new
      status = Process.run(cmd, args, output: output, error: error)
      return nil unless status.success?
      output.to_s.strip.to_u64?
    rescue
      nil
    end

    class State
      getter layers : Array(LayerState)
      getter max_seq : Int32

      def initialize(hp : Qwen35Hparams, @max_seq : Int32 = 1024, kv_cache_f16 : Bool = false)
        @layers = Array(LayerState).new(hp.n_layer) { LayerState.new(kv_cache_f16) }
      end

      protected def initialize(@layers : Array(LayerState), @max_seq : Int32)
      end

      def fork : State
        ensure_adaptive_copyable!
        State.new(@layers.map(&.fork), @max_seq)
      end

      def copy_from!(src : State) : Nil
        raise ArgumentError.new("max_seq mismatch: #{@max_seq} != #{src.max_seq}") unless @max_seq == src.max_seq
        raise ArgumentError.new("layer count mismatch: #{@layers.size} != #{src.layers.size}") unless @layers.size == src.layers.size
        ensure_adaptive_copyable!
        src.ensure_adaptive_copyable!

        @layers.each_with_index do |layer, i|
          layer.copy_from!(src.layers[i])
        end
      end

      def adaptive_kv? : Bool
        @layers.any? { |layer| !layer.adaptive_kv.nil? }
      end

      def kv_cache_f16? : Bool
        selected = @layers.first?.try(&.kv_cache_f16) || false
        unless @layers.all? { |layer| layer.kv_cache_f16 == selected }
          raise ArgumentError.new("mixed KV cache element types in one state")
        end
        selected
      end

      def adaptive_kv_layer_indices : Array(Int32)
        indices = [] of Int32
        @layers.each_with_index do |layer, index|
          indices << index.to_i32 if layer.adaptive_kv
        end
        indices
      end

      protected def ensure_adaptive_copyable! : Nil
        if adaptive_kv?
          raise ArgumentError.new("adaptive resident QBit KV fork/copy is unsupported")
        end
      end
    end

    struct LayerTop1TraceRow
      getter layer : Int32
      getter top1 : Int32
      getter logit : Float32

      def initialize(@layer : Int32, @top1 : Int32, @logit : Float32)
      end
    end

    # Pre-allocate the GPU-resident state buffers used by Qwen35 Metal
    # prefill/decode. This keeps latency-sensitive prefill timing focused on
    # model work rather than first-touch MetalBuffer allocation and zeroing.
    #
    # K/V cache rows are overwritten before first use at start_pos=0, but the
    # buffers are cleared here to preserve strict fresh-state semantics for
    # callers that prepare a state ahead of time.
    def prepare_state_metal!(state : State,
                             hp : Qwen35Hparams,
                             clear : Bool = true,
                             admit_adaptive_resident_kv : Bool = true) : Nil
      {% if flag?(:cpu_only) %}
        return
      {% else %}
        return unless Qwen35Metal.available?

        adaptive_config = if admit_adaptive_resident_kv
                            adaptive_resident_kv_config(hp, state.max_seq)
                          end
        if adaptive_config && state.kv_cache_f16?
          raise ArgumentError.new("adaptive resident QBit KV cannot coexist with an F16 KV owner")
        end
        adaptive_indices = state.adaptive_kv_layer_indices
        if state.kv_cache_f16? && !adaptive_indices.empty?
          raise ArgumentError.new("adaptive resident QBit KV cannot coexist with an F16 KV owner")
        end
        if config = adaptive_config
          row_count64 = state.max_seq.to_i64 * hp.n_head_kv
          if row_count64 > Int32::MAX
            raise ArgumentError.new("adaptive resident QBit KV plan row count exceeds Int32")
          end
          if adaptive_indices.any?
            unless adaptive_indices.sort == config.keys.sort
              raise ArgumentError.new("adaptive resident QBit KV layer map cannot change after allocation")
            end
            config.each do |layer_index, tier|
              tiers = Array(QwenQBitAdaptiveKV::Tier).new(row_count64.to_i, tier)
              plan = QwenQBitAdaptiveKV.plan(tiers)
              expected_bytes = 2_i64 * plan.payload_bytes
              unless state.layers[layer_index].adaptive_kv.not_nil!.compressed_bytes == expected_bytes
                raise ArgumentError.new("adaptive resident QBit KV tier map cannot change after allocation")
              end
            end
          else
            config.each_key do |layer_index|
              selected_state = state.layers[layer_index]
              if selected_state.k_cache || selected_state.v_cache ||
                 selected_state.k_cache_buf || selected_state.v_cache_buf
                raise ArgumentError.new("adaptive resident QBit KV requires a fresh state without an F32 owner")
              end
            end
            allocated = {} of Int32 => QwenQBitAdaptiveResidentKV::Cache
            begin
              config.each do |layer_index, tier|
                tiers = Array(QwenQBitAdaptiveKV::Tier).new(row_count64.to_i, tier)
                plan = QwenQBitAdaptiveKV.plan(tiers)
                allocated[layer_index] = QwenQBitAdaptiveResidentKV.allocate(
                  plan, plan, state.max_seq, hp.n_head_kv, hp.head_dim,
                )
              end
            rescue ex
              allocated.each_value(&.release)
              raise ex
            end
            allocated.each do |layer_index, cache|
              state.layers[layer_index].adaptive_kv = cache
            end
            adaptive_indices = config.keys.sort
          end
        end

        if clear
          adaptive_indices.each do |layer_index|
            cache = state.layers[layer_index].adaptive_kv.not_nil!
            unless cache.cache_len == 0
              raise ArgumentError.new("adaptive resident QBit KV cannot clear a published cache in place")
            end
          end
        end

        kv_dim = hp.head_dim * hp.n_head_kv
        qkv_dim = 2 * hp.ssm_group_count * hp.ssm_state_size + hp.ssm_time_step_rank * hp.ssm_state_size
        conv_bytes = ((hp.ssm_conv_kernel - 1) * qkv_dim).to_i64 * sizeof(Float32)
        ssm_bytes = (hp.ssm_time_step_rank * hp.ssm_state_size * hp.ssm_state_size).to_i64 * sizeof(Float32)
        state.layers.each_with_index do |layer, il|
          layer.position = 0
          if hp.full_attention?(il)
            if adaptive_indices.includes?(il)
              if layer.k_cache || layer.v_cache || layer.k_cache_buf || layer.v_cache_buf
                raise ArgumentError.new("adaptive resident QBit KV cannot coexist with an F32 owner")
              end
            else
              if layer.kv_cache_f16 && (layer.k_cache || layer.v_cache)
                raise ArgumentError.new("F16 KV cache cannot coexist with F32 host cache arrays")
              end
              kv_bytes = layer.kv_cache_bytes(state.max_seq, kv_dim)
              layer.k_cache_buf ||= ML::MetalBuffer.new(kv_bytes)
              layer.v_cache_buf ||= ML::MetalBuffer.new(kv_bytes)
              unless layer.k_cache_buf.not_nil!.size == kv_bytes && layer.v_cache_buf.not_nil!.size == kv_bytes
                raise ArgumentError.new("KV cache buffer size does not match its declared element type")
              end
              if clear
                clear_metal_buffer(layer.k_cache_buf)
                clear_metal_buffer(layer.v_cache_buf)
              end
            end
          else
            layer.conv_state_buf ||= ML::MetalBuffer.new(conv_bytes)
            layer.ssm_state_buf ||= ML::MetalBuffer.new(ssm_bytes)
            if clear
              clear_metal_buffer(layer.conv_state_buf)
              clear_metal_buffer(layer.ssm_state_buf)
            end
          end
        end
      {% end %}
    end

    # Release every GPU-owned buffer in a sequence state before replacing it.
    # The device fence prevents unified-memory storage from being recycled
    # while an asynchronously submitted kernel is still retiring.
    def release_state_metal!(state : State) : Nil
      ML::Metal::Device.synchronize
      state.layers.each do |layer|
        layer.k_cache_buf.try(&.release)
        layer.v_cache_buf.try(&.release)
        layer.conv_state_buf.try(&.release)
        layer.ssm_state_buf.try(&.release)
        layer.adaptive_kv.try(&.release)
        layer.k_cache_buf = nil
        layer.v_cache_buf = nil
        layer.conv_state_buf = nil
        layer.ssm_state_buf = nil
        layer.adaptive_kv = nil
      end
    end

    private def adaptive_resident_kv_config(hp : Qwen35Hparams,
                                            max_seq : Int32) : Hash(Int32, QwenQBitAdaptiveKV::Tier)?
      layer_raw = ENV["QWEN35_ADAPTIVE_RESIDENT_KV_LAYER"]?
      tier_raw = ENV["QWEN35_ADAPTIVE_RESIDENT_KV_TIER"]?
      map_raw = ENV["QWEN35_ADAPTIVE_RESIDENT_KV_MAP"]?
      return nil unless layer_raw || tier_raw || map_raw
      if map_raw && (layer_raw || tier_raw)
        raise ArgumentError.new("QWEN35_ADAPTIVE_RESIDENT_KV_MAP cannot be combined with the single-layer selectors")
      end
      unless layer_raw && tier_raw
        unless map_raw
          raise ArgumentError.new("adaptive resident QBit KV requires both QWEN35_ADAPTIVE_RESIDENT_KV_LAYER and QWEN35_ADAPTIVE_RESIDENT_KV_TIER")
        end
      end
      unless max_seq > 0
        raise ArgumentError.new("adaptive resident QBit KV maximum sequence must be positive")
      end
      unless hp.head_dim == QwenQBitAdaptiveKV::ROW_VALUES
        raise ArgumentError.new("adaptive resident QBit KV requires head dimension 256")
      end
      unless hp.n_head_kv > 0 && hp.n_head % hp.n_head_kv == 0 && hp.n_head // hp.n_head_kv == 6
        raise ArgumentError.new("adaptive resident QBit KV requires Qwen3.8 GQA6 shape")
      end
      if ENV["QWEN35_PREFILL_APPEND_CMD_OFF"]? == "1" ||
         ENV["QWEN35_PREFILL_RESIDENT_BOUNDARY_OFF"]? == "1"
        raise ArgumentError.new("adaptive resident QBit KV requires the shared prefill command corridor")
      end
      if ENV["QWEN35_PREFILL_FUSE_FULL_REC_OFF"]? == "1" ||
         ENV["QWEN35_FULL_PREFILL_CHUNK_OFF"]? == "1" ||
         ENV["QWEN35_PREFILL_REC_RUN_OFF"]? == "1" ||
         ENV["QWEN35_PREFILL_CHUNK_OFF"]? == "1"
        raise ArgumentError.new("adaptive resident QBit KV requires fused chunked full+recurrent prefill")
      end

      if map = map_raw
        sections = map.split(';')
        unless sections.size.in?(1..2) && !sections[0].strip.empty?
          raise ArgumentError.new("adaptive resident QBit KV map must be DEFAULT_TIER[;LAYER=TIER,...]")
        end
        default_tier = adaptive_resident_kv_tier(sections[0].strip)
        config = {} of Int32 => QwenQBitAdaptiveKV::Tier
        hp.full_attention_layers.each { |layer_index| config[layer_index] = default_tier }
        if sections.size == 2
          overrides = sections[1].strip
          if overrides.empty?
            raise ArgumentError.new("adaptive resident QBit KV map overrides cannot be empty")
          end
          seen = Set(Int32).new
          overrides.split(',').each do |entry|
            layer_text, override_tier = entry.split('=', 2)
            unless layer_text && override_tier
              raise ArgumentError.new("adaptive resident QBit KV map entries must use LAYER=TIER")
            end
            layer_index = layer_text.strip.to_i32?
            unless layer_index && layer_index >= 0 && layer_index < hp.n_layer
              raise ArgumentError.new("adaptive resident QBit KV layer selector is out of range")
            end
            unless hp.full_attention?(layer_index)
              raise ArgumentError.new("adaptive resident QBit KV selected layer must be full-attention")
            end
            unless seen.add?(layer_index)
              raise ArgumentError.new("adaptive resident QBit KV map contains a duplicate layer")
            end
            config[layer_index] = adaptive_resident_kv_tier(override_tier.strip)
          end
        end
        return config
      end

      layer_index = layer_raw.not_nil!.to_i32?
      unless layer_index && layer_index >= 0 && layer_index < hp.n_layer
        raise ArgumentError.new("adaptive resident QBit KV layer selector is out of range")
      end
      unless hp.full_attention?(layer_index)
        raise ArgumentError.new("adaptive resident QBit KV selected layer must be full-attention")
      end
      {layer_index => adaptive_resident_kv_tier(tier_raw.not_nil!)}
    end

    # Stable request-known identity for the complete resident tier plan. Cache
    # lookup must separate raw-F32 artifacts and every adaptive tier map before
    # reading or restoring either representation.
    def adaptive_resident_kv_layout_id(hp : Qwen35Hparams,
                                       max_seq : Int32) : String?
      return nil unless config = adaptive_resident_kv_config(hp, max_seq)

      String.build do |io|
        io << "qkv-adaptive-qbit-v1|"
        config.keys.sort.each_with_index do |layer_index, index|
          io << ',' unless index == 0
          io << layer_index << '=' << config[layer_index].value
        end
      end
    end

    private def adaptive_resident_kv_tier(raw : String) : QwenQBitAdaptiveKV::Tier
      case raw.downcase
      when "p4"   then QwenQBitAdaptiveKV::Tier::P4
      when "p5"   then QwenQBitAdaptiveKV::Tier::P5
      when "bf16" then QwenQBitAdaptiveKV::Tier::BF16
      else
        raise ArgumentError.new("adaptive resident QBit KV tier must be p4, p5, or bf16")
      end
    end

    # Allocate only the recurrent portion of a Metal state. Recurrent
    # checkpoints never read or write full-attention KV from the checkpoint
    # object, so allocating a second max_seq-sized KV cache here wastes unified
    # memory and can turn a latency optimization into host memory pressure.
    def prepare_recurrent_state_metal!(state : State, hp : Qwen35Hparams, clear : Bool = false) : Nil
      {% if flag?(:cpu_only) %}
        return
      {% else %}
        return unless Qwen35Metal.available?

        qkv_dim = 2 * hp.ssm_group_count * hp.ssm_state_size + hp.ssm_time_step_rank * hp.ssm_state_size
        conv_bytes = ((hp.ssm_conv_kernel - 1) * qkv_dim).to_i64 * sizeof(Float32)
        ssm_bytes = (hp.ssm_time_step_rank * hp.ssm_state_size * hp.ssm_state_size).to_i64 * sizeof(Float32)

        state.layers.each_with_index do |layer, il|
          layer.position = 0
          next if hp.full_attention?(il)

          layer.conv_state_buf ||= ML::MetalBuffer.new(conv_bytes)
          layer.ssm_state_buf ||= ML::MetalBuffer.new(ssm_bytes)
          if clear
            clear_metal_buffer(layer.conv_state_buf)
            clear_metal_buffer(layer.ssm_state_buf)
          end
        end
      {% end %}
    end

    # A recurrent checkpoint is useful only when both capture and the later
    # suffix replay keep recurrent ownership on Metal. Checking this before
    # prefill turns unsupported weight layouts into an ordinary-prefill
    # fallback instead of a mid-request checkpoint exception.
    def recurrent_checkpoint_metal_supported?(weights : Qwen35Weights) : Bool
      {% if flag?(:cpu_only) %}
        false
      {% else %}
        return false unless Qwen35Metal.available?
        weights.layers.all? do |layer|
          case layer
          in Qwen35FullAttnWeights
            true
          in Qwen35RecurrentWeights
            metal_qw_supported?(layer.attn_qkv_qw) &&
              metal_qw_supported?(layer.attn_gate_qw) &&
              metal_qw_supported?(layer.ssm_alpha_qw) &&
              metal_qw_supported?(layer.ssm_beta_qw) &&
              metal_qw_supported?(layer.ssm_out_qw) &&
              metal_qw_supported?(layer.ffn_gate_qw) &&
              metal_qw_supported?(layer.ffn_up_qw) &&
              metal_qw_supported?(layer.ffn_down_qw)
          end
        end
      {% end %}
    end

    # Canonicalize the unused part of exact KV before persistence. Generation
    # may have written rows beyond the completed-transcript boundary; attention
    # ignores them via the persisted position, but zeroing prevents irrelevant
    # output-history bytes from entering the durable artifact.
    def clear_kv_tail_metal!(state : State, hp : Qwen35Hparams, live_tokens : Int32) : Nil
      if state.adaptive_kv?
        raise ArgumentError.new("adaptive resident KV tail clearing is unsupported")
      end
      raise ArgumentError.new("live KV token count is outside state capacity") unless live_tokens >= 0 && live_tokens <= state.max_seq
      {% if flag?(:cpu_only) %}
        raise "clear_kv_tail_metal requires Metal"
      {% else %}
        raise "clear_kv_tail_metal requires Metal" unless Qwen35Metal.available?
        ML::Metal::Dispatch.execute_blit do |enc|
          state.layers.each_with_index do |layer, il|
            next unless hp.full_attention?(il)
            row_bytes = (hp.head_dim * hp.n_head_kv).to_i64 * layer.kv_cache_element_bytes
            offset = live_tokens.to_i64 * row_bytes
            k_buf = layer.k_cache_buf.not_nil!
            v_buf = layer.v_cache_buf.not_nil!
            tail_bytes = k_buf.size - offset
            raise ArgumentError.new("KV tail offset exceeds buffer at layer #{il}") if tail_bytes < 0 || v_buf.size - offset != tail_bytes
            next if tail_bytes == 0
            raise ArgumentError.new("KV tail clear exceeds Int32 encoder limit") if offset > Int32::MAX || tail_bytes > Int32::MAX
            enc.fill_buffer(k_buf, 0_u8, offset.to_i32, tail_bytes.to_i32)
            enc.fill_buffer(v_buf, 0_u8, offset.to_i32, tail_bytes.to_i32)
          end
        end
      {% end %}
    end

    # Copy only the live GPU-resident decode state into an already prepared
    # destination state. This is the branch-state primitive needed by exact
    # tree/speculative verification: recurrent state is copied in full, while
    # full-attention KV copies are bounded by the caller-provided live token
    # count instead of the allocated max sequence.
    def copy_state_metal_used!(dst : State, src : State, hp : Qwen35Hparams,
                               used_tokens : Int32? = nil,
                               rec_only : Bool = false,
                               full_kv_capacity : Bool = false) : Nil
      if dst.adaptive_kv? || src.adaptive_kv?
        raise ArgumentError.new("adaptive resident KV fork/copy is unsupported")
      end
      {% if flag?(:cpu_only) %}
        dst.copy_from!(src)
      {% else %}
        unless Qwen35Metal.available?
          dst.copy_from!(src)
          return
        end

        ML::Metal::Dispatch.execute_blit do |enc|
          encode_state_metal_used_copy!(enc, dst, src, hp, used_tokens: used_tokens,
            rec_only: rec_only, full_kv_capacity: full_kv_capacity)
        end
      {% end %}
    end

    def copy_recurrent_conv_metal_used!(dst : State, src : State, hp : Qwen35Hparams) : Nil
      {% if flag?(:cpu_only) %}
        dst.copy_from!(src)
      {% else %}
        unless Qwen35Metal.available?
          dst.copy_from!(src)
          return
        end

        ML::Metal::Dispatch.execute_blit do |enc|
          src.layers.each_with_index do |src_layer, il|
            next if hp.full_attention?(il)
            dst_layer = dst.layers[il]
            src_conv = src_layer.conv_state_buf.not_nil!
            dst_conv = dst_layer.conv_state_buf.not_nil!
            enc.copy_buffer(src_conv, 0, dst_conv, 0, checked_blit_bytes(src_conv.size))
          end
        end
      {% end %}
    end

    # Transfer ownership of recurrent GPU state buffers from `src` into `dst`
    # without a blit. This is only valid when `src` is a disposable checkpoint
    # state and the caller has proven full-attention KV rows remain valid or
    # will be overwritten before use.
    def swap_recurrent_state_metal_buffers!(dst : State, src : State, hp : Qwen35Hparams) : Nil
      if dst.adaptive_kv? || src.adaptive_kv?
        raise ArgumentError.new("adaptive resident KV checkpoint swapping is unsupported")
      end
      {% if flag?(:cpu_only) %}
        dst.copy_from!(src)
      {% else %}
        unless Qwen35Metal.available?
          dst.copy_from!(src)
          return
        end

        raise ArgumentError.new("max_seq mismatch: #{dst.max_seq} != #{src.max_seq}") unless dst.max_seq == src.max_seq
        raise ArgumentError.new("layer count mismatch: #{dst.layers.size} != #{src.layers.size}") unless dst.layers.size == src.layers.size

        src.layers.each_with_index do |src_layer, il|
          dst_layer = dst.layers[il]
          dst_layer.position = src_layer.position
          next if hp.full_attention?(il)

          old_conv_buf = dst_layer.conv_state_buf
          old_ssm_buf = dst_layer.ssm_state_buf
          old_conv_state = dst_layer.conv_state
          old_ssm_state = dst_layer.ssm_state

          dst_layer.conv_state_buf = src_layer.conv_state_buf
          dst_layer.ssm_state_buf = src_layer.ssm_state_buf
          dst_layer.conv_state = src_layer.conv_state
          dst_layer.ssm_state = src_layer.ssm_state

          src_layer.conv_state_buf = old_conv_buf
          src_layer.ssm_state_buf = old_ssm_buf
          src_layer.conv_state = old_conv_state
          src_layer.ssm_state = old_ssm_state
        end
      {% end %}
    end

    def rollback_recurrent_ssm_metal_from_log!(state : State, log_state : State, hp : Qwen35Hparams) : Nil
      {% if flag?(:cpu_only) %}
        raise "rollback_recurrent_ssm_metal_from_log requires Metal"
      {% else %}
        raise "rollback_recurrent_ssm_metal_from_log requires Metal" unless Qwen35Metal.available?
        h_k = hp.ssm_group_count
        h_v = hp.ssm_time_step_rank
        s = hp.ssm_state_size
        state_bufs = [] of ML::MetalBuffer
        log_bufs = [] of ML::MetalBuffer
        state.layers.each_with_index do |layer, il|
          next if hp.full_attention?(il)
          state_bufs << layer.ssm_state_buf.not_nil!
          log_bufs << log_state.layers[il].ssm_state_buf.not_nil!
        end
        Qwen35Metal.rollback_delta_net_states_from_logs(state_bufs, log_bufs, h_k, h_v, s)
      {% end %}
    end

    def encode_state_metal_used_copy!(enc : ML::Metal::BlitEncoder,
                                      dst : State, src : State, hp : Qwen35Hparams,
                                      used_tokens : Int32? = nil,
                                      rec_only : Bool = false,
                                      full_kv_capacity : Bool = false) : Nil
      {% if flag?(:cpu_only) %}
        dst.copy_from!(src)
      {% else %}
        raise ArgumentError.new("max_seq mismatch: #{dst.max_seq} != #{src.max_seq}") unless dst.max_seq == src.max_seq
        raise ArgumentError.new("layer count mismatch: #{dst.layers.size} != #{src.layers.size}") unless dst.layers.size == src.layers.size

        src.layers.each_with_index do |src_layer, il|
          dst_layer = dst.layers[il]
          unless dst_layer.kv_cache_f16 == src_layer.kv_cache_f16
            raise ArgumentError.new("KV cache element type mismatch at layer #{il}")
          end
          dst_layer.position = src_layer.position

          if hp.full_attention?(il)
            next if rec_only

            src_k = src_layer.k_cache_buf.not_nil!
            src_v = src_layer.v_cache_buf.not_nil!
            dst_k = dst_layer.k_cache_buf.not_nil!
            dst_v = dst_layer.v_cache_buf.not_nil!
            live_tokens = used_tokens || src_layer.position
            kv_row_bytes = (hp.head_dim * hp.n_head_kv).to_i64 * src_layer.kv_cache_element_bytes
            bytes = full_kv_capacity ? src_k.size : live_tokens.to_i64 * kv_row_bytes
            raise ArgumentError.new("live KV bytes exceed source buffer at layer #{il}: #{bytes} > #{src_k.size}") if bytes > src_k.size || bytes > src_v.size
            raise ArgumentError.new("live KV bytes exceed destination buffer at layer #{il}: #{bytes} > #{dst_k.size}") if bytes > dst_k.size || bytes > dst_v.size
            next if bytes <= 0

            enc.copy_buffer(src_k, 0, dst_k, 0, checked_blit_bytes(bytes))
            enc.copy_buffer(src_v, 0, dst_v, 0, checked_blit_bytes(bytes))
          else
            src_conv = src_layer.conv_state_buf.not_nil!
            src_ssm = src_layer.ssm_state_buf.not_nil!
            dst_conv = dst_layer.conv_state_buf.not_nil!
            dst_ssm = dst_layer.ssm_state_buf.not_nil!
            enc.copy_buffer(src_conv, 0, dst_conv, 0, checked_blit_bytes(src_conv.size))
            enc.copy_buffer(src_ssm, 0, dst_ssm, 0, checked_blit_bytes(src_ssm.size))
          end
        end
      {% end %}
    end

    private def clear_metal_buffer(buf : ML::MetalBuffer?) : Nil
      {% if flag?(:cpu_only) %}
        return
      {% else %}
        return unless b = buf
        b.contents.as(Pointer(UInt8)).clear(b.size)
      {% end %}
    end

    private def checked_blit_bytes(bytes : Int64) : Int32
      raise ArgumentError.new("copy byte size exceeds Int32 encoder limit: #{bytes}") if bytes > Int32::MAX
      bytes.to_i32
    end

    # ─────────────────────────────────────────────────────────────────────
    # Primitives
    # ─────────────────────────────────────────────────────────────────────

    # RMSNorm: y[i] = x[i] * rsqrt(mean(x^2) + eps) * w[i]
    # Single token (dim-length vector). Returns new Array.
    def rms_norm(x : Array(Float32), w : Array(Float32), eps : Float32 = 1.0e-6_f32) : Array(Float32)
      dim = x.size
      ss = 0.0_f64
      dim.times { |j| ss += x[j].to_f64 * x[j].to_f64 }
      inv_rms = (1.0 / Math.sqrt(ss / dim.to_f64 + eps.to_f64)).to_f32
      Array(Float32).new(dim) { |j| x[j] * inv_rms * w[j] }
    end

    # In-place variant — avoids alloc for hot paths.
    def rms_norm!(x : Array(Float32), w : Array(Float32), eps : Float32 = 1.0e-6_f32) : Nil
      dim = x.size
      ss = 0.0_f64
      dim.times { |j| ss += x[j].to_f64 * x[j].to_f64 }
      inv_rms = (1.0 / Math.sqrt(ss / dim.to_f64 + eps.to_f64)).to_f32
      dim.times { |j| x[j] = x[j] * inv_rms * w[j] }
    end

    # RMSNorm on a per-head slice of a longer vector starting at `offset`,
    # with weight of size `head_dim`. Used for attn_q_norm / attn_k_norm
    # (one set of weights shared across heads).
    def rms_norm_slice!(x : Array(Float32), offset : Int32, len : Int32,
                        w : Array(Float32), eps : Float32 = 1.0e-6_f32) : Nil
      ss = 0.0_f64
      len.times { |j| ss += x[offset + j].to_f64 * x[offset + j].to_f64 }
      inv_rms = (1.0 / Math.sqrt(ss / len.to_f64 + eps.to_f64)).to_f32
      len.times { |j| x[offset + j] = x[offset + j] * inv_rms * w[j] }
    end

    # SiLU / Swish: x * sigmoid(x). Elementwise.
    def silu!(x : Array(Float32)) : Nil
      x.size.times { |i| v = x[i]; x[i] = v / (1.0_f32 + Math.exp(-v)) }
    end

    def silu(x : Float32) : Float32
      x / (1.0_f32 + Math.exp(-x))
    end

    # Sigmoid. Elementwise.
    def sigmoid!(x : Array(Float32)) : Nil
      x.size.times { |i| x[i] = 1.0_f32 / (1.0_f32 + Math.exp(-x[i])) }
    end

    def sigmoid(x : Float32) : Float32
      1.0_f32 / (1.0_f32 + Math.exp(-x))
    end

    # L2 normalize a slice [offset..offset+len).
    # y[i] = x[i] / sqrt(sum(x^2) + eps)
    def l2_norm_slice!(x : Array(Float32), offset : Int32, len : Int32, eps : Float32 = 1.0e-6_f32) : Nil
      ss = 0.0_f64
      len.times { |j| ss += x[offset + j].to_f64 * x[offset + j].to_f64 }
      inv_norm = (1.0 / Math.sqrt(ss + eps.to_f64)).to_f32
      len.times { |j| x[offset + j] = x[offset + j] * inv_norm }
    end

    # Partial M-RoPE (NeoX-style pairing) for a single head-sized vector [head_dim],
    # rotating only the first `n_rot` dims. Keeps dims [n_rot..head_dim) unchanged.
    #
    # NeoX pairing: pair is (dst[i], dst[i + n_rot/2]) for i in 0...n_rot/2
    # Per ggml-metal.metal:4505-4509:
    #   x0 = src[i]; x1 = src[i + n_rot/2]
    #   dst[i]           = x0*cos - x1*sin
    #   dst[i + n_rot/2] = x0*sin + x1*cos
    #
    # For text-only decoding all M-RoPE sections map to the same sequence position,
    # so theta_base = pos and standard RoPE frequencies apply on the first n_rot dims.
    def rope_partial!(x : Array(Float32), head_offset : Int32,
                      n_rot : Int32, head_dim : Int32,
                      pos : Int32, freq_base : Float32) : Nil
      half = n_rot // 2
      half.times do |i|
        freq = 1.0_f32 / (freq_base ** (2.0_f32 * i / n_rot))
        theta = pos.to_f32 * freq
        cos_t = Math.cos(theta)
        sin_t = Math.sin(theta)
        i0 = head_offset + i
        i1 = head_offset + i + half
        x0 = x[i0]; x1 = x[i1]
        x[i0] = x0 * cos_t - x1 * sin_t
        x[i1] = x0 * sin_t + x1 * cos_t
      end
      # dims [n_rot..head_dim) are passed through unchanged (no action needed)
    end

    # Softmax over a slice [offset..offset+len) in place.
    def softmax_slice!(x : Array(Float32), offset : Int32, len : Int32) : Nil
      maxv = x[offset]
      len.times { |i| v = x[offset + i]; maxv = v if v > maxv }
      sum = 0.0_f32
      len.times do |i|
        e = Math.exp(x[offset + i] - maxv)
        x[offset + i] = e
        sum += e
      end
      inv = 1.0_f32 / sum
      len.times { |i| x[offset + i] *= inv }
    end

    # Threshold: only use Metal when the op is large enough to amortize
    # upload/download overhead. Tiny matmuls (e.g. ssm_alpha 4096→32)
    # stay on CPU.
    METAL_QK_MIN_OUT = 256
    METAL_QK_MIN_IN  = 256

    private def metal_qw_supported?(qw : QuantWeight) : Bool
      qw.type.q4_k? || qw.type.q5_k? || qw.type.q6_k? || qw.type.q8_0? || qw.type.iq4_nl? || qw.type.f32?
    end

    private def metal_qw_eligible?(qw : QuantWeight) : Bool
      qw.out_dim >= METAL_QK_MIN_OUT && qw.in_dim >= METAL_QK_MIN_IN
    end

    # DeltaNet / GatedDeltaRule state-update + output, shared between the
    # CPU path and the Metal spec reference. `state` and `y` are written
    # in place. `ghead` is the per-head decay multiplier (i.e. caller
    # computes `exp(softplus(...) * ssm_a[h])` up front).
    def delta_net_step!(state : Array(Float32),
                        q_conv : Array(Float32),
                        k_conv : Array(Float32),
                        v_conv : Array(Float32),
                        ghead : Array(Float32),
                        beta : Array(Float32),
                        y : Array(Float32),
                        h_k : Int32, h_v : Int32, s : Int32,
                        scale : Float32) : Nil
      h_v.times do |h|
        k_head = h % h_k
        q_off = k_head * s
        k_off = k_head * s
        v_off = h * s
        st_base = h * s * s
        gh = ghead[h]
        bh = beta[h]

        (s * s).times { |i| state[st_base + i] *= gh }

        sk = Array(Float32).new(s) do |d2|
          row_off = st_base + d2 * s
          a = 0.0_f32
          s.times { |d1| a += state[row_off + d1] * k_conv[k_off + d1] }
          a
        end

        delt = Array(Float32).new(s) { |d2| bh * (v_conv[v_off + d2] - sk[d2]) }

        s.times do |d2|
          row_off = st_base + d2 * s
          dd = delt[d2]
          s.times { |d1| state[row_off + d1] += k_conv[k_off + d1] * dd }
        end

        y_off = h * s
        s.times do |d2|
          row_off = st_base + d2 * s
          acc = 0.0_f32
          s.times { |d1| acc += state[row_off + d1] * q_conv[q_off + d1] }
          y[y_off + d2] = acc * scale
        end
      end
    end

    # Route the DeltaNet step to Metal when available, CPU otherwise.
    # State persists across decode steps on whichever backend owns it:
    # Array(Float32) for CPU, MetalBuffer for GPU — never both at once.
    private def delta_net_step_routed(lstate : LayerState,
                                      q_conv : Array(Float32),
                                      k_conv : Array(Float32),
                                      v_conv : Array(Float32),
                                      ghead : Array(Float32),
                                      beta : Array(Float32),
                                      h_k : Int32, h_v : Int32, s : Int32,
                                      scale : Float32) : Array(Float32)
      {% unless flag?(:cpu_only) %}
        if Qwen35Metal.available?
          bytes = (h_v * s * s).to_i64 * sizeof(Float32)
          state_buf = lstate.ssm_state_buf
          if state_buf.nil?
            state_buf = ML::MetalBuffer.new(bytes)
            state_buf.contents.as(Pointer(UInt8)).clear(bytes)
            lstate.ssm_state_buf = state_buf
          end
          return Qwen35Metal.delta_net_step(state_buf, q_conv, k_conv, v_conv,
            ghead, beta, h_k, h_v, s, scale)
        end
      {% end %}

      state = lstate.ssm_state ||= Array(Float32).new(h_v * s * s, 0.0_f32)
      y = Array(Float32).new(h_v * s, 0.0_f32)
      delta_net_step!(state, q_conv, k_conv, v_conv, ghead, beta, y,
        h_k, h_v, s, scale)
      y
    end

    # Recurrent Metal fast path:
    #   delta_net_step -> RMSNorm(y)*silu(z) -> ssm_out projection
    # in one command buffer. Returns nil when disabled or unsupported.
    private def delta_net_project_routed(lstate : LayerState,
                                         q_conv : Array(Float32),
                                         k_conv : Array(Float32),
                                         v_conv : Array(Float32),
                                         ghead : Array(Float32),
                                         beta : Array(Float32),
                                         z : Array(Float32),
                                         ssm_norm : Array(Float32),
                                         out_qw : QuantWeight,
                                         h_k : Int32, h_v : Int32, s : Int32,
                                         scale : Float32,
                                         eps : Float32) : Array(Float32)?
      {% unless flag?(:cpu_only) %}
        return nil if ENV["QWEN35_DN_FUSE_OFF"]? == "1"
        return nil unless out_qw.type.q4_k? || out_qw.type.q5_k? || out_qw.type.q6_k?
        return nil unless Qwen35Metal.available?

        bytes = (h_v * s * s).to_i64 * sizeof(Float32)
        state_buf = lstate.ssm_state_buf
        if state_buf.nil?
          state_buf = ML::MetalBuffer.new(bytes)
          state_buf.contents.as(Pointer(UInt8)).clear(bytes)
          lstate.ssm_state_buf = state_buf
        end
        return Qwen35Metal.delta_net_project(
          state_buf,
          q_conv, k_conv, v_conv, ghead, beta, z, ssm_norm, out_qw,
          h_k, h_v, s, scale, eps,
        )
      {% else %}
        nil
      {% end %}
    end

    private def recurrent_attn_project_routed(lstate : LayerState,
                                              cur : Array(Float32),
                                              lw : Qwen35RecurrentWeights,
                                              hp : Qwen35Hparams) : Array(Float32)?
      {% unless flag?(:cpu_only) %}
        return nil if ENV["QWEN35_RECURRENT_FUSE_OFF"]? == "1"
        return nil unless Qwen35Metal.available?
        return nil unless metal_qw_supported?(lw.attn_qkv_qw) &&
                          metal_qw_supported?(lw.attn_gate_qw) &&
                          metal_qw_supported?(lw.ssm_alpha_qw) &&
                          metal_qw_supported?(lw.ssm_beta_qw) &&
                          metal_qw_supported?(lw.ssm_out_qw)

        qkv_dim = 2 * hp.ssm_group_count * hp.ssm_state_size + hp.ssm_time_step_rank * hp.ssm_state_size
        conv_bytes = ((hp.ssm_conv_kernel - 1) * qkv_dim).to_i64 * sizeof(Float32)
        conv_buf = lstate.conv_state_buf
        if conv_buf.nil?
          conv_buf = ML::MetalBuffer.new(conv_bytes)
          conv_buf.contents.as(Pointer(UInt8)).clear(conv_bytes)
          lstate.conv_state_buf = conv_buf
        end

        ssm_bytes = (hp.ssm_time_step_rank * hp.ssm_state_size * hp.ssm_state_size).to_i64 * sizeof(Float32)
        ssm_buf = lstate.ssm_state_buf
        if ssm_buf.nil?
          ssm_buf = ML::MetalBuffer.new(ssm_bytes)
          ssm_buf.contents.as(Pointer(UInt8)).clear(ssm_bytes)
          lstate.ssm_state_buf = ssm_buf
        end

        return Qwen35Metal.recurrent_attn_project(
          cur, conv_buf, ssm_buf,
          lw.attn_qkv_qw, lw.attn_gate_qw, lw.ssm_alpha_qw, lw.ssm_beta_qw,
          lw.ssm_conv1d, lw.ssm_dt_bias, lw.ssm_a, lw.ssm_norm, lw.ssm_out_qw,
          hp.ssm_group_count, hp.ssm_time_step_rank, hp.ssm_state_size, hp.ssm_conv_kernel, hp.rms_eps,
        )
      {% else %}
        nil
      {% end %}
    end

    # Fused recurrent-layer GPU route:
    #   recurrent attention -> residual add + post-attn RMSNorm -> FFN -> residual add
    private def recurrent_layer_project_routed(inpSA : Array(Float32),
                                               cur : Array(Float32),
                                               lstate : LayerState,
                                               lw : Qwen35RecurrentWeights,
                                               hp : Qwen35Hparams) : Array(Float32)?
      {% unless flag?(:cpu_only) %}
        return nil if ENV["QWEN35_RECURRENT_LAYER_FUSE_OFF"]? == "1"
        return nil unless Qwen35Metal.available?
        supported = metal_qw_supported?(lw.attn_qkv_qw) &&
                    metal_qw_supported?(lw.attn_gate_qw) &&
                    metal_qw_supported?(lw.ssm_alpha_qw) &&
                    metal_qw_supported?(lw.ssm_beta_qw) &&
                    metal_qw_supported?(lw.ssm_out_qw) &&
                    metal_qw_supported?(lw.ffn_gate_qw) &&
                    metal_qw_supported?(lw.ffn_up_qw) &&
                    metal_qw_supported?(lw.ffn_down_qw)
        return nil unless supported

        qkv_dim = 2 * hp.ssm_group_count * hp.ssm_state_size + hp.ssm_time_step_rank * hp.ssm_state_size
        conv_bytes = ((hp.ssm_conv_kernel - 1) * qkv_dim).to_i64 * sizeof(Float32)
        conv_buf = lstate.conv_state_buf
        if conv_buf.nil?
          conv_buf = ML::MetalBuffer.new(conv_bytes)
          conv_buf.contents.as(Pointer(UInt8)).clear(conv_bytes)
          lstate.conv_state_buf = conv_buf
        end

        ssm_bytes = (hp.ssm_time_step_rank * hp.ssm_state_size * hp.ssm_state_size).to_i64 * sizeof(Float32)
        ssm_buf = lstate.ssm_state_buf
        if ssm_buf.nil?
          ssm_buf = ML::MetalBuffer.new(ssm_bytes)
          ssm_buf.contents.as(Pointer(UInt8)).clear(ssm_bytes)
          lstate.ssm_state_buf = ssm_buf
        end

        return Qwen35Metal.recurrent_layer_project(
          inpSA, cur, conv_buf, ssm_buf,
          lw.attn_qkv_qw, lw.attn_gate_qw, lw.ssm_alpha_qw, lw.ssm_beta_qw,
          lw.ssm_conv1d, lw.ssm_dt_bias, lw.ssm_a, lw.ssm_norm, lw.ssm_out_qw,
          lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
          hp.ssm_group_count, hp.ssm_time_step_rank, hp.ssm_state_size, hp.ssm_conv_kernel, hp.rms_eps,
        )
      {% else %}
        nil
      {% end %}
    end

    # Fused FFN route on Metal:
    #   gate_proj + up_proj -> swiglu -> down_proj
    private def ffn_project_routed(x : Array(Float32),
                                   gate_qw : QuantWeight,
                                   up_qw : QuantWeight,
                                   down_qw : QuantWeight) : Array(Float32)?
      {% unless flag?(:cpu_only) %}
        return nil if ENV["QWEN35_FFN_FUSE_OFF"]? == "1"
        return nil unless metal_qw_supported?(gate_qw) && metal_qw_supported?(up_qw) && metal_qw_supported?(down_qw)
        return nil unless Qwen35Metal.available?
        return Qwen35Metal.ffn_project(x, gate_qw, up_qw, down_qw)
      {% else %}
        nil
      {% end %}
    end

    # Append K, V to the layer's KV cache at `pos` and run gated GQA
    # attention. On Metal: writes K/V into persistent MetalBuffers via
    # unified-memory `contents` and dispatches `Qwen35Metal.attn_decode`.
    # On CPU: allocates the Array caches lazily and runs the per-head
    # dot-product / softmax / V-weighted-sum + sigmoid(gate) multiply.
    # Returns gated attention output of length `n_head * head_dim`.
    private def attn_decode_routed(lstate : LayerState,
                                   q : Array(Float32), gate : Array(Float32),
                                   k : Array(Float32), v : Array(Float32),
                                   pos : Int32, n_head : Int32, n_head_kv : Int32,
                                   head_dim : Int32, heads_per_group : Int32,
                                   kv_dim : Int32, max_seq : Int32,
                                   scale : Float32) : Array(Float32)
      if lstate.kv_cache_f16
        raise ArgumentError.new("F16 KV cache requires the whole-wave Metal decode route")
      end
      q_dim = n_head * head_dim
      base = pos * kv_dim

      {% unless flag?(:cpu_only) %}
        if Qwen35Metal.available? && ENV["QWEN35_ATTN_CPU"]? != "1"
          bytes = (max_seq * kv_dim).to_i64 * sizeof(Float32)
          k_buf = lstate.k_cache_buf
          v_buf = lstate.v_cache_buf
          if k_buf.nil?
            k_buf = ML::MetalBuffer.new(bytes)
            k_buf.contents.as(Pointer(UInt8)).clear(bytes)
            lstate.k_cache_buf = k_buf
          end
          if v_buf.nil?
            v_buf = ML::MetalBuffer.new(bytes)
            v_buf.contents.as(Pointer(UInt8)).clear(bytes)
            lstate.v_cache_buf = v_buf
          end
          k_ptr = k_buf.contents.as(Pointer(Float32)) + base
          v_ptr = v_buf.contents.as(Pointer(Float32)) + base
          kv_dim.times do |i|
            k_ptr[i] = k[i]
            v_ptr[i] = v[i]
          end
          return Qwen35Metal.attn_decode(q, gate, k_buf, v_buf,
            pos, n_head, n_head_kv, head_dim,
            heads_per_group, scale)
        end
      {% end %}

      k_cache = lstate.k_cache ||= Array(Float32).new(max_seq * kv_dim, 0.0_f32)
      v_cache = lstate.v_cache ||= Array(Float32).new(max_seq * kv_dim, 0.0_f32)
      kv_dim.times do |i|
        k_cache[base + i] = k[i]
        v_cache[base + i] = v[i]
      end

      attn_o = Array(Float32).new(q_dim, 0.0_f32)
      scores = Array(Float32).new(pos + 1, 0.0_f32)
      n_head.times do |h|
        kv_h = h // heads_per_group
        q_off = h * head_dim
        (pos + 1).times do |p|
          k_off = p * kv_dim + kv_h * head_dim
          s = 0.0_f32
          head_dim.times { |d| s += q[q_off + d] * k_cache[k_off + d] }
          scores[p] = s * scale
        end
        softmax_slice!(scores, 0, pos + 1)
        out_off = h * head_dim
        (pos + 1).times do |p|
          v_off = p * kv_dim + kv_h * head_dim
          w = scores[p]
          head_dim.times { |d| attn_o[out_off + d] += w * v_cache[v_off + d] }
        end
      end
      q_dim.times { |i| attn_o[i] = attn_o[i] * sigmoid(gate[i]) }
      attn_o
    end

    # Same routing as `attn_decode_routed`, but keeps the attention output on
    # GPU and immediately runs the output projection there. This is an
    # optimization-only path for full-attention decode; CPU fallback remains
    # the source of truth.
    private def attn_decode_project_routed(lstate : LayerState,
                                           q : Array(Float32), gate : Array(Float32),
                                           k : Array(Float32), v : Array(Float32),
                                           out_qw : QuantWeight,
                                           pos : Int32, n_head : Int32, n_head_kv : Int32,
                                           head_dim : Int32, heads_per_group : Int32,
                                           kv_dim : Int32, max_seq : Int32,
                                           scale : Float32) : Array(Float32)?
      if lstate.kv_cache_f16
        raise ArgumentError.new("F16 KV cache requires the whole-wave Metal decode route")
      end
      {% unless flag?(:cpu_only) %}
        return nil if ENV["QWEN35_ATTN_CPU"]? == "1"
        return nil if ENV["QWEN35_ATTN_FUSE_OFF"]? == "1"
        return nil unless out_qw.type.q4_k? || out_qw.type.q5_k? || out_qw.type.q6_k?
        return nil unless Qwen35Metal.available?

        base = pos * kv_dim
        bytes = lstate.kv_cache_bytes(max_seq, kv_dim)
        k_buf = lstate.k_cache_buf
        v_buf = lstate.v_cache_buf
        if k_buf.nil?
          k_buf = ML::MetalBuffer.new(bytes)
          k_buf.contents.as(Pointer(UInt8)).clear(bytes)
          lstate.k_cache_buf = k_buf
        end
        if v_buf.nil?
          v_buf = ML::MetalBuffer.new(bytes)
          v_buf.contents.as(Pointer(UInt8)).clear(bytes)
          lstate.v_cache_buf = v_buf
        end
        k_ptr = k_buf.contents.as(Pointer(Float32)) + base
        v_ptr = v_buf.contents.as(Pointer(Float32)) + base
        kv_dim.times do |i|
          k_ptr[i] = k[i]
          v_ptr[i] = v[i]
        end
        return Qwen35Metal.attn_decode_project(
          q, gate, k_buf, v_buf, out_qw,
          pos, n_head, n_head_kv, head_dim, heads_per_group, scale,
        )
      {% else %}
        nil
      {% end %}
    end

    # Full-attention GPU route:
    #   qkv projections -> split/norm/rope -> kv write -> attn -> out proj
    private def full_attn_layer_project_routed(inpSA : Array(Float32),
                                               cur : Array(Float32),
                                               lstate : LayerState,
                                               lw : Qwen35FullAttnWeights,
                                               hp : Qwen35Hparams,
                                               pos : Int32,
                                               heads_per_group : Int32,
                                               kv_dim : Int32,
                                               max_seq : Int32) : Array(Float32)?
      if lstate.kv_cache_f16
        raise ArgumentError.new("F16 KV cache requires the whole-wave Metal decode route")
      end
      {% unless flag?(:cpu_only) %}
        return nil if ENV["QWEN35_FULL_LAYER_FUSE_OFF"]? == "1"
        return nil unless Qwen35Metal.available?
        supported = metal_qw_supported?(lw.attn_q_qw) &&
                    metal_qw_supported?(lw.attn_k_qw) &&
                    metal_qw_supported?(lw.attn_v_qw) &&
                    metal_qw_supported?(lw.attn_output_qw) &&
                    metal_qw_supported?(lw.ffn_gate_qw) &&
                    metal_qw_supported?(lw.ffn_up_qw) &&
                    metal_qw_supported?(lw.ffn_down_qw)
        return nil unless supported

        bytes = lstate.kv_cache_bytes(max_seq, kv_dim)
        k_buf = lstate.k_cache_buf
        v_buf = lstate.v_cache_buf
        if k_buf.nil?
          k_buf = ML::MetalBuffer.new(bytes)
          k_buf.contents.as(Pointer(UInt8)).clear(bytes)
          lstate.k_cache_buf = k_buf
        end
        if v_buf.nil?
          v_buf = ML::MetalBuffer.new(bytes)
          v_buf.contents.as(Pointer(UInt8)).clear(bytes)
          lstate.v_cache_buf = v_buf
        end

        scale = (1.0 / Math.sqrt(hp.head_dim.to_f64)).to_f32
        return Qwen35Metal.full_attn_layer_project(
          inpSA, cur,
          lw.attn_q_qw, lw.attn_k_qw, lw.attn_v_qw,
          lw.attn_q_norm, lw.attn_k_norm, lw.attn_output_qw,
          k_buf, v_buf,
          lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
          pos, hp.n_head, hp.n_head_kv, hp.head_dim, hp.rope_dim_count,
          heads_per_group, hp.rope_freq_base, hp.rms_eps, scale,
        )
      {% else %}
        nil
      {% end %}
    end

    private def full_attn_layer_chunk_project_routed(inp : Array(Float32),
                                                     n_tokens : Int32,
                                                     start_pos : Int32,
                                                     lstate : LayerState,
                                                     lw : Qwen35FullAttnWeights,
                                                     hp : Qwen35Hparams,
                                                     max_seq : Int32,
                                                     read_output : Bool = true,
                                                     output_buf : ML::MetalBuffer? = nil,
                                                     input_buf : ML::MetalBuffer? = nil,
                                                     append_command_buffer : PrefillCommandBuffer? = nil,
                                                     pending_adaptive_caches : Array(QwenQBitAdaptiveResidentKV::Cache)? = nil,
                                                     scratch_arena : PrefillScratchArena? = nil) : Array(Float32)?
      {% unless flag?(:cpu_only) %}
        return nil if ENV["QWEN35_FULL_PREFILL_CHUNK_OFF"]? == "1"
        return nil unless Qwen35Metal.available?
        supported = metal_qw_supported?(lw.attn_q_qw) &&
                    metal_qw_supported?(lw.attn_k_qw) &&
                    metal_qw_supported?(lw.attn_v_qw) &&
                    metal_qw_supported?(lw.attn_output_qw) &&
                    metal_qw_supported?(lw.ffn_gate_qw) &&
                    metal_qw_supported?(lw.ffn_up_qw) &&
                    metal_qw_supported?(lw.ffn_down_qw)
        return nil unless supported

        kv_dim = hp.head_dim * hp.n_head_kv
        bytes = lstate.kv_cache_bytes(max_seq, kv_dim)
        adaptive_cache = lstate.adaptive_kv
        k_buf = nil.as(ML::MetalBuffer?)
        v_buf = nil.as(ML::MetalBuffer?)
        adaptive_encoder = nil.as(AdaptivePrefillEncoder?)
        unless adaptive_cache.nil?
          selected_cache = adaptive_cache.as(QwenQBitAdaptiveResidentKV::Cache)
          unless append_command_buffer && pending_adaptive_caches
            raise ArgumentError.new("adaptive resident QBit KV requires shared-command publication ownership")
          end
          if lstate.k_cache || lstate.v_cache || lstate.k_cache_buf || lstate.v_cache_buf
            raise ArgumentError.new("adaptive resident QBit KV cannot coexist with an F32 owner")
          end
          unless selected_cache.cache_len == start_pos
            raise ArgumentError.new("adaptive resident QBit KV prefill start does not match the live prefix")
          end
          adaptive_encoder = ->(command : ML::Metal::CommandBuffer, q_source : ML::MetalBuffer, gate_source : ML::MetalBuffer, k_source : ML::MetalBuffer, v_source : ML::MetalBuffer, output : ML::MetalBuffer) do
            QwenQBitAdaptiveResidentKV.encode_prefill_chunk_and_append(
              command, selected_cache,
              q_source, gate_source, k_source, v_source, output,
              n_tokens, hp.n_head, hp.n_head // hp.n_head_kv,
              (1.0 / Math.sqrt(hp.head_dim.to_f64)).to_f32,
              expected_start_token: start_pos,
            )
          end
        else
          k_buf = lstate.k_cache_buf
          v_buf = lstate.v_cache_buf
          if k_buf.nil?
            k_buf = ML::MetalBuffer.new(bytes)
            k_buf.contents.as(Pointer(UInt8)).clear(bytes)
            lstate.k_cache_buf = k_buf
          end
          if v_buf.nil?
            v_buf = ML::MetalBuffer.new(bytes)
            v_buf.contents.as(Pointer(UInt8)).clear(bytes)
            lstate.v_cache_buf = v_buf
          end
        end

        scale = (1.0 / Math.sqrt(hp.head_dim.to_f64)).to_f32
        begin
          result = with_prefill_scratch_arena(scratch_arena) do
            Qwen35Metal.full_attn_layer_chunk_project(
              inp,
              lw.attn_q_qw, lw.attn_k_qw, lw.attn_v_qw,
              lw.attn_norm, lw.attn_q_norm, lw.attn_k_norm, lw.attn_output_qw,
              k_buf, v_buf,
              lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
              start_pos, n_tokens,
              hp.n_head, hp.n_head_kv, hp.head_dim, hp.rope_dim_count,
              hp.n_head // hp.n_head_kv, hp.rope_freq_base, hp.rms_eps, scale,
              read_output: read_output,
              output_buf: output_buf,
              input_buf: input_buf,
              append_command_buffer: append_command_buffer,
              adaptive_prefill_encoder: adaptive_encoder,
              kv_cache_f16: lstate.kv_cache_f16,
            )
          end
        rescue ex
          if failed_cache = adaptive_cache
            if cmd = append_command_buffer
              QwenQBitAdaptiveResidentKV.cancel_pending_append!(failed_cache, cmd) unless cmd.committed?
            end
          end
          raise ex
        end
        if published_cache = adaptive_cache
          unless result
            QwenQBitAdaptiveResidentKV.cancel_pending_append!(published_cache, append_command_buffer.not_nil!)
            raise ArgumentError.new("adaptive resident QBit KV full-attention prefill route declined after allocation")
          end
          pending_adaptive_caches.not_nil! << published_cache
        end
        result
      {% else %}
        nil
      {% end %}
    end

    private def final_full_attn_layer_chunk_last_routed(inp : Array(Float32),
                                                        n_tokens : Int32,
                                                        start_pos : Int32,
                                                        lstate : LayerState,
                                                        lw : Qwen35FullAttnWeights,
                                                        hp : Qwen35Hparams,
                                                        max_seq : Int32,
                                                        input_buf : ML::MetalBuffer? = nil) : Array(Float32)?
      {% unless flag?(:cpu_only) %}
        return nil if ENV["QWEN35_FINAL_FULL_LAST_OFF"]? == "1"
        return nil unless Qwen35Metal.available?
        supported = metal_qw_supported?(lw.attn_q_qw) &&
                    metal_qw_supported?(lw.attn_k_qw) &&
                    metal_qw_supported?(lw.attn_v_qw) &&
                    metal_qw_supported?(lw.attn_output_qw) &&
                    metal_qw_supported?(lw.ffn_gate_qw) &&
                    metal_qw_supported?(lw.ffn_up_qw) &&
                    metal_qw_supported?(lw.ffn_down_qw)
        return nil unless supported

        kv_dim = hp.head_dim * hp.n_head_kv
        bytes = lstate.kv_cache_bytes(max_seq, kv_dim)
        k_buf = lstate.k_cache_buf
        v_buf = lstate.v_cache_buf
        if k_buf.nil?
          k_buf = ML::MetalBuffer.new(bytes)
          k_buf.contents.as(Pointer(UInt8)).clear(bytes)
          lstate.k_cache_buf = k_buf
        end
        if v_buf.nil?
          v_buf = ML::MetalBuffer.new(bytes)
          v_buf.contents.as(Pointer(UInt8)).clear(bytes)
          lstate.v_cache_buf = v_buf
        end

        scale = (1.0 / Math.sqrt(hp.head_dim.to_f64)).to_f32
        Qwen35Metal.full_attn_layer_chunk_project_last(
          inp,
          lw.attn_q_qw, lw.attn_k_qw, lw.attn_v_qw,
          lw.attn_norm, lw.attn_q_norm, lw.attn_k_norm, lw.attn_output_qw,
          k_buf, v_buf,
          lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
          start_pos, n_tokens,
          hp.n_head, hp.n_head_kv, hp.head_dim, hp.rope_dim_count,
          hp.n_head // hp.n_head_kv, hp.rope_freq_base, hp.rms_eps, scale,
          input_buf: input_buf,
          kv_cache_f16: lstate.kv_cache_f16,
        )
      {% else %}
        nil
      {% end %}
    end

    private def final_full_attn_layer_chunk_last_top1_routed(inp : Array(Float32),
                                                             n_tokens : Int32,
                                                             start_pos : Int32,
                                                             lstate : LayerState,
                                                             lw : Qwen35FullAttnWeights,
                                                             output_norm : Array(Float32),
                                                             output_qw : QuantWeight,
                                                             hp : Qwen35Hparams,
                                                             max_seq : Int32,
                                                             input_buf : ML::MetalBuffer? = nil) : {Int32, Float32}?
      {% unless flag?(:cpu_only) %}
        return nil if ENV["QWEN35_FINAL_FULL_LAST_OFF"]? == "1"
        return nil unless Qwen35Metal.available?
        supported = metal_qw_supported?(lw.attn_q_qw) &&
                    metal_qw_supported?(lw.attn_k_qw) &&
                    metal_qw_supported?(lw.attn_v_qw) &&
                    metal_qw_supported?(lw.attn_output_qw) &&
                    metal_qw_supported?(lw.ffn_gate_qw) &&
                    metal_qw_supported?(lw.ffn_up_qw) &&
                    metal_qw_supported?(lw.ffn_down_qw) &&
                    metal_qw_supported?(output_qw)
        return nil unless supported

        kv_dim = hp.head_dim * hp.n_head_kv
        bytes = lstate.kv_cache_bytes(max_seq, kv_dim)
        k_buf = lstate.k_cache_buf
        v_buf = lstate.v_cache_buf
        if k_buf.nil?
          k_buf = ML::MetalBuffer.new(bytes)
          k_buf.contents.as(Pointer(UInt8)).clear(bytes)
          lstate.k_cache_buf = k_buf
        end
        if v_buf.nil?
          v_buf = ML::MetalBuffer.new(bytes)
          v_buf.contents.as(Pointer(UInt8)).clear(bytes)
          lstate.v_cache_buf = v_buf
        end

        scale = (1.0 / Math.sqrt(hp.head_dim.to_f64)).to_f32
        Qwen35Metal.full_attn_layer_chunk_project_last_top1(
          inp,
          lw.attn_q_qw, lw.attn_k_qw, lw.attn_v_qw,
          lw.attn_norm, lw.attn_q_norm, lw.attn_k_norm, lw.attn_output_qw,
          k_buf, v_buf,
          lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
          start_pos, n_tokens,
          hp.n_head, hp.n_head_kv, hp.head_dim, hp.rope_dim_count,
          hp.n_head // hp.n_head_kv, hp.rope_freq_base, hp.rms_eps, scale,
          output_norm, output_qw,
          input_buf: input_buf,
          kv_cache_f16: lstate.kv_cache_f16,
        )
      {% else %}
        nil
      {% end %}
    end

    private def final_full_attn_layer_chunk_kv_cache_only_routed(inp : Array(Float32),
                                                                 n_tokens : Int32,
                                                                 start_pos : Int32,
                                                                 lstate : LayerState,
                                                                 lw : Qwen35FullAttnWeights,
                                                                 hp : Qwen35Hparams,
                                                                 max_seq : Int32,
                                                                 input_buf : ML::MetalBuffer? = nil,
                                                                 append_command_buffer : PrefillCommandBuffer? = nil,
                                                                 scratch_arena : PrefillScratchArena? = nil) : Bool
      {% unless flag?(:cpu_only) %}
        return false if ENV["QWEN35_PREFILL_FINAL_KV_ONLY_OFF"]? == "1"
        return false if ENV["QWEN35_FINAL_FULL_LAST_OFF"]? == "1"
        return false if ENV["QWEN35_FULL_PREFILL_CHUNK_OFF"]? == "1"
        return false unless Qwen35Metal.available?
        supported = metal_qw_supported?(lw.attn_k_qw) &&
                    metal_qw_supported?(lw.attn_v_qw)
        return false unless supported

        kv_dim = hp.head_dim * hp.n_head_kv
        bytes = lstate.kv_cache_bytes(max_seq, kv_dim)
        k_buf = lstate.k_cache_buf
        v_buf = lstate.v_cache_buf
        if k_buf.nil?
          k_buf = ML::MetalBuffer.new(bytes)
          k_buf.contents.as(Pointer(UInt8)).clear(bytes)
          lstate.k_cache_buf = k_buf
        end
        if v_buf.nil?
          v_buf = ML::MetalBuffer.new(bytes)
          v_buf.contents.as(Pointer(UInt8)).clear(bytes)
          lstate.v_cache_buf = v_buf
        end

        with_prefill_scratch_arena(scratch_arena) do
          Qwen35Metal.full_attn_layer_chunk_kv_cache_only(
            inp,
            lw.attn_k_qw, lw.attn_v_qw,
            lw.attn_norm, lw.attn_k_norm,
            k_buf, v_buf,
            start_pos, n_tokens,
            hp.n_head_kv, hp.head_dim, hp.rope_dim_count,
            hp.rope_freq_base, hp.rms_eps,
            input_buf: input_buf,
            append_command_buffer: append_command_buffer,
            kv_cache_f16: lstate.kv_cache_f16,
          )
        end
      {% else %}
        false
      {% end %}
    end

    private def full_attn_then_recurrent_chunk_project_many_routed(inp : Array(Float32),
                                                                   n_tokens : Int32,
                                                                   start_pos : Int32,
                                                                   state : State,
                                                                   weights : Qwen35Weights,
                                                                   il : Int32,
                                                                   hp : Qwen35Hparams,
                                                                   max_seq : Int32,
                                                                   checkpoint_index : Int32? = nil,
                                                                   checkpoint_state : State? = nil,
                                                                   checkpoint_rollback_log : Bool = false,
                                                                   input_buf : ML::MetalBuffer? = nil,
                                                                   output_buf : ML::MetalBuffer? = nil,
                                                                   read_output : Bool = true,
                                                                   append_command_buffer : PrefillCommandBuffer? = nil,
                                                                   pending_adaptive_caches : Array(QwenQBitAdaptiveResidentKV::Cache)? = nil,
                                                                   scratch_arena : PrefillScratchArena? = nil) : {Array(Float32), Int32}?
      {% unless flag?(:cpu_only) %}
        return nil if ENV["QWEN35_PREFILL_FUSE_FULL_REC_OFF"]? == "1"
        return nil if ENV["QWEN35_FULL_PREFILL_CHUNK_OFF"]? == "1"
        return nil if ENV["QWEN35_PREFILL_REC_RUN_OFF"]? == "1"
        return nil unless Qwen35Metal.available?
        full_lw = weights.layers[il].as?(Qwen35FullAttnWeights)
        return nil unless full_lw

        run_start = il + 1
        return nil if run_start >= weights.layers.size
        run_end = run_start
        while run_end < weights.layers.size
          break unless weights.layers[run_end].is_a?(Qwen35RecurrentWeights)
          run_end += 1
        end
        return nil if run_end == run_start

        full_state = state.layers[il]
        adaptive_cache = full_state.adaptive_kv
        if adaptive_cache
          if checkpoint_index || checkpoint_state
            raise ArgumentError.new("adaptive resident QBit KV checkpoint is unsupported")
          end
          unless append_command_buffer && pending_adaptive_caches
            raise ArgumentError.new("adaptive resident QBit KV requires shared-command publication ownership")
          end
        end

        supported = metal_qw_supported?(full_lw.attn_q_qw) &&
                    metal_qw_supported?(full_lw.attn_k_qw) &&
                    metal_qw_supported?(full_lw.attn_v_qw) &&
                    metal_qw_supported?(full_lw.attn_output_qw) &&
                    metal_qw_supported?(full_lw.ffn_gate_qw) &&
                    metal_qw_supported?(full_lw.ffn_up_qw) &&
                    metal_qw_supported?(full_lw.ffn_down_qw)
        unless supported
          if adaptive_cache
            raise ArgumentError.new("adaptive resident QBit KV fused prefill route is unavailable")
          end
          return nil
        end

        rec_layers = [] of Qwen35RecurrentWeights
        conv_bufs = [] of ML::MetalBuffer
        ssm_bufs = [] of ML::MetalBuffer
        checkpoint_requested = !checkpoint_index.nil?
        checkpoint_conv_bufs = [] of ML::MetalBuffer
        checkpoint_ssm_bufs = [] of ML::MetalBuffer
        if checkpoint_requested
          return nil if checkpoint_state.nil?
        end
        h_k = hp.ssm_group_count
        h_v = hp.ssm_time_step_rank
        s = hp.ssm_state_size
        qkv_dim = 2 * h_k * s + h_v * s
        conv_k = hp.ssm_conv_kernel

        j = run_start
        while j < run_end
          rw = weights.layers[j].as(Qwen35RecurrentWeights)
          supported &&= metal_qw_supported?(rw.attn_qkv_qw) &&
                        metal_qw_supported?(rw.attn_gate_qw) &&
                        metal_qw_supported?(rw.ssm_alpha_qw) &&
                        metal_qw_supported?(rw.ssm_beta_qw) &&
                        metal_qw_supported?(rw.ssm_out_qw) &&
                        metal_qw_supported?(rw.ffn_gate_qw) &&
                        metal_qw_supported?(rw.ffn_up_qw) &&
                        metal_qw_supported?(rw.ffn_down_qw)
          rec_layers << rw

          lstate = state.layers[j]
          conv_bytes = ((conv_k - 1) * qkv_dim).to_i64 * sizeof(Float32)
          conv_buf = lstate.conv_state_buf
          if conv_buf.nil?
            conv_buf = ML::MetalBuffer.new(conv_bytes)
            if conv_state = lstate.conv_state
              conv_buf.write(conv_state)
            else
              conv_buf.contents.as(Pointer(UInt8)).clear(conv_bytes)
            end
            lstate.conv_state_buf = conv_buf
          end
          conv_bufs << conv_buf
          if checkpoint_requested
            checkpoint_conv_bufs << checkpoint_state.not_nil!.layers[j].conv_state_buf.not_nil!
          end

          ssm_bytes = (h_v * s * s).to_i64 * sizeof(Float32)
          ssm_buf = lstate.ssm_state_buf
          if ssm_buf.nil?
            ssm_buf = ML::MetalBuffer.new(ssm_bytes)
            if ssm_state = lstate.ssm_state
              ssm_buf.write(ssm_state)
            else
              ssm_buf.contents.as(Pointer(UInt8)).clear(ssm_bytes)
            end
            lstate.ssm_state_buf = ssm_buf
          end
          ssm_bufs << ssm_buf
          if checkpoint_requested
            checkpoint_ssm_bufs << checkpoint_state.not_nil!.layers[j].ssm_state_buf.not_nil!
          end
          j += 1
        end
        return nil unless supported

        kv_dim = hp.head_dim * hp.n_head_kv
        bytes = full_state.kv_cache_bytes(max_seq, kv_dim)
        k_buf = nil.as(ML::MetalBuffer?)
        v_buf = nil.as(ML::MetalBuffer?)
        adaptive_encoder = nil.as(AdaptivePrefillEncoder?)
        if adaptive_cache
          cache = adaptive_cache.not_nil!
          if full_state.k_cache || full_state.v_cache ||
             full_state.k_cache_buf || full_state.v_cache_buf
            raise ArgumentError.new("adaptive resident QBit KV cannot coexist with an F32 owner")
          end
          unless cache.cache_len == start_pos
            raise ArgumentError.new("adaptive resident QBit KV prefill start does not match the live prefix")
          end
          adaptive_encoder = ->(command : ML::Metal::CommandBuffer, q_source : ML::MetalBuffer, gate_source : ML::MetalBuffer, k_source : ML::MetalBuffer, v_source : ML::MetalBuffer, output : ML::MetalBuffer) do
            QwenQBitAdaptiveResidentKV.encode_prefill_chunk_and_append(
              command, cache,
              q_source, gate_source, k_source, v_source, output,
              n_tokens, hp.n_head, hp.n_head // hp.n_head_kv,
              (1.0 / Math.sqrt(hp.head_dim.to_f64)).to_f32,
              expected_start_token: start_pos,
            )
          end
        else
          k_buf = full_state.k_cache_buf
          v_buf = full_state.v_cache_buf
          if k_buf.nil?
            k_buf = ML::MetalBuffer.new(bytes)
            k_buf.contents.as(Pointer(UInt8)).clear(bytes)
            full_state.k_cache_buf = k_buf
          end
          if v_buf.nil?
            v_buf = ML::MetalBuffer.new(bytes)
            v_buf.contents.as(Pointer(UInt8)).clear(bytes)
            full_state.v_cache_buf = v_buf
          end
        end

        scale = (1.0 / Math.sqrt(hp.head_dim.to_f64)).to_f32
        begin
          out = with_prefill_scratch_arena(scratch_arena) do
            Qwen35Metal.full_attn_then_recurrent_chunk_project_many(
              inp,
              full_lw.attn_q_qw, full_lw.attn_k_qw, full_lw.attn_v_qw,
              full_lw.attn_norm, full_lw.attn_q_norm, full_lw.attn_k_norm,
              full_lw.attn_output_qw, k_buf, v_buf, full_lw.post_attention_norm,
              full_lw.ffn_gate_qw, full_lw.ffn_up_qw, full_lw.ffn_down_qw,
              start_pos, n_tokens,
              hp.n_head, hp.n_head_kv, hp.head_dim, hp.rope_dim_count,
              hp.n_head // hp.n_head_kv, hp.rope_freq_base, hp.rms_eps, scale,
              conv_bufs, ssm_bufs, rec_layers, h_k, h_v, s, conv_k,
              "full#{il}+rec#{run_start}-#{run_end - 1}",
              checkpoint_index: checkpoint_index,
              checkpoint_conv_state_bufs: checkpoint_requested ? checkpoint_conv_bufs : nil,
              checkpoint_ssm_state_bufs: checkpoint_requested ? checkpoint_ssm_bufs : nil,
              checkpoint_rollback_log: checkpoint_rollback_log,
              input_buf: input_buf,
              output_buf: output_buf,
              read_output: read_output,
              append_command_buffer: append_command_buffer,
              adaptive_prefill_encoder: adaptive_encoder,
              kv_cache_f16: full_state.kv_cache_f16)
          end
        rescue ex
          if cmd = append_command_buffer
            unless cmd.committed?
              adaptive_cache.try do |cache|
                QwenQBitAdaptiveResidentKV.cancel_pending_append!(cache, cmd)
              end
            end
          end
          raise ex
        end
        if adaptive_cache
          cache = adaptive_cache.not_nil!
          unless out
            QwenQBitAdaptiveResidentKV.cancel_pending_append!(cache, append_command_buffer.not_nil!)
            raise ArgumentError.new("adaptive resident QBit KV fused prefill route declined after allocation")
          end
          pending_adaptive_caches.not_nil! << cache
        end
        out ? {out, run_end} : nil
      {% else %}
        nil
      {% end %}
    end

    # Full-attention GPU route:
    #   qkv projections -> split/norm/rope -> kv write -> attn -> out proj
    private def full_attn_project_routed(lstate : LayerState,
                                         cur : Array(Float32),
                                         lw : Qwen35FullAttnWeights,
                                         hp : Qwen35Hparams,
                                         pos : Int32,
                                         heads_per_group : Int32,
                                         kv_dim : Int32,
                                         max_seq : Int32,
                                         scale : Float32) : Array(Float32)?
      if lstate.kv_cache_f16
        raise ArgumentError.new("F16 KV cache requires the whole-wave Metal decode route")
      end
      {% unless flag?(:cpu_only) %}
        return nil if ENV["QWEN35_FULL_ATTN_FUSE_OFF"]? == "1"
        return nil unless Qwen35Metal.available?
        return nil unless metal_qw_supported?(lw.attn_q_qw) &&
                          metal_qw_supported?(lw.attn_k_qw) &&
                          metal_qw_supported?(lw.attn_v_qw) &&
                          metal_qw_supported?(lw.attn_output_qw)

        bytes = (max_seq * kv_dim).to_i64 * sizeof(Float32)
        k_buf = lstate.k_cache_buf
        v_buf = lstate.v_cache_buf
        if k_buf.nil?
          k_buf = ML::MetalBuffer.new(bytes)
          k_buf.contents.as(Pointer(UInt8)).clear(bytes)
          lstate.k_cache_buf = k_buf
        end
        if v_buf.nil?
          v_buf = ML::MetalBuffer.new(bytes)
          v_buf.contents.as(Pointer(UInt8)).clear(bytes)
          lstate.v_cache_buf = v_buf
        end

        return Qwen35Metal.full_attn_project(
          cur,
          lw.attn_q_qw, lw.attn_k_qw, lw.attn_v_qw,
          lw.attn_q_norm, lw.attn_k_norm, lw.attn_output_qw,
          k_buf, v_buf, pos, hp.n_head, hp.n_head_kv, hp.head_dim,
          hp.rope_dim_count, heads_per_group, hp.rope_freq_base, scale,
        )
      {% else %}
        nil
      {% end %}
    end

    # Try to run a GEMV (batch=1) on Metal if the type is supported and
    # the op is large enough. Returns `nil` if the call should fall back
    # to CPU.
    private def metal_matvec_or_nil(qw : QuantWeight, x : Array(Float32)) : Array(Float32)?
      {% if flag?(:cpu_only) %}
        nil
      {% else %}
        unless metal_qw_eligible?(qw)
          Qwen35Metal::Profile.bump_cpu_fallback
          return nil
        end
        unless Qwen35Metal.available?
          Qwen35Metal::Profile.bump_cpu_fallback
          return nil
        end
        Qwen35Metal.matmul(qw, x, 1)
      {% end %}
    end

    # Batched matvec — runs all qws against the same input in ONE Metal
    # command buffer (one commit+wait, one readback of all outputs).
    # Mixed batches are split: large eligible qws stay batched on Metal,
    # while genuinely tiny qws (e.g. alpha/beta) fall back to CPU only
    # for those slots.
    def qmatvec_many(qws : Array(QuantWeight), x : Array(Float32)) : Array(Array(Float32))
      return [] of Array(Float32) if qws.empty?
      results = Array(Array(Float32)?).new(qws.size, nil)
      {% unless flag?(:cpu_only) %}
        if ENV["QWEN35_BATCH_OFF"]? != "1"
          eligible_idx = Array(Int32).new
          eligible_qws = Array(QuantWeight).new
          qws.each_with_index do |qw, i|
            if metal_qw_supported?(qw)
              eligible_idx << i.to_i32
              eligible_qws << qw
            end
          end
          if !eligible_qws.empty? && Qwen35Metal.available?
            if gpu_results = Qwen35Metal.matmul_many(eligible_qws, x)
              eligible_idx.each_with_index do |orig_i, gpu_i|
                results[orig_i] = gpu_results[gpu_i]
              end
            end
          end
        end
      {% end %}
      qws.each_with_index do |qw, i|
        results[i] ||= qmatvec_nobias(qw, x)
      end
      results.map(&.not_nil!)
    end

    # Quantized matvec — wrapper that routes to QuantMatmul for a single row.
    # result = bias + W @ x, where W is [out_dim, in_dim] stored row-major.
    def qmatvec(qw : QuantWeight, x : Array(Float32), bias : Array(Float32)? = nil) : Array(Float32)
      if (out = metal_matvec_or_nil(qw, x))
        if bias
          out.size.times { |i| out[i] += bias[i] }
        end
        out
      else
        b = bias || Array(Float32).new(qw.out_dim, 0.0_f32)
        QuantMatmul.matmul_add(x, 1, qw.in_dim, qw.raw, qw.type, qw.out_dim, b)
      end
    end

    # Same but without bias (save the allocation when we know bias=0).
    def qmatvec_nobias(qw : QuantWeight, x : Array(Float32)) : Array(Float32)
      if (out = metal_matvec_or_nil(qw, x))
        out
      else
        zero = Array(Float32).new(qw.out_dim, 0.0_f32)
        QuantMatmul.matmul_add(x, 1, qw.in_dim, qw.raw, qw.type, qw.out_dim, zero)
      end
    end

    # Batched quantized matmul for token-major activations:
    #   x:   [batch, in_dim]
    #   out: [batch, out_dim]
    #
    # This is the projection building block for layerwise prefill. It uses the
    # existing Metal batch path when available and falls back to the CPU fused
    # matmul without changing numerics.
    private def qmatmul_nobias(qw : QuantWeight, x : Array(Float32), batch : Int32) : Array(Float32)
      raise ArgumentError.new("qmatmul_nobias batch must be positive") unless batch > 0
      raise ArgumentError.new("qmatmul_nobias x size mismatch: expected #{batch * qw.in_dim}, got #{x.size}") unless x.size == batch * qw.in_dim

      if metal_qw_supported?(qw) && Qwen35Metal.available?
        if (gpu_out = Qwen35Metal.matmul(qw, x, batch))
          return gpu_out
        end
      end

      zero = Array(Float32).new(qw.out_dim, 0.0_f32)
      QuantMatmul.matmul_add(x, batch, qw.in_dim, qw.raw, qw.type, qw.out_dim, zero)
    end

    private def rms_norm_rows(x : Array(Float32), rows : Int32, dim : Int32,
                              w : Array(Float32), eps : Float32) : Array(Float32)
      raise ArgumentError.new("rms_norm_rows x size mismatch") unless x.size == rows * dim
      out = Array(Float32).new(x.size, 0.0_f32)
      rows.times do |r|
        base = r * dim
        ss = 0.0_f64
        dim.times { |j| ss += x[base + j].to_f64 * x[base + j].to_f64 }
        inv_rms = (1.0 / Math.sqrt(ss / dim.to_f64 + eps.to_f64)).to_f32
        dim.times { |j| out[base + j] = x[base + j] * inv_rms * w[j] }
      end
      out
    end

    private def output_project_routed(x : Array(Float32),
                                      norm_weight : Array(Float32),
                                      out_qw : QuantWeight,
                                      eps : Float32) : Array(Float32)?
      {% unless flag?(:cpu_only) %}
        return nil if ENV["QWEN35_HEAD_FUSE_OFF"]? == "1"
        return nil unless metal_qw_supported?(out_qw)
        return nil unless Qwen35Metal.available?
        Qwen35Metal.rmsnorm_project(x, norm_weight, out_qw, eps)
      {% else %}
        nil
      {% end %}
    end

    private def output_project_top1_routed(x : Array(Float32),
                                           norm_weight : Array(Float32),
                                           out_qw : QuantWeight,
                                           eps : Float32) : {Int32, Float32}?
      {% unless flag?(:cpu_only) %}
        return nil if ENV["QWEN35_HEAD_TOP1_FUSED"]? == "0"
        return nil unless metal_qw_supported?(out_qw)
        return nil unless Qwen35Metal.available?
        if packed = Qwen35Metal.rmsnorm_project_top1(x, norm_weight, out_qw, eps)
          return {packed[0].to_i32, packed[1]} if packed.size == 2
        end
      {% end %}
      nil
    end

    private def output_project_top1_resident_routed(x_buf : ML::MetalBuffer,
                                                    row : Int32,
                                                    norm_weight : Array(Float32),
                                                    out_qw : QuantWeight,
                                                    eps : Float32) : {Int32, Float32}?
      {% unless flag?(:cpu_only) %}
        return nil if row < 0
        return nil if ENV["QWEN35_HEAD_TOP1_FUSED"]? == "0"
        return nil unless Qwen35Metal.available?
        element_offset = row.to_i64 * out_qw.in_dim.to_i64
        if packed = Qwen35Metal.rmsnorm_project_top1_buffer(
             x_buf, element_offset, norm_weight, out_qw, eps,
           )
          return {packed[0].to_i32, packed[1]} if packed.size == 2
        end
      {% end %}
      nil
    end

    private def prefill_adaptive_resident_top1_supported?(weights : Qwen35Weights,
                                                          state : State,
                                                          rows : Int32) : Bool
      {% unless flag?(:cpu_only) %}
        return false unless ENV["QWEN35_PREFILL_TOP1_ADAPTIVE_RESIDENT"]? == "1"
        return false if ENV["QWEN35_HEAD_TOP1_FUSED"]? == "0"
        return false if ENV["QWEN35_PREFILL_CHUNK_OFF"]? == "1"
        return false if ENV["QWEN35_PREFILL_RESIDENT_BOUNDARY_OFF"]? == "1"
        return false unless Qwen35Metal.available?
        return false unless state.layers[-1].adaptive_kv
        return false if rows > prefill_chunk_size(true)
        hp = weights.hparams
        return false unless weights.output.in_dim == hp.n_embd
        return false unless weights.output_norm.size == hp.n_embd
        last_layer = weights.layers[-1].as?(Qwen35FullAttnWeights)
        return false unless last_layer
        return false unless Qwen35Metal.rmsnorm_project_top1_supported?(weights.output)
        metal_qw_supported?(last_layer.attn_q_qw) &&
          metal_qw_supported?(last_layer.attn_k_qw) &&
          metal_qw_supported?(last_layer.attn_v_qw) &&
          metal_qw_supported?(last_layer.attn_output_qw) &&
          metal_qw_supported?(last_layer.ffn_gate_qw) &&
          metal_qw_supported?(last_layer.ffn_up_qw) &&
          metal_qw_supported?(last_layer.ffn_down_qw)
      {% else %}
        false
      {% end %}
    end

    private def output_project_top1s_routed(x : Array(Float32),
                                            rows : Int32,
                                            norm_weight : Array(Float32),
                                            out_qw : QuantWeight,
                                            eps : Float32) : Array({Int32, Float32})?
      {% unless flag?(:cpu_only) %}
        return nil if ENV["QWEN35_HEAD_TOP1_FUSED"]? == "0"
        return nil unless metal_qw_supported?(out_qw)
        return nil unless Qwen35Metal.available?
        if ENV["QWEN35_HEAD_FULL_ROWS_GUARDED"]? == "1" && ENV["QWEN35_HEAD_FULL_ROWS_OFF"]? != "1"
          if top1s = Qwen35Metal.rmsnorm_project_full_top1_rows_guarded(x, rows, norm_weight, out_qw, eps)
            return top1s
          end
        end
        if ENV["QWEN35_HEAD_FULL_ROWS"]? == "1" && ENV["QWEN35_HEAD_FULL_ROWS_OFF"]? != "1"
          if top1s = Qwen35Metal.rmsnorm_project_full_top1_rows(x, rows, norm_weight, out_qw, eps)
            return top1s
          end
        end
        rows_min = (ENV["QWEN35_HEAD_TOP1_ROWS_MIN"]? || "8").to_i
        if ENV["QWEN35_HEAD_TOP1_ROWS_OFF"]? != "1" &&
           (ENV["QWEN35_HEAD_TOP1_ROWS"]? == "1" || rows >= rows_min)
          return Qwen35Metal.rmsnorm_project_top1_rows(x, rows, norm_weight, out_qw, eps)
        end
      {% end %}
      nil
    end

    private def output_project_top1s_resident_routed(x_buf : ML::MetalBuffer,
                                                     rows : Int32,
                                                     norm_weight : Array(Float32),
                                                     out_qw : QuantWeight,
                                                     eps : Float32) : Array({Int32, Float32})?
      {% unless flag?(:cpu_only) %}
        return nil if ENV["QWEN35_HEAD_TOP1_FUSED"]? == "0"
        return nil unless metal_qw_supported?(out_qw)
        return nil unless Qwen35Metal.available?
        rows_min = (ENV["QWEN35_HEAD_TOP1_ROWS_MIN"]? || "8").to_i
        return nil unless ENV["QWEN35_HEAD_TOP1_ROWS_OFF"]? != "1" &&
                          (ENV["QWEN35_HEAD_TOP1_ROWS"]? == "1" || rows >= rows_min)
        return Qwen35Metal.rmsnorm_project_top1_rows_buffer(x_buf, rows, norm_weight, out_qw, eps)
      {% end %}
      nil
    end

    private def prefill_top1_resident_rows_supported?(weights : Qwen35Weights, rows : Int32) : Bool
      {% unless flag?(:cpu_only) %}
        return false unless ENV["QWEN35_PREFILL_TOP1_RESIDENT_ROWS"]? == "1"
        return false unless ENV["QWEN35_PREFILL_RESIDENT_BOUNDARY_OFF"]? != "1"
        return false unless Qwen35Metal.available?
        return false unless weights.output.type.q6_k? && weights.output.in_dim % 256 == 0
        rows_min = (ENV["QWEN35_HEAD_TOP1_ROWS_MIN"]? || "8").to_i
        return false unless ENV["QWEN35_HEAD_TOP1_ROWS_OFF"]? != "1" &&
                            (ENV["QWEN35_HEAD_TOP1_ROWS"]? == "1" || rows >= rows_min)
        last_layer = weights.layers[-1].as?(Qwen35FullAttnWeights)
        return false unless last_layer
        metal_qw_supported?(weights.output) &&
          metal_qw_supported?(last_layer.attn_q_qw) &&
          metal_qw_supported?(last_layer.attn_k_qw) &&
          metal_qw_supported?(last_layer.attn_v_qw) &&
          metal_qw_supported?(last_layer.attn_output_qw) &&
          metal_qw_supported?(last_layer.ffn_gate_qw) &&
          metal_qw_supported?(last_layer.ffn_up_qw) &&
          metal_qw_supported?(last_layer.ffn_down_qw)
      {% else %}
        false
      {% end %}
    end

    # Admit the terminal-row prefill shortcut only when the complete final
    # layer contract is known before any prompt state is mutated. The shortcut
    # is deliberately limited to a position-zero, single chunk: longer/non-zero
    # spans retain the ordinary prefill + final-token decode path.
    def prefill_full_logits_last_supported?(weights : Qwen35Weights,
                                            state : State,
                                            rows : Int32,
                                            start_pos : Int32) : Bool
      {% unless flag?(:cpu_only) %}
        return false unless rows > 1 && start_pos == 0
        return false if ENV["QWEN35_FINAL_FULL_LAST_OFF"]? == "1"
        return false if ENV["QWEN35_PREFILL_CHUNK_OFF"]? == "1"
        return false unless rows <= prefill_chunk_size(false)
        return false unless rows <= state.max_seq
        return false unless Qwen35Metal.available?
        return false if state.adaptive_kv?

        hp = weights.hparams
        return false unless weights.layers.size == hp.n_layer
        return false unless state.layers.size == weights.layers.size
        return false unless state.layers.all? { |layer| layer.position == 0 }
        return false unless hp.n_head > 0 && hp.n_head_kv > 0
        return false unless hp.n_head % hp.n_head_kv == 0

        last_layer = weights.layers.last.as?(Qwen35FullAttnWeights)
        return false unless last_layer
        last_state = state.layers.last
        return false if last_state.adaptive_kv

        hidden_dim = hp.n_embd
        q_dim = hp.n_head * hp.head_dim
        kv_dim = hp.n_head_kv * hp.head_dim
        ffn_dim = last_layer.ffn_gate_qw.out_dim
        return false unless last_layer.attn_norm.size == hidden_dim
        return false unless last_layer.attn_q_norm.size == hp.head_dim
        return false unless last_layer.attn_k_norm.size == hp.head_dim
        return false unless last_layer.post_attention_norm.size == hidden_dim
        return false unless last_layer.attn_q_qw.in_dim == hidden_dim &&
                            last_layer.attn_q_qw.out_dim == 2 * q_dim
        return false unless last_layer.attn_k_qw.in_dim == hidden_dim &&
                            last_layer.attn_k_qw.out_dim == kv_dim
        return false unless last_layer.attn_v_qw.in_dim == hidden_dim &&
                            last_layer.attn_v_qw.out_dim == kv_dim
        return false unless last_layer.attn_output_qw.in_dim == q_dim &&
                            last_layer.attn_output_qw.out_dim == hidden_dim
        return false unless last_layer.ffn_gate_qw.in_dim == hidden_dim && ffn_dim > 0
        return false unless last_layer.ffn_up_qw.in_dim == hidden_dim &&
                            last_layer.ffn_up_qw.out_dim == ffn_dim
        return false unless last_layer.ffn_down_qw.in_dim == ffn_dim &&
                            last_layer.ffn_down_qw.out_dim == hidden_dim
        return false unless weights.output_norm.size == hidden_dim
        return false unless weights.output.in_dim == hidden_dim

        supported = metal_qw_supported?(last_layer.attn_q_qw) &&
                    metal_qw_supported?(last_layer.attn_k_qw) &&
                    metal_qw_supported?(last_layer.attn_v_qw) &&
                    metal_qw_supported?(last_layer.attn_output_qw) &&
                    metal_qw_supported?(last_layer.ffn_gate_qw) &&
                    metal_qw_supported?(last_layer.ffn_up_qw) &&
                    metal_qw_supported?(last_layer.ffn_down_qw)
        return false unless supported
        return false unless Qwen35Metal.full_attn_layer_chunk_project_last_supported?(
                              last_layer.attn_q_qw, last_layer.attn_k_qw, last_layer.attn_v_qw,
                              last_layer.attn_output_qw, last_layer.ffn_gate_qw,
                              last_layer.ffn_up_qw, last_layer.ffn_down_qw,
                              rows,
                              state.kv_cache_f16?,
                            )

        required_kv_values = state.max_seq.to_i64 * kv_dim.to_i64
        qkv_dim = 2 * hp.ssm_group_count * hp.ssm_state_size + hp.ssm_time_step_rank * hp.ssm_state_size
        required_conv_values = (hp.ssm_conv_kernel - 1).to_i64 * qkv_dim.to_i64
        required_ssm_values = hp.ssm_time_step_rank.to_i64 * hp.ssm_state_size.to_i64 * hp.ssm_state_size.to_i64
        state.layers.each_with_index do |layer, layer_index|
          if hp.full_attention?(layer_index)
            required_kv_bytes = required_kv_values * layer.kv_cache_element_bytes
            return false if layer.kv_cache_f16 && (layer.k_cache || layer.v_cache)
            return false if (cache = layer.k_cache) && cache.size < required_kv_values
            return false if (cache = layer.v_cache) && cache.size < required_kv_values
            return false if (buf = layer.k_cache_buf) && buf.size < required_kv_bytes
            return false if (buf = layer.v_cache_buf) && buf.size < required_kv_bytes
          else
            return false if (conv = layer.conv_state) && conv.size < required_conv_values
            return false if (ssm = layer.ssm_state) && ssm.size < required_ssm_values
            return false if (buf = layer.conv_state_buf) && buf.size < required_conv_values * sizeof(Float32)
            return false if (buf = layer.ssm_state_buf) && buf.size < required_ssm_values * sizeof(Float32)
          end
        end
        true
      {% else %}
        false
      {% end %}
    end

    # ─────────────────────────────────────────────────────────────────────
    # Full-attention layer forward (single-token decode)
    # ─────────────────────────────────────────────────────────────────────
    #
    # Structure (matches llama.cpp qwen35::build_layer_attn + surrounding code):
    #   inpSA = input
    #   cur = RMSNorm(inpSA, attn_norm)
    #   Q_full = attn_q_qw @ cur                           # [2*head_dim*n_head]
    #   Split Q_full per-head into Q [head_dim*n_head] and gate [head_dim*n_head]
    #     (interleaved: per head, first head_dim = Q, next head_dim = gate)
    #   K = attn_k_qw @ cur                                # [head_dim*n_head_kv]
    #   V = attn_v_qw @ cur                                # [head_dim*n_head_kv]
    #   Per-head RMSNorm on Q (attn_q_norm) and K (attn_k_norm)
    #   M-RoPE partial on first rope_dim_count dims of each Q and K head
    #   Write K[pos], V[pos] into KV cache
    #   GQA attention: heads_per_group = n_head / n_head_kv
    #     scores[p] = (Q · K[p]) / sqrt(head_dim)  for p in 0..pos
    #     softmax; out = sum_p scores[p] * V[p]
    #   out *= sigmoid(gate)  (elementwise per position in out)
    #   attn_out = attn_output_qw @ out
    #   cur = inpSA + attn_out   (residual 1)
    #   ffn_res = cur
    #   cur = RMSNorm(cur, post_attention_norm)
    #   gate_ff = silu(ffn_gate_qw @ cur); up = ffn_up_qw @ cur
    #   cur = ffn_down_qw @ (gate_ff * up)
    #   cur = ffn_res + cur   (residual 2)
    #
    # Returns the new hidden state (Array(Float32) size n_embd).
    def forward_full_attn_layer(inpSA : Array(Float32), pos : Int32,
                                lw : Qwen35FullAttnWeights,
                                lstate : LayerState,
                                hp : Qwen35Hparams,
                                max_seq : Int32) : Array(Float32)
      if lstate.adaptive_kv
        raise ArgumentError.new("adaptive resident QBit KV decode requires the synchronous whole-token Metal route")
      end
      n_embd = hp.n_embd
      n_head = hp.n_head
      n_head_kv = hp.n_head_kv
      head_dim = hp.head_dim
      n_ff = hp.n_ff
      kv_dim = head_dim * n_head_kv
      q_dim = head_dim * n_head
      heads_per_group = n_head // n_head_kv

      # 1. attn_norm
      cur = rms_norm(inpSA, lw.attn_norm, hp.rms_eps)

      scale = (1.0 / Math.sqrt(head_dim.to_f64)).to_f32
      fused_layer = full_attn_layer_project_routed(inpSA, cur, lstate, lw, hp, pos, heads_per_group, kv_dim, max_seq)
      return fused_layer if fused_layer

      attn_out = full_attn_project_routed(
        lstate, cur, lw, hp, pos, heads_per_group, kv_dim, max_seq, scale,
      )
      unless attn_out
        # 2-4. Batched Q+gate/K/V projections (all from same `cur` → one sync)
        qkv_outs = qmatvec_many([lw.attn_q_qw, lw.attn_k_qw, lw.attn_v_qw], cur)
        q_full = qkv_outs[0] # [2 * head_dim * n_head]
        k = qkv_outs[1]      # [head_dim * n_head_kv]
        v = qkv_outs[2]      # [head_dim * n_head_kv]

        # 3. Split Q and gate (interleaved per head: [Q_h0, gate_h0, Q_h1, gate_h1, ...])
        q = Array(Float32).new(q_dim, 0.0_f32)
        gate = Array(Float32).new(q_dim, 0.0_f32)
        n_head.times do |h|
          src_base = h * 2 * head_dim
          dst_base = h * head_dim
          head_dim.times do |d|
            q[dst_base + d] = q_full[src_base + d]
            gate[dst_base + d] = q_full[src_base + head_dim + d]
          end
        end

        # 5. Per-head RMSNorm on Q and K (shared weights across heads)
        n_head.times do |h|
          rms_norm_slice!(q, h * head_dim, head_dim, lw.attn_q_norm, hp.rms_eps)
        end
        n_head_kv.times do |h|
          rms_norm_slice!(k, h * head_dim, head_dim, lw.attn_k_norm, hp.rms_eps)
        end

        # 6. M-RoPE partial on first rope_dim_count dims of each Q and K head
        n_head.times do |h|
          rope_partial!(q, h * head_dim, hp.rope_dim_count, head_dim, pos, hp.rope_freq_base)
        end
        n_head_kv.times do |h|
          rope_partial!(k, h * head_dim, hp.rope_dim_count, head_dim, pos, hp.rope_freq_base)
        end

        # 7. Append K, V to cache at current position + 8-9. GQA attention + gate.
        attn_out = attn_decode_project_routed(
          lstate, q, gate, k, v, lw.attn_output_qw,
          pos, n_head, n_head_kv, head_dim, heads_per_group, kv_dim, max_seq, scale,
        )
        unless attn_out
          attn_o = attn_decode_routed(lstate, q, gate, k, v, pos, n_head, n_head_kv,
            head_dim, heads_per_group, kv_dim, max_seq, scale)
          # 10. Output projection
          attn_out = qmatvec_nobias(lw.attn_output_qw, attn_o) # [n_embd]
        end
      end

      # 11. Residual
      inpL2 = Array(Float32).new(n_embd) { |i| inpSA[i] + attn_out.not_nil![i] }

      # 12. post_attention_norm
      cur2 = rms_norm(inpL2, lw.post_attention_norm, hp.rms_eps)

      # 13. SwiGLU FFN
      ffn_out = ffn_project_routed(cur2, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw)
      unless ffn_out
        gu = qmatvec_many([lw.ffn_gate_qw, lw.ffn_up_qw], cur2)
        gate_ff = gu[0] # [n_ff]
        up_ff = gu[1]   # [n_ff]
        silu!(gate_ff)
        combined = Array(Float32).new(n_ff) { |i| gate_ff[i] * up_ff[i] }
        ffn_out = qmatvec_nobias(lw.ffn_down_qw, combined) # [n_embd]
      end

      # 14. Residual
      Array(Float32).new(n_embd) { |i| inpL2[i] + ffn_out.not_nil![i] }
    end

    # ─────────────────────────────────────────────────────────────────────
    # DeltaNet / GatedDeltaRule recurrent layer forward (single-token decode)
    # ─────────────────────────────────────────────────────────────────────
    #
    # Structure (matches llama.cpp qwen35::build_layer_attn_linear +
    #            delta_net_base::build_delta_net_autoregressive):
    #
    # Per step:
    #   cur = RMSNorm(inpSA, attn_norm)
    #   qkv_mixed = attn_qkv_qw @ cur                         # [qkv_dim = 2*H_k*S_k + H_v*S_v]
    #   z         = attn_gate_qw @ cur                        # [d_inner = H_v * S_v]
    #   alpha     = ssm_alpha_qw @ cur                        # [H_v]
    #   beta      = sigmoid(ssm_beta_qw @ cur)                # [H_v]
    #   a_soft    = softplus(alpha + ssm_dt_bias)             # [H_v]
    #   g         = a_soft * ssm_a                            # [H_v]  (ssm_a already pre-transformed)
    #
    #   conv_input[0..K-2] = conv_state;  conv_input[K-1] = qkv_mixed
    #   new_conv_state     = conv_input[1..K]                 # shift window
    #   conv_out[ch]       = sum_{k=0}^{K-1} conv_input[k,ch] * ssm_conv1d[k,ch]
    #   conv_out           = silu(conv_out)
    #   q_conv = conv_out[0                     .. H_k*S_k)         # [H_k, S_k]
    #   k_conv = conv_out[H_k*S_k               .. 2*H_k*S_k)       # [H_k, S_k]
    #   v_conv = conv_out[2*H_k*S_k             .. 2*H_k*S_k+H_v*S_v)  # [H_v, S_v]
    #   L2-norm each (S_k)-slice of q_conv and k_conv
    #   Repeat q_conv/k_conv so each v-head h_v maps to k-head (h_v % H_k)
    #
    #   Delta rule per v-head h (scale = 1/sqrt(S_k), ghead = exp(g[h]), bhead = beta[h]):
    #     state[h] *= ghead                                    # [S_v × S_v] decay
    #     sk[d2]   = sum_{d1} state[h, d1, d2] * K[h, d1]
    #     delt[d2] = bhead * (V[h, d2] - sk[d2])
    #     state[h, d1, d2] += K[h, d1] * delt[d2]              # outer product add
    #     out[h, d2] = sum_{d1} state[h, d1, d2] * (Q[h, d1] * scale)
    #
    #   norm[h, d] = RMSNorm_per_head(out[h,:], ssm_norm) * silu(z[h, d])
    #   attn = ssm_out_qw @ flatten(norm)                     # [n_embd]
    #
    # Then standard residual + post_attention_norm + SwiGLU FFN + residual.
    #
    # State layout in lstate.ssm_state (Array(Float32)):
    #   state[h * S_v * S_v + d2 * S_v + d1]  (h-major, d2-major, d1 contiguous)
    #
    # Conv state layout in lstate.conv_state (Array(Float32)):
    #   conv[t * qkv_dim + ch]  (time-major, channels minor)
    def forward_recurrent_layer(inpSA : Array(Float32), _pos : Int32,
                                lw : Qwen35RecurrentWeights,
                                lstate : LayerState,
                                hp : Qwen35Hparams,
                                _max_seq : Int32) : Array(Float32)
      n_embd = hp.n_embd
      n_ff = hp.n_ff
      h_k = hp.ssm_group_count    # num_k_heads
      h_v = hp.ssm_time_step_rank # num_v_heads
      s_k = hp.ssm_state_size     # head_k_dim
      s_v = hp.ssm_state_size     # head_v_dim (same per qwen35)
      d_inner = hp.ssm_inner_size # H_v * S_v
      qkv_dim = 2 * h_k * s_k + h_v * s_v
      conv_k = hp.ssm_conv_kernel # typically 4
      heads_per_k = h_v // h_k

      # 1. attn_norm
      cur = rms_norm(inpSA, lw.attn_norm, hp.rms_eps)

      fused_layer = recurrent_layer_project_routed(inpSA, cur, lstate, lw, hp)
      return fused_layer if fused_layer

      attn_out = recurrent_attn_project_routed(lstate, cur, lw, hp)
      unless attn_out
        # 2-4. Batched qkv/gate/alpha/beta projections (all from same `cur` → one sync)
        proj = qmatvec_many([lw.attn_qkv_qw, lw.attn_gate_qw, lw.ssm_alpha_qw, lw.ssm_beta_qw], cur)
        qkv_mixed = proj[0] # [qkv_dim]
        z = proj[1]         # [d_inner = h_v * s_v]
        alpha = proj[2]     # [h_v]
        beta = proj[3]      # [h_v]
        h_v.times { |i| beta[i] = sigmoid(beta[i]) }

        # 5. gate per head (g[h] = softplus(alpha[h] + ssm_dt_bias[h]) * ssm_a[h])
        # ssm_a is pre-transformed (-A_log.exp() in llama.cpp), multiply directly
        g = Array(Float32).new(h_v) do |i|
          xi = alpha[i] + lw.ssm_dt_bias[i]
          sp = xi > 20.0_f32 ? xi : Math.log(1.0_f32 + Math.exp(xi)).to_f32
          sp * lw.ssm_a[i]
        end

        # 6. Conv state (lazy alloc). Layout: conv[t*qkv_dim + ch], t in 0..K-2 (K=conv_k)
        conv_state = lstate.conv_state ||= Array(Float32).new((conv_k - 1) * qkv_dim, 0.0_f32)

        # 7. Convolution output for current token.
        # GGUF ssm_conv1d dims=[K, qkv_dim] with dims[0]=K innermost → layout is conv1d[ch*K + t].
        # conv_state is OUR internal buffer (layout [t*qkv_dim + ch]) — unchanged.
        #    conv_out[ch] = sum_{k=0}^{K-2} conv_state[k*qkv_dim+ch] * conv1d[ch*K + k]
        #                 + qkv_mixed[ch]                          * conv1d[ch*K + (K-1)]
        conv_out = Array(Float32).new(qkv_dim) do |ch|
          acc = 0.0_f32
          w_base = ch * conv_k
          (conv_k - 1).times do |t|
            acc += conv_state[t * qkv_dim + ch] * lw.ssm_conv1d[w_base + t]
          end
          acc += qkv_mixed[ch] * lw.ssm_conv1d[w_base + (conv_k - 1)]
          acc
        end

        # 8. Update conv_state: shift window. new_state[t] = old_state[t+1], last = qkv_mixed
        (conv_k - 2).times do |t|
          src_off = (t + 1) * qkv_dim
          dst_off = t * qkv_dim
          qkv_dim.times { |ch| conv_state[dst_off + ch] = conv_state[src_off + ch] }
        end
        last_off = (conv_k - 2) * qkv_dim
        qkv_dim.times { |ch| conv_state[last_off + ch] = qkv_mixed[ch] }

        # 9. SiLU on conv output
        silu!(conv_out)

        # 10. Split conv_out into q, k, v
        q_conv = Array(Float32).new(h_k * s_k) { |i| conv_out[i] }
        k_conv = Array(Float32).new(h_k * s_k) { |i| conv_out[h_k * s_k + i] }
        v_conv = Array(Float32).new(h_v * s_v) { |i| conv_out[2 * h_k * s_k + i] }

        # 11. L2-norm each S_k-slice of q_conv and k_conv
        h_k.times do |h|
          l2_norm_slice!(q_conv, h * s_k, s_k, hp.rms_eps)
          l2_norm_slice!(k_conv, h * s_k, s_k, hp.rms_eps)
        end

        # 12. Delta rule state update + output computation
        scale = (1.0 / Math.sqrt(s_k.to_f64)).to_f32
        # Convert g[h] to ghead = exp(g[h]) inline; kernel and reference share
        # `delta_net_step!` below, which expects the already-exp'd decay.
        ghead = Array(Float32).new(h_v) { |h| Math.exp(g[h].to_f64).to_f32 }

        attn_out = delta_net_project_routed(
          lstate, q_conv, k_conv, v_conv, ghead, beta, z,
          lw.ssm_norm, lw.ssm_out_qw, h_k, h_v, s_k, scale, hp.rms_eps,
        )
        unless attn_out
          y = delta_net_step_routed(lstate, q_conv, k_conv, v_conv, ghead, beta,
            h_k, h_v, s_k, scale)

          # 13. Gated RMSNorm: norm[h,d] = RMSNorm_per_head(y[h,:], ssm_norm) * silu(z[h,d])
          h_v.times do |h|
            rms_norm_slice!(y, h * s_v, s_v, lw.ssm_norm, hp.rms_eps)
          end
          (h_v * s_v).times { |i| y[i] = y[i] * silu(z[i]) }

          # 14. Output projection (ssm_out)
          attn_out = qmatvec_nobias(lw.ssm_out_qw, y) # [n_embd]
        end
      end

      # 15. Residual 1
      inpL2 = Array(Float32).new(n_embd) { |i| inpSA[i] + attn_out.not_nil![i] }

      # 16. Post-attention norm
      cur2 = rms_norm(inpL2, lw.post_attention_norm, hp.rms_eps)

      # 17. SwiGLU FFN
      ffn_out = ffn_project_routed(cur2, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw)
      unless ffn_out
        gu = qmatvec_many([lw.ffn_gate_qw, lw.ffn_up_qw], cur2)
        gate_ff = gu[0]
        up_ff = gu[1]
        silu!(gate_ff)
        combined = Array(Float32).new(n_ff) { |i| gate_ff[i] * up_ff[i] }
        ffn_out = qmatvec_nobias(lw.ffn_down_qw, combined)
      end

      # 18. Residual 2
      Array(Float32).new(n_embd) { |i| inpL2[i] + ffn_out.not_nil![i] }
    end

    # Multi-token recurrent layer prefill.
    #
    # Exact semantics are the same as repeated `forward_recurrent_layer` calls:
    # the convolution and DeltaNet states are scanned in token order. The
    # speedup comes from batching projections and running the recurrent prep +
    # DeltaNet scan once per layer chunk instead of once per token.
    private def forward_recurrent_layer_chunk(inp : Array(Float32),
                                              n_tokens : Int32,
                                              lw : Qwen35RecurrentWeights,
                                              lstate : LayerState,
                                              hp : Qwen35Hparams,
                                              max_seq : Int32) : Array(Float32)
      {% unless flag?(:cpu_only) %}
        if ENV["QWEN35_PREFILL_CHUNK_OFF"]? != "1" && n_tokens > 1
          supported = Qwen35Metal.available? &&
                      metal_qw_supported?(lw.attn_qkv_qw) &&
                      metal_qw_supported?(lw.attn_gate_qw) &&
                      metal_qw_supported?(lw.ssm_alpha_qw) &&
                      metal_qw_supported?(lw.ssm_beta_qw) &&
                      metal_qw_supported?(lw.ssm_out_qw) &&
                      metal_qw_supported?(lw.ffn_gate_qw) &&
                      metal_qw_supported?(lw.ffn_up_qw) &&
                      metal_qw_supported?(lw.ffn_down_qw)
          if supported
            h_k = hp.ssm_group_count
            h_v = hp.ssm_time_step_rank
            s = hp.ssm_state_size
            qkv_dim = 2 * h_k * s + h_v * s
            conv_k = hp.ssm_conv_kernel

            conv_bytes = ((conv_k - 1) * qkv_dim).to_i64 * sizeof(Float32)
            conv_buf = lstate.conv_state_buf
            if conv_buf.nil?
              conv_buf = ML::MetalBuffer.new(conv_bytes)
              if conv_state = lstate.conv_state
                conv_buf.write(conv_state)
              else
                conv_buf.contents.as(Pointer(UInt8)).clear(conv_bytes)
              end
              lstate.conv_state_buf = conv_buf
            end

            ssm_bytes = (h_v * s * s).to_i64 * sizeof(Float32)
            ssm_buf = lstate.ssm_state_buf
            if ssm_buf.nil?
              ssm_buf = ML::MetalBuffer.new(ssm_bytes)
              if ssm_state = lstate.ssm_state
                ssm_buf.write(ssm_state)
              else
                ssm_buf.contents.as(Pointer(UInt8)).clear(ssm_bytes)
              end
              lstate.ssm_state_buf = ssm_buf
            end

            if gpu_out = Qwen35Metal.recurrent_layer_chunk_project(
                 inp, conv_buf, ssm_buf, lw.attn_norm,
                 lw.attn_qkv_qw, lw.attn_gate_qw, lw.ssm_alpha_qw, lw.ssm_beta_qw,
                 lw.ssm_conv1d, lw.ssm_dt_bias, lw.ssm_a, lw.ssm_norm, lw.ssm_out_qw,
                 lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
                 h_k, h_v, s, conv_k, n_tokens, hp.rms_eps)
              return gpu_out
            end
          end
        end
      {% end %}

      out = Array(Float32).new(inp.size, 0.0_f32)
      n_tokens.times do |t|
        row = inp[t * hp.n_embd, hp.n_embd]
        y = forward_recurrent_layer(row, 0, lw, lstate, hp, max_seq)
        hp.n_embd.times { |i| out[t * hp.n_embd + i] = y[i] }
      end
      out
    end

    # ─────────────────────────────────────────────────────────────────────
    # Full decoder forward (single-token autoregressive)
    # ─────────────────────────────────────────────────────────────────────
    #
    # Given a token id and the current per-sequence state + position, compute
    # the logits vector [vocab_size] for the NEXT token.
    #
    # Steps:
    #   x = embedding_lookup(token_embd, token_id)   # [n_embd]
    #   for il in 0 ... n_layer:
    #     if full_attention?(il):
    #       x = forward_full_attn_layer(x, pos, layers[il], state.layers[il], hp, state.max_seq)
    #     else:
    #       x = forward_recurrent_layer(x, pos, layers[il], state.layers[il], hp, state.max_seq)
    #   x = RMSNorm(x, output_norm)
    #   logits = output @ x                          # [vocab_size]
    #
    # Note: full-attention layers internally update their KV cache; recurrent
    # layers update conv_state / ssm_state. Caller must bump state.layers[il].position
    # (for full-attn it's tracked by pos arg anyway; position field unused in
    # current impl but reserved for future fused paths).
    def forward(weights : Qwen35Weights, token_id : Int32, pos : Int32,
                state : State) : Array(Float32)
      if logits = forward_decode_wave_routed(weights, token_id, pos, state)
        return logits
      end
      if state.kv_cache_f16?
        raise ArgumentError.new("F16 KV cache requires the whole-token Metal decode route")
      end

      hp = weights.hparams
      max_seq = state.max_seq

      x = embedding_lookup(weights.token_embd, token_id)

      weights.layers.each_with_index do |lw, il|
        case lw
        in Qwen35FullAttnWeights
          x = forward_full_attn_layer(x, pos, lw, state.layers[il], hp, max_seq)
        in Qwen35RecurrentWeights
          x = forward_recurrent_layer(x, pos, lw, state.layers[il], hp, max_seq)
        end
      end

      if logits = output_project_routed(x, weights.output_norm, weights.output, hp.rms_eps)
        logits
      else
        rms_norm!(x, weights.output_norm, hp.rms_eps)
        qmatvec_nobias(weights.output, x)
      end
    end

    # Full decoder body for one token, returning the pre-output-norm hidden.
    #
    # This intentionally bypasses the fused decode-wave route because that route
    # can update state without materializing the final hidden vector. It is a
    # correctness/probe helper, not the hot decode path.
    def forward_hidden(weights : Qwen35Weights, token_id : Int32, pos : Int32,
                       state : State) : Array(Float32)
      if state.kv_cache_f16?
        raise ArgumentError.new("F16 KV cache does not support host-materialized single-token hidden output")
      end
      hp = weights.hparams
      max_seq = state.max_seq

      x = embedding_lookup(weights.token_embd, token_id)

      weights.layers.each_with_index do |lw, il|
        case lw
        in Qwen35FullAttnWeights
          x = forward_full_attn_layer(x, pos, lw, state.layers[il], hp, max_seq)
        in Qwen35RecurrentWeights
          x = forward_recurrent_layer(x, pos, lw, state.layers[il], hp, max_seq)
        end
      end

      x
    end

    # Project a pre-output-norm hidden through the normal Qwen output head.
    # The input is duplicated because the CPU fallback RMSNorm mutates in place.
    def hidden_top1(weights : Qwen35Weights, hidden : Array(Float32)) : {Int32, Float32}
      hp = weights.hparams
      x = hidden.dup
      if top1 = output_project_top1_routed(x, weights.output_norm, weights.output, hp.rms_eps)
        return top1
      end

      rms_norm!(x, weights.output_norm, hp.rms_eps)
      logits = qmatvec_nobias(weights.output, x)
      maxv = logits.max
      {logits.index(maxv).not_nil!.to_i32, maxv}
    end

    # Probe helper: run one token through the decoder and project the hidden
    # after each layer. This mutates `state` exactly like normal decode for the
    # consumed token, but it is intentionally diagnostic rather than hot-path.
    def forward_layer_top1_trace(weights : Qwen35Weights, token_id : Int32, pos : Int32,
                                 state : State) : Array(LayerTop1TraceRow)
      hp = weights.hparams
      max_seq = state.max_seq
      x = embedding_lookup(weights.token_embd, token_id)
      rows = Array(LayerTop1TraceRow).new(weights.layers.size)

      weights.layers.each_with_index do |lw, il|
        case lw
        in Qwen35FullAttnWeights
          x = forward_full_attn_layer(x, pos, lw, state.layers[il], hp, max_seq)
        in Qwen35RecurrentWeights
          x = forward_recurrent_layer(x, pos, lw, state.layers[il], hp, max_seq)
        end
        top1, logit = hidden_top1(weights, x)
        rows << LayerTop1TraceRow.new(il.to_i32, top1, logit)
      end

      rows
    end

    # Project a pre-output-norm hidden through the normal Qwen output head and
    # return the best two logits. This is used by exact routing controllers
    # that need a legal confidence signal from an already-computed boundary row.
    def hidden_top2(weights : Qwen35Weights, hidden : Array(Float32)) : {Int32, Float32, Int32, Float32}
      hp = weights.hparams
      x = hidden.dup
      rms_norm!(x, weights.output_norm, hp.rms_eps)
      top2_from_logits(qmatvec_nobias(weights.output, x))
    end

    # Project a pre-norm hidden through an explicit RMSNorm/head pair.
    # Used by GGUF MTP, where the draft head can have its own shared norm/head
    # while still reusing the target runtime's fused Q6/Q8 top1 kernels.
    def hidden_top1_with_norm(hidden : Array(Float32),
                              norm_weight : Array(Float32),
                              out_qw : QuantWeight,
                              eps : Float32) : {Int32, Float32}
      x = hidden.dup
      if top1 = output_project_top1_routed(x, norm_weight, out_qw, eps)
        return top1
      end

      rms_norm!(x, norm_weight, eps)
      logits = qmatvec_nobias(out_qw, x)
      maxv = logits.max
      {logits.index(maxv).not_nil!.to_i32, maxv}
    end

    private def top2_from_logits(logits : Array(Float32)) : {Int32, Float32, Int32, Float32}
      best = -Float32::INFINITY
      second = -Float32::INFINITY
      best_id = 0_i32
      second_id = 0_i32
      logits.each_with_index do |v, id|
        id32 = id.to_i32
        if v > best || (v == best && id32 < best_id)
          second = best
          second_id = best_id
          best = v
          best_id = id32
        elsif id32 != best_id && (v > second || (v == second && id32 < second_id))
          second = v
          second_id = id32
        end
      end
      {best_id, best, second_id, second}
    end

    private def top1_from_allowed_logits(logits : Array(Float32),
                                         allowed_ids : Array(Int32)) : {Int32, Float32}
      best = -Float32::INFINITY
      best_id = allowed_ids[0]
      allowed_ids.each do |id|
        value = logits[id]
        if value > best || (value == best && id < best_id)
          best = value
          best_id = id
        end
      end
      {best_id, best}
    end

    # Rank a grammar-certified token frontier from an already materialized
    # logit vector. This pure helper is also the reference ordering for the
    # opt-in full-head diagnostic below.
    def rank_allowed_logits(logits : Array(Float32),
                            allowed_ids : Array(Int32)) : Array(AllowedTokenScore)
      raise ArgumentError.new("allowed-token ranking requires at least one id") if allowed_ids.empty?

      ranked = allowed_ids.map do |id|
        raise ArgumentError.new("allowed token id #{id} out of range 0...#{logits.size}") if id < 0 || id >= logits.size
        value = logits[id]
        raise ArgumentError.new("allowed token id #{id} has a non-finite logit") unless value.finite?
        AllowedTokenScore.new(id, value)
      end
      ranked.sort! do |a, b|
        if a.logit == b.logit
          a.token_id <=> b.token_id
        elsif a.logit > b.logit
          -1
        else
          1
        end
      end
      ranked
    end

    # Greedy decode helper. By default, the Metal wave path avoids
    # materializing full lm-head logits and returns only top-1. Set
    # `QWEN35_HEAD_TOP1_FUSED=0` to force the full-logit fallback.
    def forward_top1(weights : Qwen35Weights, token_id : Int32, pos : Int32,
                     state : State) : {Int32, Float32}
      if packed = forward_decode_wave_routed(weights, token_id, pos, state, top1: true)
        if packed.size == 2
          return {packed[0].to_i32, packed[1]}
        end

        maxv = packed.max
        return {packed.index(maxv).not_nil!.to_i32, maxv}
      end

      logits = forward(weights, token_id, pos, state)
      maxv = logits.max
      {logits.index(maxv).not_nil!.to_i32, maxv}
    end

    # Constrained greedy helper. The returned token is the maximum-logit token
    # inside `allowed_ids`; callers must only pass grammar-certified ids.
    def forward_top1_allowed(weights : Qwen35Weights,
                             token_id : Int32,
                             pos : Int32,
                             state : State,
                             allowed_ids : Array(Int32)) : {Int32, Float32}
      raise ArgumentError.new("forward_top1_allowed requires at least one allowed id") if allowed_ids.empty?
      allowed_ids.each do |id|
        raise ArgumentError.new("allowed token id #{id} out of range 0...#{weights.output.out_dim}") if id < 0 || id >= weights.output.out_dim
      end

      {% unless flag?(:cpu_only) %}
        # Adaptive decode must consume the token exactly once. Preflight the
        # specialized allowed-token head before entering its whole-token wave;
        # an unsupported output format uses the exact full-logit wave instead.
        if state.adaptive_kv? &&
           !Qwen35Metal.rmsnorm_project_top1_allowed_ids_supported?(weights.output)
          return top1_from_allowed_logits(forward(weights, token_id, pos, state), allowed_ids)
        end
      {% end %}

      if packed = forward_decode_wave_routed(weights, token_id, pos, state, top1: true, top1_allowed_ids: allowed_ids)
        return {packed[0].to_i32, packed[1]} if packed.size == 2
        return top1_from_allowed_logits(packed, allowed_ids)
      end

      logits = forward(weights, token_id, pos, state)
      top1_from_allowed_logits(logits, allowed_ids)
    end

    # Opt-in diagnostic path for a constrained frontier. It executes exactly
    # one decoder step, like forward_top1_allowed, but materializes the full
    # head once so callers can inspect all allowed logits. Keep it out of the
    # default decode path because the fused allowed-token head is cheaper.
    def forward_rank_allowed(weights : Qwen35Weights,
                             token_id : Int32,
                             pos : Int32,
                             state : State,
                             allowed_ids : Array(Int32)) : Array(AllowedTokenScore)
      raise ArgumentError.new("forward_rank_allowed requires at least one allowed id") if allowed_ids.empty?
      allowed_ids.each do |id|
        raise ArgumentError.new("allowed token id #{id} out of range 0...#{weights.output.out_dim}") if id < 0 || id >= weights.output.out_dim
      end

      rank_allowed_logits(forward(weights, token_id, pos, state), allowed_ids)
    end

    # Greedy exact decode suffix with token handoff kept on the GPU.
    # Each wave writes its top1 id into `token_ids[i + 1]`; the next wave reads
    # that id as its embedding input. This is exact greedy decoding, but avoids
    # a CPU readback/wait between suffix tokens.
    def forward_top1_chain_gpu(weights : Qwen35Weights,
                               token_id : Int32,
                               pos : Int32,
                               state : State,
                               steps : Int32) : Array(Int32)?
      return [] of Int32 if steps <= 0
      {% unless flag?(:cpu_only) %}
        return nil unless Qwen35Metal.available?

        token_ids_buf = ML::MetalBuffer.new((steps + 1).to_i64 * sizeof(UInt32))
        token_ptr = token_ids_buf.contents.as(Pointer(UInt32))
        token_ptr[0] = token_id.to_u32

        submissions = [] of Qwen35Metal::DecodeWaveSubmission
        steps.times do |i|
          submission = forward_decode_wave_routed_async(weights, 0, pos + i, state,
            top1: true,
            fresh_scratch: true,
            token_ids_buf: token_ids_buf,
            token_index: i,
            top1_store_token_ids_buf: token_ids_buf,
            top1_store_index: i + 1)
          return nil if submission.nil?
          submissions << submission
        end
        submissions.each do |submission|
          submission.pending_cmds.each(&.wait)
          submission.cmd.wait
        end

        Array(Int32).new(steps) { |i| token_ptr[i + 1].to_i32 }
      {% else %}
        nil
      {% end %}
    end

    def forward_top2(weights : Qwen35Weights, token_id : Int32, pos : Int32,
                     state : State) : {Int32, Float32, Int32, Float32}
      if packed = forward_decode_wave_routed(weights, token_id, pos, state, top1: true, top2: true)
        if packed.size == 4
          return {packed[0].to_i32, packed[1], packed[2].to_i32, packed[3]}
        elsif packed.size == 2
          return {packed[0].to_i32, packed[1], -1_i32, -Float32::INFINITY}
        end

        return top2_from_logits(packed)
      end

      top2_from_logits(forward(weights, token_id, pos, state))
    end

    {% unless flag?(:cpu_only) %}
      # Experimental scheduling primitive for the two-lane decoder branch.
      # The caller owns state independence: do not submit multiple in-flight
      # waves that mutate the same KV/SSM buffers.
      def forward_top1_async(weights : Qwen35Weights,
                             token_id : Int32,
                             pos : Int32,
                             state : State,
                             fresh_scratch : Bool = true,
                             scratch_namespace : String? = nil) : Qwen35Metal::DecodeWaveSubmission?
        forward_decode_wave_routed_async(weights, token_id, pos, state, top1: true,
          fresh_scratch: fresh_scratch, scratch_namespace: scratch_namespace)
      end

      def wait_forward_top1(submission : Qwen35Metal::DecodeWaveSubmission) : {Int32, Float32}
        packed = Qwen35Metal.wait_forward_decode_wave(submission)
        raise "async top1 decode returned #{packed.size} values" unless packed.size == 2
        {packed[0].to_i32, packed[1]}
      end

      # Self-draft greedy decode with low-rank substitution on selected
      # recurrent layers. The caller owns the lifecycle of `lowrank_state_bufs`
      # and `lowrank_basis_bufs`; this routine just plumbs them into the wave.
      # Returns nil when the Metal wave path is unavailable.
      def forward_self_draft_top1(weights : Qwen35Weights,
                                  token_id : Int32,
                                  pos : Int32,
                                  state : State,
                                  lowrank_layer_indices : Set(Int32),
                                  lowrank_state_bufs : Hash(Int32, ML::MetalBuffer),
                                  lowrank_basis_bufs : Hash(Int32, ML::MetalBuffer),
                                  lowrank_rank : Int32,
                                  lowrank_skip_ffn : Bool = false,
                                  skip_recurrent_ffn : Bool = false,
                                  lowrank_skip_ffn_layer_indices : Set(Int32)? = nil,
                                  lowrank_updown_x_mean_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                  lowrank_updown_c_mean_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                  lowrank_updown_coeff_w_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                  lowrank_updown_down_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                  lowrank_updown_coeff_q8_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                  lowrank_updown_coeff_q8_scale_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                  lowrank_updown_down_q8_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                  lowrank_updown_down_q8_scale_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                  lowrank_updown_rank : Int32 = 0,
                                  lowrank_updown_layer_indices : Set(Int32)? = nil) : {Int32, Float32}?
        submission = forward_decode_wave_routed_async(weights, token_id, pos, state,
          top1: true, emit_head: true,
          lowrank_layer_indices: lowrank_layer_indices,
          lowrank_state_bufs: lowrank_state_bufs,
          lowrank_basis_bufs: lowrank_basis_bufs,
          lowrank_rank: lowrank_rank,
          lowrank_skip_ffn: lowrank_skip_ffn,
          skip_recurrent_ffn: skip_recurrent_ffn,
          lowrank_skip_ffn_layer_indices: lowrank_skip_ffn_layer_indices,
          lowrank_updown_x_mean_bufs: lowrank_updown_x_mean_bufs,
          lowrank_updown_c_mean_bufs: lowrank_updown_c_mean_bufs,
          lowrank_updown_coeff_w_bufs: lowrank_updown_coeff_w_bufs,
          lowrank_updown_down_bufs: lowrank_updown_down_bufs,
          lowrank_updown_coeff_q8_bufs: lowrank_updown_coeff_q8_bufs,
          lowrank_updown_coeff_q8_scale_bufs: lowrank_updown_coeff_q8_scale_bufs,
          lowrank_updown_down_q8_bufs: lowrank_updown_down_q8_bufs,
          lowrank_updown_down_q8_scale_bufs: lowrank_updown_down_q8_scale_bufs,
          lowrank_updown_rank: lowrank_updown_rank,
          lowrank_updown_layer_indices: lowrank_updown_layer_indices)
        return nil unless submission
        packed = Qwen35Metal.wait_forward_decode_wave(submission)
        raise "self-draft decode returned #{packed.size} values" unless packed.size == 2
        {packed[0].to_i32, packed[1]}
      end

      def forward_self_draft_top2(weights : Qwen35Weights,
                                  token_id : Int32,
                                  pos : Int32,
                                  state : State,
                                  lowrank_layer_indices : Set(Int32),
                                  lowrank_state_bufs : Hash(Int32, ML::MetalBuffer),
                                  lowrank_basis_bufs : Hash(Int32, ML::MetalBuffer),
                                  lowrank_rank : Int32,
                                  lowrank_skip_ffn : Bool = false,
                                  skip_recurrent_ffn : Bool = false,
                                  lowrank_skip_ffn_layer_indices : Set(Int32)? = nil,
                                  lowrank_updown_x_mean_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                  lowrank_updown_c_mean_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                  lowrank_updown_coeff_w_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                  lowrank_updown_down_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                  lowrank_updown_coeff_q8_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                  lowrank_updown_coeff_q8_scale_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                  lowrank_updown_down_q8_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                  lowrank_updown_down_q8_scale_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                  lowrank_updown_rank : Int32 = 0,
                                  lowrank_updown_layer_indices : Set(Int32)? = nil) : {Int32, Float32, Int32, Float32}?
        submission = forward_decode_wave_routed_async(weights, token_id, pos, state,
          top1: true, top2: true, emit_head: true,
          lowrank_layer_indices: lowrank_layer_indices,
          lowrank_state_bufs: lowrank_state_bufs,
          lowrank_basis_bufs: lowrank_basis_bufs,
          lowrank_rank: lowrank_rank,
          lowrank_skip_ffn: lowrank_skip_ffn,
          skip_recurrent_ffn: skip_recurrent_ffn,
          lowrank_skip_ffn_layer_indices: lowrank_skip_ffn_layer_indices,
          lowrank_updown_x_mean_bufs: lowrank_updown_x_mean_bufs,
          lowrank_updown_c_mean_bufs: lowrank_updown_c_mean_bufs,
          lowrank_updown_coeff_w_bufs: lowrank_updown_coeff_w_bufs,
          lowrank_updown_down_bufs: lowrank_updown_down_bufs,
          lowrank_updown_coeff_q8_bufs: lowrank_updown_coeff_q8_bufs,
          lowrank_updown_coeff_q8_scale_bufs: lowrank_updown_coeff_q8_scale_bufs,
          lowrank_updown_down_q8_bufs: lowrank_updown_down_q8_bufs,
          lowrank_updown_down_q8_scale_bufs: lowrank_updown_down_q8_scale_bufs,
          lowrank_updown_rank: lowrank_updown_rank,
          lowrank_updown_layer_indices: lowrank_updown_layer_indices)
        return nil unless submission
        packed = Qwen35Metal.wait_forward_decode_wave(submission)
        raise "self-draft top2 decode returned #{packed.size} values" unless packed.size == 4
        {packed[0].to_i32, packed[1], packed[2].to_i32, packed[3]}
      end

      def forward_self_draft_top1_from_token_buf_async(weights : Qwen35Weights,
                                                       token_ids_buf : ML::MetalBuffer,
                                                       token_index : Int32,
                                                       pos : Int32,
                                                       state : State,
                                                       lowrank_layer_indices : Set(Int32),
                                                       lowrank_state_bufs : Hash(Int32, ML::MetalBuffer),
                                                       lowrank_basis_bufs : Hash(Int32, ML::MetalBuffer),
                                                       lowrank_rank : Int32,
                                                       lowrank_skip_ffn : Bool = false,
                                                       skip_recurrent_ffn : Bool = false,
                                                       lowrank_updown_x_mean_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                                       lowrank_updown_c_mean_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                                       lowrank_updown_coeff_w_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                                       lowrank_updown_down_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                                       lowrank_updown_coeff_q8_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                                       lowrank_updown_coeff_q8_scale_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                                       lowrank_updown_down_q8_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                                       lowrank_updown_down_q8_scale_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                                       lowrank_updown_rank : Int32 = 0,
                                                       lowrank_skip_ffn_layer_indices : Set(Int32)? = nil,
                                                       lowrank_updown_layer_indices : Set(Int32)? = nil,
                                                       scratch_namespace : String? = nil,
                                                       command_queue_name : String? = nil,
                                                       append_command_buffer : PrefillCommandBuffer? = nil) : Qwen35Metal::DecodeWaveSubmission?
        forward_decode_wave_routed_async(weights, 0, pos, state,
          top1: true, emit_head: true,
          scratch_namespace: scratch_namespace,
          lowrank_layer_indices: lowrank_layer_indices,
          lowrank_state_bufs: lowrank_state_bufs,
          lowrank_basis_bufs: lowrank_basis_bufs,
          lowrank_rank: lowrank_rank,
          lowrank_skip_ffn: lowrank_skip_ffn,
          skip_recurrent_ffn: skip_recurrent_ffn,
          lowrank_skip_ffn_layer_indices: lowrank_skip_ffn_layer_indices,
          lowrank_updown_x_mean_bufs: lowrank_updown_x_mean_bufs,
          lowrank_updown_c_mean_bufs: lowrank_updown_c_mean_bufs,
          lowrank_updown_coeff_w_bufs: lowrank_updown_coeff_w_bufs,
          lowrank_updown_down_bufs: lowrank_updown_down_bufs,
          lowrank_updown_coeff_q8_bufs: lowrank_updown_coeff_q8_bufs,
          lowrank_updown_coeff_q8_scale_bufs: lowrank_updown_coeff_q8_scale_bufs,
          lowrank_updown_down_q8_bufs: lowrank_updown_down_q8_bufs,
          lowrank_updown_down_q8_scale_bufs: lowrank_updown_down_q8_scale_bufs,
          lowrank_updown_rank: lowrank_updown_rank,
          lowrank_updown_layer_indices: lowrank_updown_layer_indices,
          token_ids_buf: token_ids_buf,
          token_index: token_index,
          command_queue_name: command_queue_name,
          append_command_buffer: append_command_buffer)
      end

      def forward_self_draft_top2_from_token_buf_async(weights : Qwen35Weights,
                                                       token_ids_buf : ML::MetalBuffer,
                                                       token_index : Int32,
                                                       pos : Int32,
                                                       state : State,
                                                       lowrank_layer_indices : Set(Int32),
                                                       lowrank_state_bufs : Hash(Int32, ML::MetalBuffer),
                                                       lowrank_basis_bufs : Hash(Int32, ML::MetalBuffer),
                                                       lowrank_rank : Int32,
                                                       lowrank_skip_ffn : Bool = false,
                                                       skip_recurrent_ffn : Bool = false,
                                                       lowrank_updown_x_mean_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                                       lowrank_updown_c_mean_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                                       lowrank_updown_coeff_w_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                                       lowrank_updown_down_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                                       lowrank_updown_coeff_q8_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                                       lowrank_updown_coeff_q8_scale_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                                       lowrank_updown_down_q8_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                                       lowrank_updown_down_q8_scale_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                                       lowrank_updown_rank : Int32 = 0,
                                                       lowrank_skip_ffn_layer_indices : Set(Int32)? = nil,
                                                       lowrank_updown_layer_indices : Set(Int32)? = nil,
                                                       scratch_namespace : String? = nil,
                                                       command_queue_name : String? = nil,
                                                       append_command_buffer : PrefillCommandBuffer? = nil) : Qwen35Metal::DecodeWaveSubmission?
        forward_decode_wave_routed_async(weights, 0, pos, state,
          top1: true, top2: true, emit_head: true,
          scratch_namespace: scratch_namespace,
          lowrank_layer_indices: lowrank_layer_indices,
          lowrank_state_bufs: lowrank_state_bufs,
          lowrank_basis_bufs: lowrank_basis_bufs,
          lowrank_rank: lowrank_rank,
          lowrank_skip_ffn: lowrank_skip_ffn,
          skip_recurrent_ffn: skip_recurrent_ffn,
          lowrank_skip_ffn_layer_indices: lowrank_skip_ffn_layer_indices,
          lowrank_updown_x_mean_bufs: lowrank_updown_x_mean_bufs,
          lowrank_updown_c_mean_bufs: lowrank_updown_c_mean_bufs,
          lowrank_updown_coeff_w_bufs: lowrank_updown_coeff_w_bufs,
          lowrank_updown_down_bufs: lowrank_updown_down_bufs,
          lowrank_updown_coeff_q8_bufs: lowrank_updown_coeff_q8_bufs,
          lowrank_updown_coeff_q8_scale_bufs: lowrank_updown_coeff_q8_scale_bufs,
          lowrank_updown_down_q8_bufs: lowrank_updown_down_q8_bufs,
          lowrank_updown_down_q8_scale_bufs: lowrank_updown_down_q8_scale_bufs,
          lowrank_updown_rank: lowrank_updown_rank,
          lowrank_updown_layer_indices: lowrank_updown_layer_indices,
          token_ids_buf: token_ids_buf,
          token_index: token_index,
          command_queue_name: command_queue_name,
          append_command_buffer: append_command_buffer)
      end

      def forward_self_draft_state_from_token_buf_async(weights : Qwen35Weights,
                                                        token_ids_buf : ML::MetalBuffer,
                                                        token_index : Int32,
                                                        pos : Int32,
                                                        state : State,
                                                        lowrank_layer_indices : Set(Int32),
                                                        lowrank_state_bufs : Hash(Int32, ML::MetalBuffer),
                                                        lowrank_basis_bufs : Hash(Int32, ML::MetalBuffer),
                                                        lowrank_rank : Int32,
                                                        lowrank_skip_ffn : Bool = false,
                                                        skip_recurrent_ffn : Bool = false,
                                                        lowrank_updown_x_mean_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                                        lowrank_updown_c_mean_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                                        lowrank_updown_coeff_w_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                                        lowrank_updown_down_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                                        lowrank_updown_coeff_q8_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                                        lowrank_updown_coeff_q8_scale_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                                        lowrank_updown_down_q8_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                                        lowrank_updown_down_q8_scale_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                                        lowrank_updown_rank : Int32 = 0,
                                                        lowrank_skip_ffn_layer_indices : Set(Int32)? = nil,
                                                        lowrank_updown_layer_indices : Set(Int32)? = nil,
                                                        scratch_namespace : String? = nil,
                                                        command_queue_name : String? = nil,
                                                        append_command_buffer : PrefillCommandBuffer? = nil) : Qwen35Metal::DecodeWaveSubmission?
        forward_decode_wave_routed_async(weights, 0, pos, state,
          top1: false, emit_head: false,
          scratch_namespace: scratch_namespace,
          lowrank_layer_indices: lowrank_layer_indices,
          lowrank_state_bufs: lowrank_state_bufs,
          lowrank_basis_bufs: lowrank_basis_bufs,
          lowrank_rank: lowrank_rank,
          lowrank_skip_ffn: lowrank_skip_ffn,
          skip_recurrent_ffn: skip_recurrent_ffn,
          lowrank_skip_ffn_layer_indices: lowrank_skip_ffn_layer_indices,
          lowrank_updown_x_mean_bufs: lowrank_updown_x_mean_bufs,
          lowrank_updown_c_mean_bufs: lowrank_updown_c_mean_bufs,
          lowrank_updown_coeff_w_bufs: lowrank_updown_coeff_w_bufs,
          lowrank_updown_down_bufs: lowrank_updown_down_bufs,
          lowrank_updown_coeff_q8_bufs: lowrank_updown_coeff_q8_bufs,
          lowrank_updown_coeff_q8_scale_bufs: lowrank_updown_coeff_q8_scale_bufs,
          lowrank_updown_down_q8_bufs: lowrank_updown_down_q8_bufs,
          lowrank_updown_down_q8_scale_bufs: lowrank_updown_down_q8_scale_bufs,
          lowrank_updown_rank: lowrank_updown_rank,
          lowrank_updown_layer_indices: lowrank_updown_layer_indices,
          token_ids_buf: token_ids_buf,
          token_index: token_index,
          command_queue_name: command_queue_name,
          append_command_buffer: append_command_buffer)
      end
    {% end %}

    # Prefill helper for prompt tokens whose logits are not needed.
    #
    # This is exact for autoregressive state construction: every layer still
    # runs and updates full-attention KV plus DeltaNet conv/SSM state, but the
    # expensive output RMSNorm/lm-head projection is skipped. Use `forward` or
    # `forward_top1` for the final prompt token when next-token logits are
    # required.
    def prefill_token(weights : Qwen35Weights, token_id : Int32, pos : Int32,
                      state : State) : Nil
      if forward_decode_wave_routed(weights, token_id, pos, state, emit_head: false)
        return
      end
      if state.kv_cache_f16?
        raise ArgumentError.new("F16 KV cache requires the whole-token Metal decode route")
      end

      hp = weights.hparams
      max_seq = state.max_seq
      x = embedding_lookup(weights.token_embd, token_id)

      weights.layers.each_with_index do |lw, il|
        case lw
        in Qwen35FullAttnWeights
          x = forward_full_attn_layer(x, pos, lw, state.layers[il], hp, max_seq)
        in Qwen35RecurrentWeights
          x = forward_recurrent_layer(x, pos, lw, state.layers[il], hp, max_seq)
        end
      end
    end

    # Prefill a known prompt span whose logits are not observed.
    #
    # This is exact for intermediate prompt tokens. It processes recurrent
    # layers in token chunks and falls back to serial full-attention layers,
    # because full-attention prefill still needs a dedicated causal chunk path.
    def prefill_tokens(weights : Qwen35Weights,
                       token_ids : Array(Int32),
                       start_pos : Int32,
                       state : State) : Nil
      return if token_ids.empty?
      if token_ids.size > 1 && prefill_gc_guard_enabled? && !@@prefill_gc_guard_active
        with_prefill_gc_guard { prefill_tokens(weights, token_ids, start_pos, state) }
        return
      end

      prefill_tokens_hidden(weights, token_ids, start_pos, state, need_output: false)
    end

    # Prefill one prompt and return the complete next-token logits vector.
    #
    # For a position-zero, single-chunk prompt whose final layer is full
    # attention, the final layer only computes K/V for intermediate rows;
    # Q/attention/FFN are required only for the last row consumed by the output
    # head. Unsupported shapes fall back before mutation to the ordinary N-1
    # prefill plus decode.
    def prefill_tokens_logits(weights : Qwen35Weights,
                              token_ids : Array(Int32),
                              start_pos : Int32,
                              state : State,
                              terminal_last_used : Array(Bool)? = nil) : Array(Float32)
      raise ArgumentError.new("prefill_tokens_logits token_ids must not be empty") if token_ids.empty?
      terminal_last_used.try { |used| used[0] = false }
      if token_ids.size > 1 && prefill_gc_guard_enabled? && !@@prefill_gc_guard_active
        return with_prefill_gc_guard do
          prefill_tokens_logits(weights, token_ids, start_pos, state, terminal_last_used)
        end
      end

      if prefill_full_logits_last_supported?(weights, state, token_ids.size, start_pos)
        last_layer = weights.layers.last.as(Qwen35FullAttnWeights)
        prefix = prefill_tokens_hidden(
          weights, token_ids, start_pos, state,
          stop_layer: weights.layers.size - 1,
        )
        last = final_full_attn_layer_chunk_last_routed(
          prefix, token_ids.size, start_pos, state.layers.last,
          last_layer, weights.hparams, state.max_seq,
        )
        raise "terminal-row full-logit route failed after prompt state mutation" unless last
        terminal_last_used.try { |used| used[0] = true }

        hp = weights.hparams
        if logits = output_project_routed(last, weights.output_norm, weights.output, hp.rms_eps)
          return logits
        end
        rms_norm!(last, weights.output_norm, hp.rms_eps)
        return qmatvec_nobias(weights.output, last)
      end

      if token_ids.size > 1
        prefill_tokens(weights, token_ids[0...-1], start_pos, state)
      end
      forward(weights, token_ids[-1], start_pos + token_ids.size - 1, state)
    end

    def prefill_tokens_top1(weights : Qwen35Weights,
                            token_ids : Array(Int32),
                            start_pos : Int32,
                            state : State) : {Int32, Float32}
      raise ArgumentError.new("prefill_tokens_top1 token_ids must not be empty") if token_ids.empty?
      if token_ids.size > 1 && prefill_gc_guard_enabled? && !@@prefill_gc_guard_active
        return with_prefill_gc_guard { prefill_tokens_top1(weights, token_ids, start_pos, state) }
      end

      if ENV["QWEN35_PREFILL_FINAL_CHUNK_OFF"]? != "1" &&
         ENV["QWEN35_PREFILL_CHUNK_OFF"]? != "1" &&
         token_ids.size > 1
        chunk_size = prefill_chunk_size(state.adaptive_kv_layer_indices.any?)
        if token_ids.size > chunk_size
          if ENV["QWEN35_PREFILL_LONG_SUFFIX_OFF"]? != "1"
            prefix_len = token_ids.size - chunk_size
            if prefix_len > 0
              prefill_tokens(weights, token_ids[0, prefix_len], start_pos, state)
              return prefill_tokens_top1(weights, token_ids[prefix_len, token_ids.size - prefix_len], start_pos + prefix_len, state)
            end
          end
          prefill_tokens(weights, token_ids[0...-1], start_pos, state)
          return forward_top1(weights, token_ids[-1], start_pos + token_ids.size - 1, state)
        end

        {% unless flag?(:cpu_only) %}
          if prefill_adaptive_resident_top1_supported?(weights, state, token_ids.size)
            hp = weights.hparams
            resident_buf = ML::MetalBuffer.new(
              token_ids.size.to_i64 * hp.n_embd.to_i64 * sizeof(Float32),
            )
            resident_written = [false]
            append_top1 = ENV["QWEN35_PREFILL_TOP1_ADAPTIVE_APPEND"]? == "1"
            top1_id_buf = append_top1 ? ML::MetalBuffer.new(sizeof(UInt32).to_i64) : nil
            top1_value_buf = append_top1 ? ML::MetalBuffer.new(sizeof(Float32).to_i64) : nil
            top1_encoded = [false]
            resident_top1_append = if append_top1
                                     PrefillResidentTop1Append.new(
                                       top1_id_buf.not_nil!, top1_value_buf.not_nil!, top1_encoded,
                                     )
                                   end
            prefill_tokens_hidden(weights, token_ids, start_pos, state,
              need_output: false,
              resident_output_buf: resident_buf,
              resident_output_written: resident_written,
              resident_top1_append: resident_top1_append)
            raise "adaptive resident final prefill did not produce a GPU hidden buffer" unless resident_written[0]
            if append_top1
              raise "adaptive resident final prefill did not append the top-1 head" unless top1_encoded[0]
              packed = Qwen35Metal.read_head_top1_buffers(
                top1_id_buf.not_nil!, top1_value_buf.not_nil!,
              )
              Qwen35Metal::Profile.bump_route_marker("adaptive_final_resident_top1_appended")
              return {packed[0].to_i32, packed[1]}
            end
            if top1 = output_project_top1_resident_routed(
                 resident_buf, token_ids.size - 1, weights.output_norm, weights.output, hp.rms_eps,
               )
              Qwen35Metal::Profile.bump_route_marker("adaptive_final_resident_top1")
              return top1
            end

            # The state is already published, so never retry the decoder body.
            # A late head-policy change or helper rejection falls back by
            # materializing only the completed final hidden row.
            row_offset = (token_ids.size - 1).to_i64 * hp.n_embd.to_i64
            row_ptr = resident_buf.contents.as(Pointer(Float32)) + row_offset
            last_hidden = Array(Float32).new(hp.n_embd) { |i| row_ptr[i] }
            Qwen35Metal::Profile.bump_group_transfer(
              "adaptive_final_head.fallback", 0_i64, hp.n_embd.to_i64 * sizeof(Float32),
            )
            Qwen35Metal::Profile.bump_route_marker("adaptive_final_resident_top1_fallback")
            return hidden_top1(weights, last_hidden)
          end
        {% end %}

        if !state.layers[-1].adaptive_kv &&
           ENV["QWEN35_FINAL_FULL_LAST_OFF"]? != "1" &&
           (last_layer = weights.layers[-1].as?(Qwen35FullAttnWeights)) &&
           metal_qw_supported?(last_layer.attn_q_qw) &&
           metal_qw_supported?(last_layer.attn_k_qw) &&
           metal_qw_supported?(last_layer.attn_v_qw) &&
           metal_qw_supported?(last_layer.attn_output_qw) &&
           metal_qw_supported?(last_layer.ffn_gate_qw) &&
           metal_qw_supported?(last_layer.ffn_up_qw) &&
           metal_qw_supported?(last_layer.ffn_down_qw)
          if ENV["QWEN35_PREFILL_TOP1_RESIDENT_LAST"]? == "1" &&
             ENV["QWEN35_PREFILL_RESIDENT_BOUNDARY_OFF"]? != "1"
            hp = weights.hparams
            prefix_buf = ML::MetalBuffer.new((token_ids.size * hp.n_embd).to_i64 * sizeof(Float32))
            resident_written = [false]
            prefill_tokens_hidden(weights, token_ids, start_pos, state,
              stop_layer: weights.layers.size - 1,
              need_output: false,
              resident_output_buf: prefix_buf,
              resident_output_written: resident_written)
            raise "resident final-prefix prefill did not produce a GPU buffer" unless resident_written[0]
            if ENV["QWEN35_PREFILL_TOP1_FUSED_LAST"]? == "1"
              if top1 = final_full_attn_layer_chunk_last_top1_routed([] of Float32, token_ids.size, start_pos, state.layers[-1], last_layer, weights.output_norm, weights.output, hp, state.max_seq, input_buf: prefix_buf)
                return top1
              end
              raise "resident fused final-prefix top1 route failed after prefix state mutation"
            end
            if last = final_full_attn_layer_chunk_last_routed([] of Float32, token_ids.size, start_pos, state.layers[-1], last_layer, hp, state.max_seq, input_buf: prefix_buf)
              if top1 = output_project_top1_routed(last, weights.output_norm, weights.output, hp.rms_eps)
                return top1
              end
              rms_norm!(last, weights.output_norm, hp.rms_eps)
              logits = qmatvec_nobias(weights.output, last)
              maxv = logits.max
              return {logits.index(maxv).not_nil!.to_i32, maxv}
            end
            raise "resident final-prefix route failed after state mutation"
          end

          x_before_last = prefill_tokens_hidden(weights, token_ids, start_pos, state, stop_layer: weights.layers.size - 1)
          if last = final_full_attn_layer_chunk_last_routed(x_before_last, token_ids.size, start_pos, state.layers[-1], last_layer, weights.hparams, state.max_seq)
            hp = weights.hparams
            if top1 = output_project_top1_routed(last, weights.output_norm, weights.output, hp.rms_eps)
              return top1
            end
            rms_norm!(last, weights.output_norm, hp.rms_eps)
            logits = qmatvec_nobias(weights.output, last)
            maxv = logits.max
            return {logits.index(maxv).not_nil!.to_i32, maxv}
          end
        end

        x = prefill_tokens_hidden(weights, token_ids, start_pos, state)
        hp = weights.hparams
        last = x[(token_ids.size - 1) * hp.n_embd, hp.n_embd]
        if top1 = output_project_top1_routed(last, weights.output_norm, weights.output, hp.rms_eps)
          return top1
        end
        rms_norm!(last, weights.output_norm, hp.rms_eps)
        logits = qmatvec_nobias(weights.output, last)
        maxv = logits.max
        return {logits.index(maxv).not_nil!.to_i32, maxv}
      end

      if token_ids.size > 1
        prefill_tokens(weights, token_ids[0...-1], start_pos, state)
      end
      forward_top1(weights, token_ids[-1], start_pos + token_ids.size - 1, state)
    end

    # Prefill a known token span and choose the following token from a
    # grammar-certified frontier. On adaptive resident KV, the constrained
    # output head is appended to the final prefill command so the last known
    # token never needs a second decoder-body pass.
    def prefill_tokens_top1_allowed(weights : Qwen35Weights,
                                    token_ids : Array(Int32),
                                    start_pos : Int32,
                                    state : State,
                                    allowed_ids : Array(Int32)) : {Int32, Float32}
      raise ArgumentError.new("prefill_tokens_top1_allowed token_ids must not be empty") if token_ids.empty?
      raise ArgumentError.new("prefill_tokens_top1_allowed requires at least one allowed id") if allowed_ids.empty?
      allowed_ids.each do |id|
        raise ArgumentError.new("allowed token id #{id} out of range 0...#{weights.output.out_dim}") if id < 0 || id >= weights.output.out_dim
      end
      if token_ids.size > 1 && prefill_gc_guard_enabled? && !@@prefill_gc_guard_active
        return with_prefill_gc_guard do
          prefill_tokens_top1_allowed(weights, token_ids, start_pos, state, allowed_ids)
        end
      end

      if ENV["QWEN35_PREFILL_FINAL_CHUNK_OFF"]? != "1" &&
         ENV["QWEN35_PREFILL_CHUNK_OFF"]? != "1" &&
         token_ids.size > 1
        chunk_size = prefill_chunk_size(state.adaptive_kv_layer_indices.any?)
        if token_ids.size > chunk_size
          if ENV["QWEN35_PREFILL_LONG_SUFFIX_OFF"]? != "1"
            prefix_len = token_ids.size - chunk_size
            if prefix_len > 0
              prefill_tokens(weights, token_ids[0, prefix_len], start_pos, state)
              return prefill_tokens_top1_allowed(
                weights,
                token_ids[prefix_len, token_ids.size - prefix_len],
                start_pos + prefix_len,
                state,
                allowed_ids,
              )
            end
          end
        else
          {% unless flag?(:cpu_only) %}
            if prefill_adaptive_resident_top1_supported?(weights, state, token_ids.size) &&
               Qwen35Metal.rmsnorm_project_top1_allowed_ids_supported?(weights.output)
              hp = weights.hparams
              resident_buf = ML::MetalBuffer.new(
                token_ids.size.to_i64 * hp.n_embd.to_i64 * sizeof(Float32),
              )
              resident_written = [false]
              top1_id_buf = ML::MetalBuffer.new(sizeof(UInt32).to_i64)
              top1_value_buf = ML::MetalBuffer.new(sizeof(Float32).to_i64)
              top1_encoded = [false]
              resident_top1_append = PrefillResidentTop1Append.new(
                top1_id_buf, top1_value_buf, top1_encoded, allowed_ids,
              )
              prefill_tokens_hidden(weights, token_ids, start_pos, state,
                need_output: false,
                resident_output_buf: resident_buf,
                resident_output_written: resident_written,
                resident_top1_append: resident_top1_append)
              raise "adaptive resident allowed-head prefill did not produce a GPU hidden buffer" unless resident_written[0]
              raise "adaptive resident allowed-head prefill did not append the constrained head" unless top1_encoded[0]
              packed = Qwen35Metal.read_head_top1_buffers(top1_id_buf, top1_value_buf)
              Qwen35Metal::Profile.bump_route_marker("adaptive_final_resident_top1_allowed_appended")
              return {packed[0].to_i32, packed[1]}
            end
          {% end %}
        end
      end

      if token_ids.size > 1
        prefill_tokens(weights, token_ids[0...-1], start_pos, state)
      end
      forward_top1_allowed(
        weights, token_ids[-1], start_pos + token_ids.size - 1, state, allowed_ids,
      )
    end

    # Conservative exact-boundary path for durable checkpoint anchors. Unlike
    # the token-parallel prefill routes, every consumed token retires through
    # the ordinary decode path before the next state mutation. This is slower,
    # but full anchors are periodic and must never serialize padded tail state.
    def prefill_tokens_top1_sequential(weights : Qwen35Weights,
                                       token_ids : Array(Int32),
                                       start_pos : Int32,
                                       state : State) : {Int32, Float32}
      raise ArgumentError.new("prefill_tokens_top1_sequential token_ids must not be empty") if token_ids.empty?
      if token_ids.size > 1 && prefill_gc_guard_enabled? && !@@prefill_gc_guard_active
        return with_prefill_gc_guard do
          prefill_tokens_top1_sequential(weights, token_ids, start_pos, state)
        end
      end

      if token_ids.size > 1
        token_ids[0...-1].each_with_index do |token_id, i|
          forward_hidden(weights, token_id, start_pos + i, state)
        end
      end
      forward_top1(weights, token_ids[-1], start_pos + token_ids.size - 1, state)
    end

    # Process a known token span and return the greedy next-token prediction
    # after each consumed token. This is the exact target-verifier primitive for
    # greedy speculative decode: candidate[i+1] is checked against result[i].
    #
    # The decoder body is chunked through prefill_tokens_hidden, while the
    # lm-head top1 is still emitted per row. A future verifier can replace the
    # per-row head projection with a batched top1 kernel without changing the
    # speculative control flow.
    def prefill_tokens_top1s(weights : Qwen35Weights,
                             token_ids : Array(Int32),
                             start_pos : Int32,
                             state : State) : Array({Int32, Float32})
      raise ArgumentError.new("prefill_tokens_top1s token_ids must not be empty") if token_ids.empty?
      if token_ids.size > 1 && prefill_gc_guard_enabled? && !@@prefill_gc_guard_active
        return with_prefill_gc_guard { prefill_tokens_top1s(weights, token_ids, start_pos, state) }
      end

      if token_ids.size == 1
        return [forward_top1(weights, token_ids[0], start_pos, state)]
      end

      if ENV["QWEN35_PREFILL_CHUNK_OFF"]? == "1"
        return Array({Int32, Float32}).new(token_ids.size) do |i|
          forward_top1(weights, token_ids[i], start_pos + i, state)
        end
      end

      {% unless flag?(:cpu_only) %}
        if prefill_top1_resident_rows_supported?(weights, token_ids.size)
          hp = weights.hparams
          hidden_bytes = (token_ids.size * hp.n_embd).to_i64 * sizeof(Float32)
          resident_buf = ML::MetalBuffer.new(hidden_bytes)
          resident_written = [false]
          prefill_tokens_hidden(weights, token_ids, start_pos, state,
            need_output: false,
            resident_output_buf: resident_buf,
            resident_output_written: resident_written)
          raise "resident top1 verifier did not produce a GPU hidden buffer despite positive preflight" unless resident_written[0]
          if top1s = output_project_top1s_resident_routed(resident_buf, token_ids.size, weights.output_norm, weights.output, hp.rms_eps)
            return top1s
          end
          raise "resident top1 verifier route unavailable despite positive preflight"
        end
      {% end %}

      hidden = prefill_tokens_hidden(weights, token_ids, start_pos, state)
      hp = weights.hparams
      if top1s = output_project_top1s_routed(hidden, token_ids.size, weights.output_norm, weights.output, hp.rms_eps)
        return top1s
      end

      results = Array({Int32, Float32}).new(token_ids.size)
      token_ids.size.times do |i|
        row = hidden[i * hp.n_embd, hp.n_embd]
        if top1 = output_project_top1_routed(row, weights.output_norm, weights.output, hp.rms_eps)
          results << top1
        else
          rms_norm!(row, weights.output_norm, hp.rms_eps)
          logits = qmatvec_nobias(weights.output, row)
          maxv = logits.max
          results << {logits.index(maxv).not_nil!.to_i32, maxv}
        end
      end
      results
    end

    # Process a known span and return both pre-output-norm hidden rows and the
    # greedy next-token prediction after each consumed token. This is a probe
    # primitive for speculative verifier loops that need the exact next boundary
    # hidden without running the decoder body twice.
    def prefill_tokens_hidden_top1s(weights : Qwen35Weights,
                                    token_ids : Array(Int32),
                                    start_pos : Int32,
                                    state : State) : NamedTuple(hidden: Array(Float32), top1s: Array({Int32, Float32}))
      raise ArgumentError.new("prefill_tokens_hidden_top1s token_ids must not be empty") if token_ids.empty?
      if token_ids.size > 1 && prefill_gc_guard_enabled? && !@@prefill_gc_guard_active
        return with_prefill_gc_guard { prefill_tokens_hidden_top1s(weights, token_ids, start_pos, state) }
      end

      hp = weights.hparams
      if token_ids.size == 1
        hidden = forward_hidden(weights, token_ids[0], start_pos, state)
        return {hidden: hidden, top1s: [hidden_top1(weights, hidden)]}
      end

      if ENV["QWEN35_PREFILL_CHUNK_OFF"]? == "1"
        hidden = Array(Float32).new(token_ids.size * hp.n_embd, 0.0_f32)
        top1s = [] of {Int32, Float32}
        token_ids.each_with_index do |token_id, i|
          row = forward_hidden(weights, token_id, start_pos + i, state)
          hp.n_embd.times { |j| hidden[i * hp.n_embd + j] = row[j] }
          top1s << hidden_top1(weights, row)
        end
        return {hidden: hidden, top1s: top1s}
      end

      hidden = prefill_tokens_hidden(weights, token_ids, start_pos, state)
      top1s = if routed = output_project_top1s_routed(hidden, token_ids.size, weights.output_norm, weights.output, hp.rms_eps)
                routed
              else
                results = [] of {Int32, Float32}
                token_ids.size.times do |i|
                  row = hidden[i * hp.n_embd, hp.n_embd]
                  results << hidden_top1(weights, row)
                end
                results
              end
      {hidden: hidden, top1s: top1s}
    end

    def prefill_tokens_hidden_top1s_recurrent_checkpoint(weights : Qwen35Weights,
                                                         token_ids : Array(Int32),
                                                         start_pos : Int32,
                                                         state : State,
                                                         checkpoint_index : Int32,
                                                         checkpoint_state : State) : NamedTuple(hidden: Array(Float32), top1s: Array({Int32, Float32}))
      raise ArgumentError.new("prefill_tokens_hidden_top1s_recurrent_checkpoint token_ids must not be empty") if token_ids.empty?
      raise ArgumentError.new("prefill_tokens_hidden_top1s_recurrent_checkpoint checkpoint index out of range") unless checkpoint_index >= 0 && checkpoint_index < token_ids.size
      if token_ids.size > 1 && prefill_gc_guard_enabled? && !@@prefill_gc_guard_active
        return with_prefill_gc_guard { prefill_tokens_hidden_top1s_recurrent_checkpoint(weights, token_ids, start_pos, state, checkpoint_index, checkpoint_state) }
      end

      hidden = prefill_tokens_hidden(weights, token_ids, start_pos, state,
        checkpoint_index: checkpoint_index, checkpoint_state: checkpoint_state)
      hp = weights.hparams
      top1s = if routed = output_project_top1s_routed(hidden, token_ids.size, weights.output_norm, weights.output, hp.rms_eps)
                routed
              else
                results = [] of {Int32, Float32}
                token_ids.size.times do |i|
                  row = hidden[i * hp.n_embd, hp.n_embd]
                  results << hidden_top1(weights, row)
                end
                results
              end
      {hidden: hidden, top1s: top1s}
    end

    def prefill_tokens_hidden_top1s_recurrent_rollback_log(weights : Qwen35Weights,
                                                           token_ids : Array(Int32),
                                                           start_pos : Int32,
                                                           state : State,
                                                           checkpoint_index : Int32,
                                                           log_state : State) : NamedTuple(hidden: Array(Float32), top1s: Array({Int32, Float32}))
      raise ArgumentError.new("prefill_tokens_hidden_top1s_recurrent_rollback_log requires exactly one rollback token") unless checkpoint_index + 1 < token_ids.size
      if token_ids.size > 1 && prefill_gc_guard_enabled? && !@@prefill_gc_guard_active
        return with_prefill_gc_guard { prefill_tokens_hidden_top1s_recurrent_rollback_log(weights, token_ids, start_pos, state, checkpoint_index, log_state) }
      end

      hidden = prefill_tokens_hidden(weights, token_ids, start_pos, state,
        checkpoint_index: checkpoint_index, checkpoint_state: log_state, checkpoint_rollback_log: true)
      hp = weights.hparams
      top1s = if routed = output_project_top1s_routed(hidden, token_ids.size, weights.output_norm, weights.output, hp.rms_eps)
                routed
              else
                results = [] of {Int32, Float32}
                token_ids.size.times do |i|
                  row = hidden[i * hp.n_embd, hp.n_embd]
                  results << hidden_top1(weights, row)
                end
                results
              end
      {hidden: hidden, top1s: top1s}
    end

    # Exact known-span verifier with a recurrent-state checkpoint captured
    # after `checkpoint_index`. This is used by branch-guard experiments that
    # want one verifier pass for prefix+guard+suffix while still keeping an
    # exact recurrent resume point after a passed guard token.
    def prefill_tokens_top1s_recurrent_checkpoint(weights : Qwen35Weights,
                                                  token_ids : Array(Int32),
                                                  start_pos : Int32,
                                                  state : State,
                                                  checkpoint_index : Int32,
                                                  checkpoint_state : State) : Array({Int32, Float32})
      raise ArgumentError.new("prefill_tokens_top1s_recurrent_checkpoint token_ids must not be empty") if token_ids.empty?
      raise ArgumentError.new("prefill_tokens_top1s_recurrent_checkpoint checkpoint index out of range") unless checkpoint_index >= 0 && checkpoint_index < token_ids.size
      if token_ids.size > 1 && prefill_gc_guard_enabled? && !@@prefill_gc_guard_active
        return with_prefill_gc_guard { prefill_tokens_top1s_recurrent_checkpoint(weights, token_ids, start_pos, state, checkpoint_index, checkpoint_state) }
      end

      hidden = prefill_tokens_hidden(weights, token_ids, start_pos, state,
        checkpoint_index: checkpoint_index, checkpoint_state: checkpoint_state)
      hp = weights.hparams
      if top1s = output_project_top1s_routed(hidden, token_ids.size, weights.output_norm, weights.output, hp.rms_eps)
        return top1s
      end

      results = [] of {Int32, Float32}
      token_ids.size.times do |i|
        row = hidden[i * hp.n_embd, hp.n_embd]
        results << hidden_top1(weights, row)
      end
      results
    end

    # Prompt prefill with one recurrent rollback point and only the final
    # next-token projection. This avoids the per-row lm-head work of the
    # speculative-verifier checkpoint primitive when the caller needs a single
    # durable boundary candidate.
    def prefill_tokens_top1_recurrent_checkpoint(weights : Qwen35Weights,
                                                 token_ids : Array(Int32),
                                                 start_pos : Int32,
                                                 state : State,
                                                 checkpoint_index : Int32,
                                                 checkpoint_state : State) : {Int32, Float32}
      raise ArgumentError.new("prefill_tokens_top1_recurrent_checkpoint token_ids must not be empty") if token_ids.empty?
      unless checkpoint_index >= 0 && checkpoint_index < token_ids.size
        raise ArgumentError.new("prefill_tokens_top1_recurrent_checkpoint checkpoint index out of range")
      end
      if token_ids.size > 1 && prefill_gc_guard_enabled? && !@@prefill_gc_guard_active
        return with_prefill_gc_guard do
          prefill_tokens_top1_recurrent_checkpoint(
            weights, token_ids, start_pos, state, checkpoint_index, checkpoint_state
          )
        end
      end

      hidden = prefill_tokens_hidden(
        weights,
        token_ids,
        start_pos,
        state,
        checkpoint_index: checkpoint_index,
        checkpoint_state: checkpoint_state,
      )
      hp = weights.hparams
      last = hidden[(token_ids.size - 1) * hp.n_embd, hp.n_embd]
      hidden_top1(weights, last)
    end

    # Process a known prompt span and return the final pre-output-norm hidden.
    # This is useful for MTP/self-draft probes that need the exact target
    # hidden at the prompt boundary without paying an extra lm-head pass.
    def prefill_tokens_last_hidden(weights : Qwen35Weights,
                                   token_ids : Array(Int32),
                                   start_pos : Int32,
                                   state : State) : Array(Float32)
      raise ArgumentError.new("prefill_tokens_last_hidden token_ids must not be empty") if token_ids.empty?
      if token_ids.size > 1 && prefill_gc_guard_enabled? && !@@prefill_gc_guard_active
        return with_prefill_gc_guard { prefill_tokens_last_hidden(weights, token_ids, start_pos, state) }
      end

      return forward_hidden(weights, token_ids[0], start_pos, state) if token_ids.size == 1

      hidden = prefill_tokens_hidden(weights, token_ids, start_pos, state)
      hp = weights.hparams
      rows = hidden.size // hp.n_embd
      raise "prefill_tokens_last_hidden: invalid hidden rows" if rows <= 0
      hidden[(rows - 1) * hp.n_embd, hp.n_embd]
    end

    private def prefill_tokens_hidden(weights : Qwen35Weights,
                                      token_ids : Array(Int32),
                                      start_pos : Int32,
                                      state : State,
                                      stop_layer : Int32? = nil,
                                      checkpoint_index : Int32? = nil,
                                      checkpoint_state : State? = nil,
                                      checkpoint_rollback_log : Bool = false,
                                      need_output : Bool = true,
                                      resident_output_buf : ML::MetalBuffer? = nil,
                                      resident_output_written : Array(Bool)? = nil,
                                      resident_top1_append : PrefillResidentTop1Append? = nil,
                                      shared_command_completed : Array(Bool)? = nil) : Array(Float32)
      raise ArgumentError.new("prefill_tokens_hidden token_ids must not be empty") if token_ids.empty?
      if resident_top1_append
        if need_output || resident_output_buf.nil? || resident_output_written.nil?
          raise ArgumentError.new("resident top-1 append requires a resident output-only prefill")
        end
        unless resident_top1_append.not_nil!.encoded.size == 1
          raise ArgumentError.new("resident top-1 append requires one completion marker")
        end
      end
      checkpoint_requested = !checkpoint_index.nil? || !checkpoint_state.nil?
      if checkpoint_requested && state.adaptive_kv?
        raise ArgumentError.new("adaptive resident QBit KV checkpoint is unsupported")
      end
      if checkpoint_requested
        raise ArgumentError.new("prefill_tokens_hidden checkpoint requires both index and state") if checkpoint_index.nil? || checkpoint_state.nil?
        raise ArgumentError.new("prefill_tokens_hidden checkpoint index out of range") unless checkpoint_index.not_nil! >= 0 && checkpoint_index.not_nil! < token_ids.size
      end

      if state.kv_cache_f16?
        unless f16_kv_prefill_supported?(weights)
          raise ArgumentError.new("F16 KV cache requires the complete Metal prefill route")
        end
        if ENV["QWEN35_PREFILL_CHUNK_OFF"]? == "1" || token_ids.size == 1
          raise ArgumentError.new("F16 KV cache does not support serial prefill")
        end
      end

      if ENV["QWEN35_PREFILL_CHUNK_OFF"]? == "1" || token_ids.size == 1
        hp = weights.hparams
        max_seq = state.max_seq
        layer_limit = stop_layer || weights.layers.size
        if checkpoint_requested
          prepare_recurrent_state_metal!(checkpoint_state.not_nil!, hp, clear: false)
        end
        hidden = Array(Float32).new(token_ids.size * hp.n_embd, 0.0_f32)
        token_ids.each_with_index do |token_id, i|
          pos = start_pos + i
          x = embedding_lookup(weights.token_embd, token_id)
          layer_limit.times do |il|
            lw = weights.layers[il]
            case lw
            in Qwen35FullAttnWeights
              x = forward_full_attn_layer(x, pos, lw, state.layers[il], hp, max_seq)
            in Qwen35RecurrentWeights
              x = forward_recurrent_layer(x, pos, lw, state.layers[il], hp, max_seq)
            end
          end
          hp.n_embd.times { |j| hidden[i * hp.n_embd + j] = x[j] }
          if checkpoint_requested && i == checkpoint_index.not_nil!
            copy_state_metal_used!(checkpoint_state.not_nil!, state, hp, used_tokens: pos + 1, rec_only: true)
          end
        end
        return need_output ? hidden : [] of Float32
      end

      hp = weights.hparams
      if checkpoint_requested
        prepare_recurrent_state_metal!(checkpoint_state.not_nil!, hp, clear: false)
      end
      max_seq = state.max_seq
      n_tokens = token_ids.size
      raise ArgumentError.new("prefill span exceeds max_seq") if start_pos < 0 || start_pos + n_tokens > max_seq

      resident_adaptive = state.adaptive_kv_layer_indices.any?
      chunk_size = prefill_chunk_size(resident_adaptive)
      if n_tokens > chunk_size
        offset = 0
        x = nil.as(Array(Float32)?)
        while offset < n_tokens
          len = Math.min(chunk_size, n_tokens - offset)
          local_checkpoint_index = nil.as(Int32?)
          local_checkpoint_state = nil.as(State?)
          if cp = checkpoint_index
            if cp >= offset && cp < offset + len
              local_checkpoint_index = cp - offset
              local_checkpoint_state = checkpoint_state
            end
          end
          chunk_need_output = need_output && offset + len >= n_tokens
          chunk_shared_command_completed = [false]
          x = prefill_tokens_hidden(weights, token_ids[offset, len], start_pos + offset, state,
            stop_layer: stop_layer, checkpoint_index: local_checkpoint_index, checkpoint_state: local_checkpoint_state,
            checkpoint_rollback_log: checkpoint_rollback_log,
            need_output: chunk_need_output,
            shared_command_completed: chunk_shared_command_completed)
          offset += len
          boundary_cooldown_ms = prefill_chunk_boundary_cooldown_ms(
            len, offset < n_tokens, chunk_shared_command_completed[0],
          )
          sleep boundary_cooldown_ms.milliseconds if boundary_cooldown_ms > 0
        end
        return need_output ? x.not_nil! : [] of Float32
      end

      x = [] of Float32

      il = 0
      layer_limit = stop_layer || weights.layers.size
      gpu_hidden = nil.as(ML::MetalBuffer?)
      handoff_a = nil.as(ML::MetalBuffer?)
      handoff_b = nil.as(ML::MetalBuffer?)
      handoff_flip = false
      handoff_bytes = (n_tokens * hp.n_embd).to_i64 * sizeof(Float32)
      append_prefill_cmd = nil
      append_prefill_gpu_work = false
      append_prefill_group_count = 0
      append_prefill_group_limit = prefill_append_group_limit(n_tokens)
      prefill_boundary_profile = ENV["QWEN35_PREFILL_BOUNDARY_PROFILE"]? == "1"
      prefill_graph_depth = prefill_graph_max_inflight(
        resident_adaptive, checkpoint_requested, prefill_boundary_profile,
      )
      cooldown_device_name = ""
      cooldown_flash_d256 = false
      {% unless flag?(:cpu_only) %}
        if Qwen35Metal.available?
          cooldown_device_name = ML::Metal::Device.instance.name
          cooldown_flash_d256 = Qwen35Metal.prefill_attn_flash_d256_policy?(
            cooldown_device_name, start_pos, n_tokens, hp.n_head, hp.n_head_kv,
            hp.head_dim, state.kv_cache_f16?, resident_adaptive,
            ENV["QWEN35_PREFILL_ATTN_FLASH_D256"]?,
          )
        end
      {% end %}
      append_prefill_cooldown_ms = prefill_append_cooldown_policy_ms(
        device_name: cooldown_device_name,
        start_pos: start_pos,
        n_tokens: n_tokens,
        n_layer: hp.n_layer,
        layer_limit: layer_limit,
        n_embd: hp.n_embd,
        n_ff: hp.n_ff,
        n_head: hp.n_head,
        n_head_kv: hp.n_head_kv,
        head_dim: hp.head_dim,
        full_attention_interval: hp.full_attention_interval,
        kv_cache_f16: state.kv_cache_f16?,
        resident_adaptive: resident_adaptive,
        checkpoint_requested: checkpoint_requested,
        boundary_profile: prefill_boundary_profile,
        graph_depth: prefill_graph_depth,
        flash_d256: cooldown_flash_d256,
        model_capability: weights.output.q4_gemv_x16_capability,
        gguf_file_type: weights.gguf_file_type,
      )
      append_prefill_started = nil.as(Time::Instant?)
      pending_adaptive_caches = [] of QwenQBitAdaptiveResidentKV::Cache
      checkpoint_resident_ok = !checkpoint_requested || ENV["QWEN35_PREFILL_CHECKPOINT_RESIDENT"]? == "1"
      resident_boundary_ok = false
      prefill_graph_scratch_arena = nil.as(PrefillScratchArena?)
      {% unless flag?(:cpu_only) %}
        resident_boundary_ok = ENV["QWEN35_PREFILL_RESIDENT_BOUNDARY_OFF"]? != "1" && checkpoint_resident_ok
      {% end %}
      flush_prefill_cmd = -> { }
      {% unless flag?(:cpu_only) %}
        append_prefill_cmd = nil.as(ML::Metal::CommandBuffer?)
        prefill_graph_queue = nil.as(ML::Metal::GraphSubmissionQueue(ML::Metal::CommandBuffer, Qwen35Metal::Scratch::Arena)?)
        prefill_graph_lease = nil.as(ML::Metal::GraphSubmissionLease(ML::Metal::CommandBuffer, Qwen35Metal::Scratch::Arena)?)
        prefill_graph_pending_flights = [] of {ML::Metal::GraphSubmissionLease(ML::Metal::CommandBuffer, Qwen35Metal::Scratch::Arena), Array(QwenQBitAdaptiveResidentKV::Cache)}
        prefill_graph_submitted_gpu_work = false
      {% end %}
      {% unless flag?(:cpu_only) %}
        append_command_available = ENV["QWEN35_PREFILL_APPEND_CMD_OFF"]? != "1" &&
                                   resident_boundary_ok && Qwen35Metal.available?
        if prefill_graph_depth > 0 && !append_command_available
          raise ArgumentError.new("CogniGraph prefill enqueue requires the resident Metal append-command route")
        end
        begin
          if prefill_graph_depth > 0
            command_queue = ML::Metal::CommandQueue.new
            prefill_graph_queue = ML::Metal::GraphSubmissionQueue(ML::Metal::CommandBuffer, Qwen35Metal::Scratch::Arena).new(prefill_graph_depth) do |slot, sequence|
              ML::Metal::CommandBuffer.new(queue: command_queue)
            end
            prefill_graph_lease = prefill_graph_queue.not_nil!.begin_submission
            prefill_graph_scratch_arena = Qwen35Metal::Scratch::Arena.new(
              "qwen35_prefill:#{Thread.current.object_id}:#{prefill_graph_lease.not_nil!.sequence}",
            )
            prefill_graph_lease.not_nil!.retain(prefill_graph_scratch_arena.not_nil!, &.release)
            append_prefill_cmd = prefill_graph_lease.not_nil!.command
          elsif append_command_available
            append_prefill_cmd = ML::Metal::CommandBuffer.new
            append_prefill_started = Time.instant if prefill_boundary_profile
          end
        rescue ex
          cleanup_prefill_command_setup(prefill_graph_queue, append_prefill_cmd)
          raise ex
        end
        flush_prefill_cmd = -> {
          if queue = prefill_graph_queue
            begin
              if cmd = append_prefill_cmd
                lease = prefill_graph_lease.not_nil!
                if append_prefill_gpu_work
                  pending_adaptive_caches.each do |cache|
                    QwenQBitAdaptiveResidentKV.finalize_pending_append(cmd, cache)
                  end
                  prefill_graph_pending_flights << {lease, pending_adaptive_caches.dup}
                  queue.submit(lease)
                  prefill_graph_submitted_gpu_work = true
                else
                  queue.cancel(lease)
                end
              end
              append_prefill_cmd = nil
              prefill_graph_lease = nil
              prefill_graph_scratch_arena = nil
              until prefill_graph_pending_flights.empty?
                flight = prefill_graph_pending_flights.first
                lease = flight[0]
                caches = flight[1]
                queue.await(lease)
                QwenQBitAdaptiveResidentKV.finish_pending_appends!(caches, lease.command)
                prefill_graph_pending_flights.shift
              end
              if ENV["QWEN35_COGNIGRAPH_PREFILL_TRACE"]? == "1"
                STDERR.puts(
                  "qwen35_cognigraph_prefill depth=#{queue.max_in_flight} " \
                  "submitted=#{queue.submitted_count} completed=#{queue.completed_count} " \
                  "max_pending=#{queue.max_pending_seen} tokens=#{n_tokens} start_pos=#{start_pos}",
                )
              end
              if prefill_graph_submitted_gpu_work
                shared_command_completed.try { |flag| flag[0] = true }
              end
            rescue ex
              if cmd = append_prefill_cmd
                unless cmd.committed?
                  if lease = prefill_graph_lease
                    begin
                      queue.cancel(lease)
                    rescue
                    end
                  end
                end
                pending_adaptive_caches.reverse_each do |cache|
                  begin
                    QwenQBitAdaptiveResidentKV.cancel_pending_append!(cache, cmd)
                  rescue
                  end
                end
              end
              raise ex
            ensure
              pending_adaptive_caches.clear
              append_prefill_cmd = nil
              append_prefill_gpu_work = false
              append_prefill_group_count = 0
            end
          elsif cmd = append_prefill_cmd
            begin
              finalize_started = prefill_boundary_profile ? Time.instant : nil
              pending_adaptive_caches.each do |cache|
                QwenQBitAdaptiveResidentKV.finalize_pending_append(cmd, cache)
              end
              finalize_finished = prefill_boundary_profile ? Time.instant : nil
              gpu_elapsed_ms = 0.0_f64
              gpu_timed = prefill_boundary_profile && append_prefill_gpu_work
              if gpu_timed
                gpu_elapsed_ms = cmd.commit_and_wait_gpu_elapsed_seconds * 1000.0
              else
                cmd.commit
                cmd.wait
              end
              submit_wait_finished = prefill_boundary_profile ? Time.instant : nil
              QwenQBitAdaptiveResidentKV.finish_pending_appends!(pending_adaptive_caches, cmd)
              publish_finished = prefill_boundary_profile ? Time.instant : nil
              if prefill_boundary_profile
                finalize_started_value = finalize_started.not_nil!
                finalize_finished_value = finalize_finished.not_nil!
                submit_wait_finished_value = submit_wait_finished.not_nil!
                publish_finished_value = publish_finished.not_nil!
                encode_started = append_prefill_started || finalize_started_value
                STDERR.puts(String.build do |io|
                  io << "qwen35_prefill_boundary"
                  io << " start_pos=" << start_pos
                  io << " tokens=" << n_tokens
                  io << " caches=" << pending_adaptive_caches.size
                  io << " groups=" << append_prefill_group_count
                  io << " cooldown_ms=" << append_prefill_cooldown_ms
                  io << " encode_ms=" << (finalize_started_value - encode_started.not_nil!).total_milliseconds.round(3)
                  io << " finalize_ms=" << (finalize_finished_value - finalize_started_value).total_milliseconds.round(3)
                  io << " submit_wait_ms=" << (submit_wait_finished_value - finalize_finished_value).total_milliseconds.round(3)
                  io << " gpu_ms=" << gpu_elapsed_ms.round(3)
                  io << " gpu_timed=" << gpu_timed
                  io << " publish_ms=" << (publish_finished_value - submit_wait_finished_value).total_milliseconds.round(3)
                end)
              end
              shared_command_completed.try { |flag| flag[0] = true } if append_prefill_gpu_work
            rescue ex
              if !cmd.committed? || cmd.completed?
                pending_adaptive_caches.each do |cache|
                  QwenQBitAdaptiveResidentKV.cancel_pending_append!(cache, cmd)
                end
              end
              raise ex
            ensure
              pending_adaptive_caches.clear
              append_prefill_cmd = nil
              append_prefill_gpu_work = false
              append_prefill_group_count = 0
            end
          elsif pending_adaptive_caches.any?
            pending_adaptive_caches.clear
            raise ArgumentError.new("adaptive resident QBit KV lost its shared prefill command")
          end
        }
      {% else %}
        if prefill_graph_depth > 0
          raise ArgumentError.new("CogniGraph prefill enqueue requires Metal")
        end
      {% end %}
      begin
        gpu_q4_embedding = false
        {% unless flag?(:cpu_only) %}
          if ENV["QWEN35_PREFILL_GPU_Q4_EMBED_OFF"]? != "1" &&
             weights.token_embd.type.q4_k? && append_prefill_cmd
            token_ids.each do |token_id|
              unless token_id >= 0 && token_id < weights.token_embd.out_dim
                raise "embedding: token_id #{token_id} out of range"
              end
            end

            token_ids_buf = ML::MetalBuffer.new(n_tokens.to_i64 * sizeof(UInt32))
            token_ids_ptr = token_ids_buf.contents.as(Pointer(UInt32))
            token_ids.each_with_index { |token_id, index| token_ids_ptr[index] = token_id.to_u32 }
            embedding_buf = ML::MetalBuffer.new(handoff_bytes)
            embedding_enc = ML::Metal::ComputeEncoder.new(append_prefill_cmd.not_nil!)
            begin
              Qwen35Metal.encode_embedding_q4k_rows_to_buffer(
                embedding_enc, weights.token_embd, token_ids_buf, embedding_buf, n_tokens,
              )
            ensure
              embedding_enc.end_encoding
            end
            append_prefill_gpu_work = true
            gpu_hidden = embedding_buf
            gpu_q4_embedding = true
            Qwen35Metal::Profile.bump_route_marker("prefill_gpu_q4_embedding")
          end
        {% end %}
        unless gpu_q4_embedding
          x = Array(Float32).new(n_tokens * hp.n_embd, 0.0_f32)
          token_ids.each_with_index do |token_id, t|
            emb = embedding_lookup(weights.token_embd, token_id)
            hp.n_embd.times { |i| x[t * hp.n_embd + i] = emb[i] }
          end
        end

        while il < layer_limit
          lw = weights.layers[il]
          case lw
          in Qwen35FullAttnWeights
            fused_read_output = true
            fused_output_buf = nil.as(ML::MetalBuffer?)
            fused_resident_final = false
            # LTP/WBA corridor: carry hidden rows across supported prefill groups
            # without exposing them to the host unless this is the final requested output.
            if resident_boundary_ok
              predicted_run_end = il + 1
              while predicted_run_end < weights.layers.size
                break unless weights.layers[predicted_run_end].is_a?(Qwen35RecurrentWeights)
                predicted_run_end += 1
              end
              fused_read_output = need_output && predicted_run_end >= layer_limit
              if !fused_read_output && predicted_run_end < layer_limit
                if handoff_flip
                  handoff_a ||= ML::MetalBuffer.new(handoff_bytes)
                  fused_output_buf = handoff_a
                else
                  handoff_b ||= ML::MetalBuffer.new(handoff_bytes)
                  fused_output_buf = handoff_b
                end
                handoff_flip = !handoff_flip
              elsif !fused_read_output && predicted_run_end >= layer_limit
                if rb = resident_output_buf
                  fused_output_buf = rb
                  fused_resident_final = true
                end
              end
            end
            flush_prefill_cmd.call if fused_read_output && !state.layers[il].adaptive_kv

            if fused = full_attn_then_recurrent_chunk_project_many_routed(
                 x, n_tokens, start_pos, state, weights, il, hp, max_seq,
                 checkpoint_index: checkpoint_index, checkpoint_state: checkpoint_state,
                 checkpoint_rollback_log: checkpoint_rollback_log,
                 input_buf: gpu_hidden,
                 output_buf: fused_output_buf,
                 read_output: fused_read_output,
                 append_command_buffer: fused_read_output ? nil : append_prefill_cmd,
                 pending_adaptive_caches: pending_adaptive_caches,
                 scratch_arena: fused_read_output ? nil : prefill_graph_scratch_arena)
              fused_appended = !fused_read_output && !append_prefill_cmd.nil?
              append_prefill_gpu_work = true unless fused_read_output
              il = fused[1]
              if fused_read_output
                x = fused[0]
                gpu_hidden = nil
              elsif ob = fused_output_buf
                x = [] of Float32
                gpu_hidden = ob
                resident_output_written.try { |flag| flag[0] = true } if fused_resident_final
              else
                flush_prefill_cmd.call
                return [] of Float32 unless need_output
                x = fused[0]
                gpu_hidden = nil
              end
              if fused_appended && !gpu_hidden.nil? && append_prefill_group_limit > 0
                append_prefill_group_count += 1
                if append_prefill_group_count >= append_prefill_group_limit && il < layer_limit
                  {% unless flag?(:cpu_only) %}
                    if queue = prefill_graph_queue
                      cmd = append_prefill_cmd.not_nil!
                      lease = prefill_graph_lease.not_nil!
                      pending_adaptive_caches.each do |cache|
                        QwenQBitAdaptiveResidentKV.finalize_pending_append(cmd, cache)
                      end
                      prefill_graph_pending_flights << {lease, pending_adaptive_caches.dup}
                      queue.submit(lease)
                      prefill_graph_submitted_gpu_work = true
                      append_prefill_cmd = nil
                      prefill_graph_lease = nil
                      prefill_graph_scratch_arena = nil
                      pending_adaptive_caches.clear
                      append_prefill_gpu_work = false
                      append_prefill_group_count = 0
                      if queue.pending_count >= queue.max_in_flight
                        completed_flight = prefill_graph_pending_flights.first
                        completed_lease = completed_flight[0]
                        completed_caches = completed_flight[1]
                        queue.await(completed_lease)
                        QwenQBitAdaptiveResidentKV.finish_pending_appends!(completed_caches, completed_lease.command)
                        prefill_graph_pending_flights.shift
                      end
                      next_lease = queue.begin_submission
                      next_arena = Qwen35Metal::Scratch::Arena.new(
                        "qwen35_prefill:#{Thread.current.object_id}:#{next_lease.sequence}",
                      )
                      next_lease.retain(next_arena, &.release)
                      prefill_graph_lease = next_lease
                      prefill_graph_scratch_arena = next_arena
                      append_prefill_cmd = next_lease.command
                    else
                      flush_prefill_cmd.call
                      sleep append_prefill_cooldown_ms.milliseconds if append_prefill_cooldown_ms > 0
                      append_prefill_cmd = ML::Metal::CommandBuffer.new
                      append_prefill_started = Time.instant if prefill_boundary_profile
                    end
                  {% else %}
                    flush_prefill_cmd.call
                    sleep append_prefill_cooldown_ms.milliseconds if append_prefill_cooldown_ms > 0
                  {% end %}
                end
              end
              next
            end

            if state.layers[il].adaptive_kv
              cmd = append_prefill_cmd
              unless cmd
                raise ArgumentError.new("adaptive resident QBit KV lost its shared prefill command")
              end
              adaptive_read_output = need_output || il + 1 < layer_limit
              adaptive_output_buf = if adaptive_read_output
                                      ML::MetalBuffer.new(handoff_bytes)
                                    elsif rb = resident_output_buf
                                      rb
                                    else
                                      nil
                                    end
              adaptive_input = gpu_hidden ? [] of Float32 : x
              adaptive_result = full_attn_layer_chunk_project_routed(
                adaptive_input, n_tokens, start_pos, state.layers[il], lw, hp, max_seq,
                read_output: false,
                output_buf: adaptive_output_buf,
                input_buf: gpu_hidden,
                append_command_buffer: cmd,
                pending_adaptive_caches: pending_adaptive_caches,
                scratch_arena: prefill_graph_scratch_arena,
              )
              unless adaptive_result
                raise ArgumentError.new("adaptive resident QBit KV full-attention prefill route is unavailable")
              end
              append_prefill_gpu_work = true
              if top1_append = resident_top1_append
                if prefill_resident_top1_append_layer?(il, layer_limit)
                  unless adaptive_output_buf
                    pending_adaptive_caches.each do |cache|
                      QwenQBitAdaptiveResidentKV.cancel_pending_append!(cache, cmd)
                    end
                    pending_adaptive_caches.clear
                    raise ArgumentError.new("resident top-1 append did not reach the final adaptive output")
                  end
                  begin
                    encoded = with_prefill_scratch_arena(prefill_graph_scratch_arena) do
                      if allowed_ids = top1_append.allowed_ids
                        Qwen35Metal.encode_rmsnorm_project_top1_allowed_ids_buffer(
                          cmd, adaptive_output_buf.not_nil!,
                          (n_tokens - 1).to_i64 * hp.n_embd.to_i64,
                          weights.output_norm, weights.output, hp.rms_eps,
                          allowed_ids,
                          top1_append.id_buf, top1_append.value_buf,
                        )
                      else
                        Qwen35Metal.encode_rmsnorm_project_top1_buffer(
                          cmd, adaptive_output_buf.not_nil!,
                          (n_tokens - 1).to_i64 * hp.n_embd.to_i64,
                          weights.output_norm, weights.output, hp.rms_eps,
                          top1_append.id_buf, top1_append.value_buf,
                        )
                      end
                    end
                    unless encoded
                      pending_adaptive_caches.each do |cache|
                        QwenQBitAdaptiveResidentKV.cancel_pending_append!(cache, cmd)
                      end
                      pending_adaptive_caches.clear
                      raise ArgumentError.new("resident top-1 append encoder rejected the final adaptive output")
                    end
                    top1_append.encoded[0] = true
                  rescue ex
                    pending_adaptive_caches.each do |cache|
                      QwenQBitAdaptiveResidentKV.cancel_pending_append!(cache, cmd)
                    end
                    pending_adaptive_caches.clear
                    raise ex
                  end
                end
              end
              flush_prefill_cmd.call
              gpu_hidden = nil
              if adaptive_read_output
                {% unless flag?(:cpu_only) %}
                  Qwen35Metal::Profile.bump_group_transfer(
                    "adaptive_standalone.boundary", 0_i64, handoff_bytes,
                  )
                {% end %}
                x = adaptive_output_buf.not_nil!.read(n_tokens * hp.n_embd)
              else
                resident_output_written.try { |flag| flag[0] = true } if resident_output_buf
                x = [] of Float32
              end
              il += 1
              next
            end

            if gb = gpu_hidden
              full_output_buf = nil.as(ML::MetalBuffer?)
              if resident_boundary_ok && !need_output && il + 1 >= layer_limit
                full_output_buf = resident_output_buf
              end
              if !need_output && il + 1 >= layer_limit && !checkpoint_requested && full_output_buf.nil? &&
                 final_full_attn_layer_chunk_kv_cache_only_routed([] of Float32, n_tokens, start_pos, state.layers[il], lw, hp, max_seq,
                   input_buf: gb, append_command_buffer: append_prefill_cmd,
                   scratch_arena: prefill_graph_scratch_arena)
                append_prefill_gpu_work = true
                x = [] of Float32
                gpu_hidden = nil
                il += 1
                next
              end
              flush_prefill_cmd.call
              x = gb.read(n_tokens * hp.n_embd)
              gpu_hidden = nil
            elsif !need_output && il + 1 >= layer_limit && !checkpoint_requested && resident_output_buf.nil? &&
                  final_full_attn_layer_chunk_kv_cache_only_routed(x, n_tokens, start_pos, state.layers[il], lw, hp, max_seq,
                    append_command_buffer: append_prefill_cmd,
                    scratch_arena: prefill_graph_scratch_arena)
              append_prefill_gpu_work = true
              x = [] of Float32
              il += 1
              next
            end

            read_output = need_output || il + 1 < layer_limit
            full_resident_final = false
            full_output_buf = nil.as(ML::MetalBuffer?)
            if resident_boundary_ok && !read_output && il + 1 >= layer_limit
              if rb = resident_output_buf
                full_output_buf = rb
                full_resident_final = true
              end
            end
            flush_prefill_cmd.call if read_output
            if gpu_out = full_attn_layer_chunk_project_routed(x, n_tokens, start_pos, state.layers[il], lw, hp, max_seq, read_output: read_output, output_buf: full_output_buf)
              flush_prefill_cmd.call unless read_output
              if read_output
                x = gpu_out
              elsif full_resident_final
                resident_output_written.try { |flag| flag[0] = true }
                x = [] of Float32
              else
                return [] of Float32
              end
            else
              if state.layers[il].kv_cache_f16
                raise ArgumentError.new("F16 KV cache full-attention prefill route became unavailable")
              end
              out = Array(Float32).new(n_tokens * hp.n_embd, 0.0_f32)
              n_tokens.times do |t|
                row = x[t * hp.n_embd, hp.n_embd]
                y = forward_full_attn_layer(row, start_pos + t, lw, state.layers[il], hp, max_seq)
                hp.n_embd.times { |i| out[t * hp.n_embd + i] = y[i] }
              end
              flush_prefill_cmd.call unless read_output
              return [] of Float32 unless read_output
              x = out
            end
            il += 1
          in Qwen35RecurrentWeights
            if ENV["QWEN35_PREFILL_REC_RUN_OFF"]? != "1"
              run_end = il
              while run_end < weights.layers.size
                break unless weights.layers[run_end].is_a?(Qwen35RecurrentWeights)
                run_end += 1
              end

              if run_end - il > 1 || checkpoint_requested
                {% unless flag?(:cpu_only) %}
                  rec_layers = [] of Qwen35RecurrentWeights
                  conv_bufs = [] of ML::MetalBuffer
                  ssm_bufs = [] of ML::MetalBuffer
                  checkpoint_conv_bufs = [] of ML::MetalBuffer
                  checkpoint_ssm_bufs = [] of ML::MetalBuffer
                  h_k = hp.ssm_group_count
                  h_v = hp.ssm_time_step_rank
                  s = hp.ssm_state_size
                  qkv_dim = 2 * h_k * s + h_v * s
                  conv_k = hp.ssm_conv_kernel
                  supported = Qwen35Metal.available?

                  j = il
                  while j < run_end
                    rw = weights.layers[j].as(Qwen35RecurrentWeights)
                    supported &&= metal_qw_supported?(rw.attn_qkv_qw) &&
                                  metal_qw_supported?(rw.attn_gate_qw) &&
                                  metal_qw_supported?(rw.ssm_alpha_qw) &&
                                  metal_qw_supported?(rw.ssm_beta_qw) &&
                                  metal_qw_supported?(rw.ssm_out_qw) &&
                                  metal_qw_supported?(rw.ffn_gate_qw) &&
                                  metal_qw_supported?(rw.ffn_up_qw) &&
                                  metal_qw_supported?(rw.ffn_down_qw)
                    rec_layers << rw

                    lstate = state.layers[j]
                    conv_bytes = ((conv_k - 1) * qkv_dim).to_i64 * sizeof(Float32)
                    conv_buf = lstate.conv_state_buf
                    if conv_buf.nil?
                      conv_buf = ML::MetalBuffer.new(conv_bytes)
                      if conv_state = lstate.conv_state
                        conv_buf.write(conv_state)
                      else
                        conv_buf.contents.as(Pointer(UInt8)).clear(conv_bytes)
                      end
                      lstate.conv_state_buf = conv_buf
                    end
                    conv_bufs << conv_buf
                    if checkpoint_requested
                      checkpoint_conv_bufs << checkpoint_state.not_nil!.layers[j].conv_state_buf.not_nil!
                    end

                    ssm_bytes = (h_v * s * s).to_i64 * sizeof(Float32)
                    ssm_buf = lstate.ssm_state_buf
                    if ssm_buf.nil?
                      ssm_buf = ML::MetalBuffer.new(ssm_bytes)
                      if ssm_state = lstate.ssm_state
                        ssm_buf.write(ssm_state)
                      else
                        ssm_buf.contents.as(Pointer(UInt8)).clear(ssm_bytes)
                      end
                      lstate.ssm_state_buf = ssm_buf
                    end
                    ssm_bufs << ssm_buf
                    if checkpoint_requested
                      checkpoint_ssm_bufs << checkpoint_state.not_nil!.layers[j].ssm_state_buf.not_nil!
                    end
                    j += 1
                  end

                  if supported
                    rec_read_output = true
                    rec_output_buf = nil.as(ML::MetalBuffer?)
                    rec_resident_final = false
                    if resident_boundary_ok
                      rec_read_output = need_output && run_end >= layer_limit
                      if !rec_read_output && run_end < layer_limit
                        if handoff_flip
                          handoff_a ||= ML::MetalBuffer.new(handoff_bytes)
                          rec_output_buf = handoff_a
                        else
                          handoff_b ||= ML::MetalBuffer.new(handoff_bytes)
                          rec_output_buf = handoff_b
                        end
                        handoff_flip = !handoff_flip
                      elsif !rec_read_output && run_end >= layer_limit
                        if rb = resident_output_buf
                          rec_output_buf = rb
                          rec_resident_final = true
                        end
                      end
                    end
                    flush_prefill_cmd.call if rec_read_output
                    gpu_out = with_prefill_scratch_arena(rec_read_output ? nil : prefill_graph_scratch_arena) do
                      Qwen35Metal.recurrent_layer_chunk_project_many(
                        x, conv_bufs, ssm_bufs, rec_layers,
                        h_k, h_v, s, conv_k, n_tokens, hp.rms_eps,
                        "rec#{il}-#{run_end - 1}",
                        checkpoint_index: checkpoint_index,
                        checkpoint_conv_state_bufs: checkpoint_requested ? checkpoint_conv_bufs : nil,
                        checkpoint_ssm_state_bufs: checkpoint_requested ? checkpoint_ssm_bufs : nil,
                        checkpoint_rollback_log: checkpoint_rollback_log,
                        input_buf: gpu_hidden,
                        output_buf: rec_output_buf,
                        read_output: rec_read_output,
                        append_command_buffer: rec_read_output ? nil : append_prefill_cmd)
                    end
                    if gpu_out
                      append_prefill_gpu_work = true unless rec_read_output
                      il = run_end
                      if rec_read_output
                        x = gpu_out
                        gpu_hidden = nil
                      elsif ob = rec_output_buf
                        x = [] of Float32
                        gpu_hidden = ob
                        resident_output_written.try { |flag| flag[0] = true } if rec_resident_final
                      else
                        flush_prefill_cmd.call
                        return [] of Float32 unless need_output
                        x = gpu_out
                        gpu_hidden = nil
                      end
                      next
                    elsif checkpoint_requested
                      raise "prefill recurrent checkpoint unsupported for recurrent run #{il}..#{run_end - 1}"
                    end
                  elsif checkpoint_requested
                    raise "prefill recurrent checkpoint requires Metal-supported recurrent run #{il}..#{run_end - 1}"
                  end
                {% else %}
                  if checkpoint_requested
                    raise "prefill recurrent checkpoint requires Metal-supported recurrent run #{il}..#{run_end - 1}"
                  end
                {% end %}
              end
            elsif checkpoint_requested
              raise "prefill recurrent checkpoint requires QWEN35_PREFILL_REC_RUN_OFF != 1"
            end

            if gb = gpu_hidden
              flush_prefill_cmd.call
              x = gb.read(n_tokens * hp.n_embd)
              gpu_hidden = nil
            end
            x = forward_recurrent_layer_chunk(x, n_tokens, lw, state.layers[il], hp, max_seq)
            il += 1
          end
        end
        if gb = gpu_hidden
          flush_prefill_cmd.call
          return [] of Float32 unless need_output
          x = gb.read(n_tokens * hp.n_embd)
        end
        flush_prefill_cmd.call
        x
      rescue ex
        {% unless flag?(:cpu_only) %}
          if queue = prefill_graph_queue
            queue.abort unless queue.failed?
            cleanup_caches = pending_adaptive_caches.dup
            prefill_graph_pending_flights.each do |flight|
              flight[1].each { |cache| cleanup_caches << cache unless cleanup_caches.includes?(cache) }
            end
            cleanup_caches.each { |cache| QwenQBitAdaptiveResidentKV.discard_pending_appends!(cache) }
          elsif cmd = append_prefill_cmd
            unless cmd.committed?
              pending_adaptive_caches.reverse_each do |cache|
                begin
                  QwenQBitAdaptiveResidentKV.cancel_pending_append!(cache, cmd)
                rescue
                end
              end
              begin
                cmd.discard
              rescue
              end
            end
          end
        {% end %}
        raise ex
      end
    end

    private def f16_kv_prefill_supported?(weights : Qwen35Weights) : Bool
      {% if flag?(:cpu_only) %}
        false
      {% else %}
        return false if ENV["QWEN35_FULL_PREFILL_CHUNK_OFF"]? == "1"
        return false unless Qwen35Metal.available?
        return false unless Qwen35Metal.kv_cache_f16_pipelines_supported?

        weights.layers.all? do |layer|
          case layer
          in Qwen35FullAttnWeights
            metal_qw_supported?(layer.attn_q_qw) &&
              metal_qw_supported?(layer.attn_k_qw) &&
              metal_qw_supported?(layer.attn_v_qw) &&
              metal_qw_supported?(layer.attn_output_qw) &&
              metal_qw_supported?(layer.ffn_gate_qw) &&
              metal_qw_supported?(layer.ffn_up_qw) &&
              metal_qw_supported?(layer.ffn_down_qw)
          in Qwen35RecurrentWeights
            true
          end
        end
      {% end %}
    end

    # Embedding lookup for a single token id → Array(Float32)[n_embd].
    # token_embd is QuantWeight with dims [n_embd, vocab_size] (row = one embedding).
    def embedding_lookup(token_embd : QuantWeight, token_id : Int32) : Array(Float32)
      n_embd = token_embd.in_dim
      raise "embedding: token_id #{token_id} out of range" if token_id < 0 || token_id >= token_embd.out_dim

      # Dequantize just the one row. row_bytes depends on quant type.
      # Use slice into raw and dequantize full block-aligned segment.
      t = token_embd.type
      # K-quants use 256 elements per block; Q8_0 uses 32.
      if (t.q4_k? || t.q5_k? || t.q6_k?) && n_embd % 256 != 0
        raise "embedding: n_embd #{n_embd} not divisible by 256 for K-quant"
      elsif t.q8_0? && n_embd % 32 != 0
        raise "embedding: n_embd #{n_embd} not divisible by 32 for Q8_0"
      end
      row_bytes = case
                  when t.f32?  then n_embd * 4
                  when t.f16?  then n_embd * 2
                  when t.q4_k? then (n_embd // 256) * 144
                  when t.q5_k? then (n_embd // 256) * 176
                  when t.q6_k? then (n_embd // 256) * 210
                  when t.q8_0? then (n_embd // 32) * 34
                  else              raise "embedding: unsupported quant type #{t.name}"
                  end
      offset = token_id.to_i64 * row_bytes.to_i64
      row_slice = Bytes.new(token_embd.raw.to_unsafe + offset, row_bytes, read_only: true)
      Dequant.dequantize(row_slice, t, n_embd)
    end

    private def forward_decode_wave_routed(weights : Qwen35Weights,
                                           token_id : Int32,
                                           pos : Int32,
                                           state : State,
                                           top1 : Bool = false,
                                           top2 : Bool = false,
                                           emit_head : Bool = true,
                                           top1_allowed_ids : Array(Int32)? = nil) : Array(Float32)?
      {% unless flag?(:cpu_only) %}
        if state.adaptive_kv?
          return forward_adaptive_decode_wave_routed(
            weights, token_id, pos, state,
            top1: top1, top2: top2, emit_head: emit_head,
            top1_allowed_ids: top1_allowed_ids,
          )
        end
        if submission = forward_decode_wave_routed_async(weights, token_id, pos, state, top1: top1, top2: top2, emit_head: emit_head, top1_allowed_ids: top1_allowed_ids)
          return Qwen35Metal.wait_forward_decode_wave(submission)
        end
      {% end %}
      nil
    end

    private def forward_adaptive_decode_wave_routed(weights : Qwen35Weights,
                                                    token_id : Int32,
                                                    pos : Int32,
                                                    state : State,
                                                    top1 : Bool,
                                                    top2 : Bool,
                                                    emit_head : Bool,
                                                    top1_allowed_ids : Array(Int32)?) : Array(Float32)
      {% unless flag?(:cpu_only) %}
        adaptive_indices = state.adaptive_kv_layer_indices
        raise ArgumentError.new("adaptive resident QBit decode requires at least one selected layer") if adaptive_indices.empty?
        unless pos >= 0 && pos < state.max_seq
          raise ArgumentError.new("adaptive resident QBit decode position exceeds state capacity")
        end
        caches = adaptive_indices.map { |layer_index| state.layers[layer_index].adaptive_kv.not_nil! }
        caches.each do |cache|
          unless cache.cache_len == pos
            raise ArgumentError.new("adaptive resident QBit decode position does not match the live prefix")
          end
        end

        submission = forward_decode_wave_routed_async(
          weights, token_id, pos, state,
          top1: top1, top2: top2, emit_head: emit_head,
          top1_allowed_ids: top1_allowed_ids,
          adaptive_decode: true,
        )
        unless submission
          raise ArgumentError.new("adaptive resident QBit decode requires the whole-token Metal wave")
        end

        command = submission.cmd
        begin
          caches.each do |cache|
            QwenQBitAdaptiveResidentKV.finalize_pending_append(command, cache)
          end
          command.commit
          result = Qwen35Metal.wait_forward_decode_wave(submission)
          QwenQBitAdaptiveResidentKV.finish_pending_appends!(caches, command)
          result
        rescue ex
          if !command.committed? || command.completed?
            caches.each do |cache|
              QwenQBitAdaptiveResidentKV.cancel_pending_append!(cache, command)
            end
          end
          raise ex
        end
      {% else %}
        raise "Metal disabled (cpu_only)"
      {% end %}
    end

    private def forward_decode_wave_routed_async(weights : Qwen35Weights,
                                                 token_id : Int32,
                                                 pos : Int32,
                                                 state : State,
                                                 top1 : Bool = false,
                                                 top2 : Bool = false,
                                                 emit_head : Bool = true,
                                                 top1_allowed_ids : Array(Int32)? = nil,
                                                 fresh_scratch : Bool = false,
                                                 scratch_namespace : String? = nil,
                                                 lowrank_layer_indices : Set(Int32)? = nil,
                                                 lowrank_state_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                                 lowrank_basis_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                                 lowrank_rank : Int32 = 0,
                                                 lowrank_skip_ffn : Bool = false,
                                                 skip_recurrent_ffn : Bool = false,
                                                 lowrank_skip_ffn_layer_indices : Set(Int32)? = nil,
                                                 lowrank_updown_x_mean_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                                 lowrank_updown_c_mean_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                                 lowrank_updown_coeff_w_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                                 lowrank_updown_down_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                                 lowrank_updown_coeff_q8_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                                 lowrank_updown_coeff_q8_scale_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                                 lowrank_updown_down_q8_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                                 lowrank_updown_down_q8_scale_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                                 lowrank_updown_rank : Int32 = 0,
                                                 lowrank_updown_layer_indices : Set(Int32)? = nil,
                                                 token_ids_buf : ML::MetalBuffer? = nil,
                                                 token_index : Int32 = 0,
                                                 top1_store_token_ids_buf : ML::MetalBuffer? = nil,
                                                 top1_store_index : Int32 = -1,
                                                 command_queue_name : String? = nil,
                                                 append_command_buffer : PrefillCommandBuffer? = nil,
                                                 adaptive_decode : Bool = false)
      {% unless flag?(:cpu_only) %}
        adaptive_indices = state.adaptive_kv_layer_indices
        kv_cache_f16 = state.kv_cache_f16?
        if kv_cache_f16 && !adaptive_indices.empty?
          raise ArgumentError.new("adaptive resident QBit KV cannot coexist with an F16 KV owner")
        end
        if !adaptive_indices.empty? && !adaptive_decode
          raise ArgumentError.new("adaptive resident QBit KV decode requires the synchronous whole-token Metal route")
        end
        if adaptive_decode && adaptive_indices.empty?
          raise ArgumentError.new("adaptive resident QBit decode requires at least one selected layer")
        end
        return nil if ENV["QWEN35_DECODE_WAVE_OFF"]? == "1"
        return nil unless Qwen35Metal.available?
        return nil if kv_cache_f16 && !Qwen35Metal.kv_cache_f16_pipelines_supported?
        return nil unless metal_qw_supported?(weights.output)

        weights.layers.each do |lw|
          supported = case lw
                      in Qwen35FullAttnWeights
                        metal_qw_supported?(lw.attn_q_qw) &&
                          metal_qw_supported?(lw.attn_k_qw) &&
                          metal_qw_supported?(lw.attn_v_qw) &&
                          metal_qw_supported?(lw.attn_output_qw) &&
                          metal_qw_supported?(lw.ffn_gate_qw) &&
                          metal_qw_supported?(lw.ffn_up_qw) &&
                          metal_qw_supported?(lw.ffn_down_qw)
                      in Qwen35RecurrentWeights
                        metal_qw_supported?(lw.attn_qkv_qw) &&
                          metal_qw_supported?(lw.attn_gate_qw) &&
                          metal_qw_supported?(lw.ssm_alpha_qw) &&
                          metal_qw_supported?(lw.ssm_beta_qw) &&
                          metal_qw_supported?(lw.ssm_out_qw) &&
                          metal_qw_supported?(lw.ffn_gate_qw) &&
                          metal_qw_supported?(lw.ffn_up_qw) &&
                          metal_qw_supported?(lw.ffn_down_qw)
                      end
          return nil unless supported
        end

        hp = weights.hparams
        max_seq = state.max_seq
        kv_dim = hp.head_dim * hp.n_head_kv
        qkv_dim = 2 * hp.ssm_group_count * hp.ssm_state_size + hp.ssm_time_step_rank * hp.ssm_state_size

        k_cache_bufs = Array(ML::MetalBuffer?).new(hp.n_layer, nil)
        v_cache_bufs = Array(ML::MetalBuffer?).new(hp.n_layer, nil)
        conv_state_bufs = Array(ML::MetalBuffer?).new(hp.n_layer, nil)
        ssm_state_bufs = Array(ML::MetalBuffer?).new(hp.n_layer, nil)
        adaptive_encoders = nil.as(Qwen35Metal::AdaptiveDecodeEncoders?)

        if adaptive_decode
          encoders = {} of Int32 => Qwen35Metal::AdaptiveDecodeEncoder
          adaptive_indices.each do |selected|
            cache = state.layers[selected].adaptive_kv.not_nil!
            unless cache.cache_len == pos
              raise ArgumentError.new("adaptive resident QBit decode position does not match the live prefix")
            end
            encoders[selected] = adaptive_decode_encoder_for(cache, hp, pos)
          end
          adaptive_encoders = encoders
        end

        # Validate every externally supplied cache owner before allocating any
        # missing state or encoding work. Buffer length is not an admissible
        # substitute for the state's declared element type.
        weights.layers.each_with_index do |lw, il|
          next unless lw.is_a?(Qwen35FullAttnWeights)
          layer = state.layers[il]
          if adaptive_decode && adaptive_indices.includes?(il)
            if layer.k_cache || layer.v_cache || layer.k_cache_buf || layer.v_cache_buf
              raise ArgumentError.new("adaptive resident QBit KV cannot coexist with an ordinary KV owner")
            end
          else
            if layer.kv_cache_f16 && (layer.k_cache || layer.v_cache)
              raise ArgumentError.new("F16 KV cache cannot coexist with F32 host cache arrays")
            end
            expected_bytes = layer.kv_cache_bytes(max_seq, kv_dim)
            if (buffer = layer.k_cache_buf) && buffer.size != expected_bytes
              raise ArgumentError.new("K cache buffer size does not match its declared element type")
            end
            if (buffer = layer.v_cache_buf) && buffer.size != expected_bytes
              raise ArgumentError.new("V cache buffer size does not match its declared element type")
            end
          end
        end

        weights.layers.each_with_index do |lw, il|
          case lw
          in Qwen35FullAttnWeights
            if adaptive_decode && adaptive_indices.includes?(il)
              layer = state.layers[il]
              if layer.k_cache || layer.v_cache || layer.k_cache_buf || layer.v_cache_buf
                raise ArgumentError.new("adaptive resident QBit KV cannot coexist with an F32 owner")
              end
            else
              bytes = state.layers[il].kv_cache_bytes(max_seq, kv_dim)
              k_buf = state.layers[il].k_cache_buf
              if k_buf.nil?
                k_buf = ML::MetalBuffer.new(bytes)
                k_buf.contents.as(Pointer(UInt8)).clear(bytes)
                state.layers[il].k_cache_buf = k_buf
              end
              v_buf = state.layers[il].v_cache_buf
              if v_buf.nil?
                v_buf = ML::MetalBuffer.new(bytes)
                v_buf.contents.as(Pointer(UInt8)).clear(bytes)
                state.layers[il].v_cache_buf = v_buf
              end
              k_cache_bufs[il] = k_buf
              v_cache_bufs[il] = v_buf
            end
          in Qwen35RecurrentWeights
            conv_bytes = ((hp.ssm_conv_kernel - 1) * qkv_dim).to_i64 * sizeof(Float32)
            conv_buf = state.layers[il].conv_state_buf
            if conv_buf.nil?
              conv_buf = ML::MetalBuffer.new(conv_bytes)
              conv_buf.contents.as(Pointer(UInt8)).clear(conv_bytes)
              state.layers[il].conv_state_buf = conv_buf
            end
            ssm_bytes = (hp.ssm_time_step_rank * hp.ssm_state_size * hp.ssm_state_size).to_i64 * sizeof(Float32)
            ssm_buf = state.layers[il].ssm_state_buf
            if ssm_buf.nil?
              ssm_buf = ML::MetalBuffer.new(ssm_bytes)
              ssm_buf.contents.as(Pointer(UInt8)).clear(ssm_bytes)
              state.layers[il].ssm_state_buf = ssm_buf
            end
            conv_state_bufs[il] = conv_buf
            ssm_state_bufs[il] = ssm_buf
          end
        end

        emb = token_ids_buf ? nil : embedding_lookup(weights.token_embd, token_id)
        Qwen35Metal.forward_decode_wave_async(
          emb, weights.layers,
          k_cache_bufs, v_cache_bufs, conv_state_bufs, ssm_state_bufs,
          weights.output_norm, weights.output, hp, pos, top1: top1, emit_head: emit_head,
          top2: top2,
          top1_allowed_ids: top1_allowed_ids,
          fresh_scratch: fresh_scratch, scratch_namespace: scratch_namespace,
          lowrank_layer_indices: lowrank_layer_indices,
          lowrank_state_bufs: lowrank_state_bufs,
          lowrank_basis_bufs: lowrank_basis_bufs,
          lowrank_rank: lowrank_rank,
          lowrank_skip_ffn: lowrank_skip_ffn,
          skip_recurrent_ffn: skip_recurrent_ffn,
          lowrank_skip_ffn_layer_indices: lowrank_skip_ffn_layer_indices,
          lowrank_updown_x_mean_bufs: lowrank_updown_x_mean_bufs,
          lowrank_updown_c_mean_bufs: lowrank_updown_c_mean_bufs,
          lowrank_updown_coeff_w_bufs: lowrank_updown_coeff_w_bufs,
          lowrank_updown_down_bufs: lowrank_updown_down_bufs,
          lowrank_updown_coeff_q8_bufs: lowrank_updown_coeff_q8_bufs,
          lowrank_updown_coeff_q8_scale_bufs: lowrank_updown_coeff_q8_scale_bufs,
          lowrank_updown_down_q8_bufs: lowrank_updown_down_q8_bufs,
          lowrank_updown_down_q8_scale_bufs: lowrank_updown_down_q8_scale_bufs,
          lowrank_updown_rank: lowrank_updown_rank,
          lowrank_updown_layer_indices: lowrank_updown_layer_indices,
          token_embd_qw: weights.token_embd,
          token_ids_buf: token_ids_buf,
          token_index: token_index,
          top1_store_token_ids_buf: top1_store_token_ids_buf,
          top1_store_index: top1_store_index,
          command_queue_name: command_queue_name,
          append_command_buffer: append_command_buffer,
          adaptive_decode_encoders: adaptive_encoders,
          kv_cache_f16: kv_cache_f16)
      {% else %}
        nil
      {% end %}
    end

    {% unless flag?(:cpu_only) %}
      private def adaptive_decode_encoder_for(cache : QwenQBitAdaptiveResidentKV::Cache,
                                              hp : Qwen35Hparams,
                                              pos : Int32) : Qwen35Metal::AdaptiveDecodeEncoder
        ->(command : ML::Metal::CommandBuffer, q_source : ML::MetalBuffer, gate_source : ML::MetalBuffer, k_source : ML::MetalBuffer, v_source : ML::MetalBuffer, output : ML::MetalBuffer) do
          QwenQBitAdaptiveResidentKV.encode_prefill_chunk_and_append(
            command, cache,
            q_source, gate_source, k_source, v_source, output,
            1, hp.n_head, hp.n_head // hp.n_head_kv,
            (1.0 / Math.sqrt(hp.head_dim.to_f64)).to_f32,
            expected_start_token: pos,
          )
        end
      end
    {% end %}
  end
end
