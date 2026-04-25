# Metal-backed Q4_K matmul for Qwen 3.5/3.6 forward pass.
#
# Standalone wrapper — does NOT integrate into the BERT-specific compute graph
# in metal_backend.cr. Exposes a simple functional API that uploads input +
# weights, dispatches the kernel, downloads the result.
#
# Scope (Phase 2): correctness-first. Performance optimizations (persistent
# buffers, fused biases, half-input variants, compute graphs) come later.

require "./reader"
require "./qwen35_weights"

{% unless flag?(:cpu_only) %}
  require "../metal/device"
  require "../metal/dispatch"
  require "../core/buffer"
{% end %}

module ML
  module GGUF
    module Qwen35Metal
      Q4K_BLOCK_BYTES  = 144
      Q5K_BLOCK_BYTES  = 176
      Q6K_BLOCK_BYTES  = 210
      Q8_0_BLOCK_BYTES =  34
      QK_K             = 256
      Q8_0_QK          =  32

      # GEMV (decode) tiling — must match the quant-specific kernels.
      MV_Q4_NSG             =  2
      MV_Q4_NR0             =  2
      MV_Q5_NSG             =  2
      MV_Q5_NR0             =  1
      MV_Q6_NSG             =  2
      MV_Q6_NR0             =  1
      MV_Q8_NSG             =  4
      MV_Q8_NR0             =  1
      HEAD_TOP1_ROWS_PER_TG = 12

      # GEMM (prefill) tiling — Q4_K only for now.
      MM_NR0   =    64
      MM_NR1   =    32
      MM_TG    =   128 # threads per threadgroup (4 simdgroups × 32)
      MM_SHMEM = 12288 # bytes: 2 × (MM_SA_SIZE + MM_SB_SIZE) = 2 × 6144

      # Above this batch, use GEMM. At or below, GEMV is faster.
      GEMM_BATCH_THRESHOLD = 8

      # Reusing one F32->F16 activation conversion across FFN gate/up now
      # pays even at pp64 after the later H16 routing cleanups.
      Q4_PAIR_H16_MIN_BATCH = 64

      {% if flag?(:cpu_only) %}
        def self.available? : Bool
          false
        end

        def self.matmul_q4k(x : Array(Float32),
                            w_raw : Bytes,
                            in_dim : Int32,
                            out_dim : Int32,
                            batch : Int32) : Array(Float32)
          raise "Metal disabled (cpu_only)"
        end

        def self.matmul_q5k(x : Array(Float32),
                            w_raw : Bytes,
                            in_dim : Int32,
                            out_dim : Int32,
                            batch : Int32) : Array(Float32)
          raise "Metal disabled (cpu_only)"
        end

        def self.matmul_q6k(x : Array(Float32),
                            w_raw : Bytes,
                            in_dim : Int32,
                            out_dim : Int32,
                            batch : Int32) : Array(Float32)
          raise "Metal disabled (cpu_only)"
        end
      {% else %}
        GEMM_Q4K_SOURCE  = {{ read_file("#{__DIR__}/kernels/gemm_q4k.metal") }}
        GEMM_Q56K_SOURCE = {{ read_file("#{__DIR__}/kernels/gemm_q56k.metal") }}
        GEMM_MM_SOURCE   = {{ read_file("#{__DIR__}/kernels/gemm_mm.metal") }}
        DELTA_NET_SOURCE = {{ read_file("#{__DIR__}/kernels/delta_net.metal") }}
        ATTN_DECODE_SOURCE = {{ read_file("#{__DIR__}/kernels/attn_decode_qwen35.metal") }}
        FFN_SOURCE = {{ read_file("#{__DIR__}/kernels/ffn_qwen35.metal") }}
        RECURRENT_SOURCE = {{ read_file("#{__DIR__}/kernels/recurrent_qwen35.metal") }}
        FULLATTN_SOURCE = {{ read_file("#{__DIR__}/kernels/fullattn_qwen35.metal") }}

        @@mv_pipeline   : ML::Metal::ComputePipeline?
        @@mv_add_pipeline : ML::Metal::ComputePipeline?
        @@mm_pipeline   : ML::Metal::ComputePipeline?
        @@mm_h16_pipeline : ML::Metal::ComputePipeline?
        @@mm5_pipeline  : ML::Metal::ComputePipeline?
        @@mm6_pipeline  : ML::Metal::ComputePipeline?
        @@mm5_f32out_pipeline : ML::Metal::ComputePipeline?
        @@mm6_f32out_pipeline : ML::Metal::ComputePipeline?
        @@mm6_f32out_add_pipeline : ML::Metal::ComputePipeline?
        @@mv5_pipeline  : ML::Metal::ComputePipeline?
        @@mv6_pipeline  : ML::Metal::ComputePipeline?
        @@mv6_add_pipeline : ML::Metal::ComputePipeline?
        @@mv8_pipeline  : ML::Metal::ComputePipeline?
        @@mv8_add_pipeline : ML::Metal::ComputePipeline?
        @@mv8_dual_pipeline : ML::Metal::ComputePipeline?
        @@mv8_top1_tiles_pipeline : ML::Metal::ComputePipeline?
        @@mv6_top1_tiles_pipeline : ML::Metal::ComputePipeline?
        @@mv6_top1_tiles_batch_pipeline : ML::Metal::ComputePipeline?
        @@top1_reduce_tiles_pipeline : ML::Metal::ComputePipeline?
        @@top1_reduce_tiles_batch_pipeline : ML::Metal::ComputePipeline?
        @@top1_reduce_f16_rows_pipeline : ML::Metal::ComputePipeline?
        @@top2_reduce_f16_rows_pipeline : ML::Metal::ComputePipeline?
        @@dn_pipeline   : ML::Metal::ComputePipeline?
        @@dn128_pipeline : ML::Metal::ComputePipeline?
        @@dn128_fused_pipeline : ML::Metal::ComputePipeline?
        @@dn128_fused_post_pipeline : ML::Metal::ComputePipeline?
        @@dn128_chunk_fused_pipeline : ML::Metal::ComputePipeline?
        @@dn128_chunk_rowwise_pipeline : ML::Metal::ComputePipeline?
        @@dn_post_pipeline : ML::Metal::ComputePipeline?
        @@dn_post_chunk_pipeline : ML::Metal::ComputePipeline?
        @@attn_pipeline : ML::Metal::ComputePipeline?
        @@f32_to_f16_pipeline : ML::Metal::ComputePipeline?
        @@f16_to_f32_pipeline : ML::Metal::ComputePipeline?
        @@ffn_swiglu_pipeline : ML::Metal::ComputePipeline?
        @@add_rmsnorm_pipeline : ML::Metal::ComputePipeline?
        @@add_rmsnorm_rows_pipeline : ML::Metal::ComputePipeline?
        @@add_vec_pipeline : ML::Metal::ComputePipeline?
        @@rmsnorm_vec_pipeline : ML::Metal::ComputePipeline?
        @@rmsnorm_rows_pipeline : ML::Metal::ComputePipeline?
        @@recurrent_ab_pipeline : ML::Metal::ComputePipeline?
        @@recurrent_ab_chunk_pipeline : ML::Metal::ComputePipeline?
        @@recurrent_conv_pipeline : ML::Metal::ComputePipeline?
        @@recurrent_shift_pipeline : ML::Metal::ComputePipeline?
        @@recurrent_conv_shift_pipeline : ML::Metal::ComputePipeline?
        @@recurrent_conv_shift_chunk_pipeline : ML::Metal::ComputePipeline?
        @@recurrent_conv_shift_chunk_h16_pipeline : ML::Metal::ComputePipeline?
        @@l2_heads_pipeline : ML::Metal::ComputePipeline?
        @@l2_heads_chunk_pipeline : ML::Metal::ComputePipeline?
        @@split_qgate_pipeline : ML::Metal::ComputePipeline?
        @@split_qgate_rows_pipeline : ML::Metal::ComputePipeline?
        @@rmsnorm_heads_pipeline : ML::Metal::ComputePipeline?
        @@rmsnorm_heads_rows_pipeline : ML::Metal::ComputePipeline?
        @@rope_partial_pipeline : ML::Metal::ComputePipeline?
        @@rope_partial_rows_pipeline : ML::Metal::ComputePipeline?
        @@kv_write_pipeline : ML::Metal::ComputePipeline?
        @@kv_write_rows_pipeline : ML::Metal::ComputePipeline?
        @@attn_rows_pipeline : ML::Metal::ComputePipeline?

        # ── Phase 4.0 instrumentation ─────────────────────────────────
        # Counters and nanosecond timers broken down by dispatch type
        # and phase (encode / wait / read). Enable with Profile.enable!
        # before a region, query with Profile.report, reset between runs.
        module Profile
          @@enabled = false
          @@gemv_count    = 0_i64
          @@gemm_count    = 0_i64
          @@dn_count      = 0_i64
          @@attn_count    = 0_i64
          @@wave_count    = 0_i64
          @@cpu_fallback  = 0_i64
          @@gemv_encode_ns = 0_i64
          @@gemv_wait_ns   = 0_i64
          @@gemv_read_ns   = 0_i64
          @@gemm_wait_ns   = 0_i64
          @@dn_encode_ns   = 0_i64
          @@dn_wait_ns     = 0_i64
          @@dn_read_ns     = 0_i64
          @@attn_encode_ns = 0_i64
          @@attn_wait_ns   = 0_i64
          @@attn_read_ns   = 0_i64
          @@wave_encode_ns = 0_i64
          @@wave_wait_ns   = 0_i64
          @@wave_read_ns   = 0_i64
          @@trace_counts = Hash(String, Int64).new(0_i64)
          @@trace_ns = Hash(String, Int64).new(0_i64)
          @@group_counts = Hash(String, Int64).new(0_i64)
          @@group_encode_ns = Hash(String, Int64).new(0_i64)
          @@group_wait_ns = Hash(String, Int64).new(0_i64)
          @@group_read_ns = Hash(String, Int64).new(0_i64)
          @@matmul_counts = Hash(String, Int64).new(0_i64)
          @@matmul_weight_bytes = Hash(String, Int64).new(0_i64)
          @@conversion_counts = Hash(String, Int64).new(0_i64)
          @@conversion_bytes = Hash(String, Int64).new(0_i64)
          @@scope_stack = [] of String

          def self.enabled? : Bool; @@enabled end
          def self.enable!  : Nil ; @@enabled = true end
          def self.disable! : Nil ; @@enabled = false end

          def self.reset : Nil
            @@gemv_count = @@gemm_count = @@dn_count = @@attn_count = 0_i64
            @@wave_count = 0_i64
            @@cpu_fallback = 0_i64
            @@gemv_encode_ns = @@gemv_wait_ns = @@gemv_read_ns = 0_i64
            @@gemm_wait_ns = 0_i64
            @@dn_encode_ns = @@dn_wait_ns = @@dn_read_ns = 0_i64
            @@attn_encode_ns = @@attn_wait_ns = @@attn_read_ns = 0_i64
            @@wave_encode_ns = @@wave_wait_ns = @@wave_read_ns = 0_i64
            @@trace_counts.clear
            @@trace_ns.clear
            @@group_counts.clear
            @@group_encode_ns.clear
            @@group_wait_ns.clear
            @@group_read_ns.clear
            @@matmul_counts.clear
            @@matmul_weight_bytes.clear
            @@conversion_counts.clear
            @@conversion_bytes.clear
            @@scope_stack.clear
          end

          # Sampling hooks — cheap branches, no-op when disabled.
          def self.bump_gemv(encode_ns : Int64, wait_ns : Int64, read_ns : Int64)
            return unless @@enabled
            @@gemv_count += 1
            @@gemv_encode_ns += encode_ns
            @@gemv_wait_ns   += wait_ns
            @@gemv_read_ns   += read_ns
          end

          def self.bump_gemm(wait_ns : Int64)
            return unless @@enabled
            @@gemm_count += 1
            @@gemm_wait_ns += wait_ns
          end

          def self.bump_dn(encode_ns : Int64, wait_ns : Int64, read_ns : Int64)
            return unless @@enabled
            @@dn_count += 1
            @@dn_encode_ns += encode_ns
            @@dn_wait_ns   += wait_ns
            @@dn_read_ns   += read_ns
          end

          def self.bump_attn(encode_ns : Int64, wait_ns : Int64, read_ns : Int64)
            return unless @@enabled
            @@attn_count += 1
            @@attn_encode_ns += encode_ns
            @@attn_wait_ns   += wait_ns
            @@attn_read_ns   += read_ns
          end

          def self.bump_wave(encode_ns : Int64, wait_ns : Int64, read_ns : Int64)
            return unless @@enabled
            @@wave_count += 1
            @@wave_encode_ns += encode_ns
            @@wave_wait_ns   += wait_ns
            @@wave_read_ns   += read_ns
          end

          def self.bump_group(label : String, encode_ns : Int64, wait_ns : Int64, read_ns : Int64) : Nil
            return unless @@enabled
            @@group_counts[label] += 1
            @@group_encode_ns[label] += encode_ns
            @@group_wait_ns[label] += wait_ns
            @@group_read_ns[label] += read_ns
          end

          def self.bump_cpu_fallback : Nil
            return unless @@enabled
            @@cpu_fallback += 1
          end

          def self.trace(name : String)
            unless @@enabled
              yield
              return
            end

            t0 = Time.instant
            @@scope_stack << name
            begin
              yield
            ensure
              @@scope_stack.pop?
              @@trace_counts[name] += 1
              @@trace_ns[name] += (Time.instant - t0).total_nanoseconds.to_i64
            end
          end

          def self.bump_matmul_shape(name : String, weight_bytes : Int64) : Nil
            return unless @@enabled
            scoped_name = if scope = @@scope_stack.last?
                            "#{scope} #{name}"
                          else
                            name
                          end
            @@matmul_counts[scoped_name] += 1
            @@matmul_weight_bytes[scoped_name] += weight_bytes
          end

          def self.bump_conversion(name : String, traffic_bytes : Int64) : Nil
            return unless @@enabled
            scoped_name = if scope = @@scope_stack.last?
                            "#{scope} #{name}"
                          else
                            name
                          end
            @@conversion_counts[scoped_name] += 1
            @@conversion_bytes[scoped_name] += traffic_bytes
          end

          def self.report_io : String
            String.build do |s|
              total_syncs = @@gemv_count + @@gemm_count + @@dn_count + @@attn_count + @@wave_count
              s << "── Qwen35Metal.Profile report ──\n"
              s << sprintf("  gemv:  %d calls  encode %.2f ms  wait %.2f ms  read %.2f ms\n",
                           @@gemv_count, @@gemv_encode_ns / 1_000_000.0,
                           @@gemv_wait_ns / 1_000_000.0, @@gemv_read_ns / 1_000_000.0)
              s << sprintf("  gemm:  %d calls  wait %.2f ms\n",
                           @@gemm_count, @@gemm_wait_ns / 1_000_000.0)
              s << sprintf("  dn:    %d calls  encode %.2f ms  wait %.2f ms  read %.2f ms\n",
                           @@dn_count, @@dn_encode_ns / 1_000_000.0,
                           @@dn_wait_ns / 1_000_000.0, @@dn_read_ns / 1_000_000.0)
              s << sprintf("  attn:  %d calls  encode %.2f ms  wait %.2f ms  read %.2f ms\n",
                           @@attn_count, @@attn_encode_ns / 1_000_000.0,
                           @@attn_wait_ns / 1_000_000.0, @@attn_read_ns / 1_000_000.0)
              s << sprintf("  wave:  %d calls  encode %.2f ms  wait %.2f ms  read %.2f ms\n",
                           @@wave_count, @@wave_encode_ns / 1_000_000.0,
                           @@wave_wait_ns / 1_000_000.0, @@wave_read_ns / 1_000_000.0)
              unless @@trace_counts.empty?
                s << "  wave encode trace:\n"
                @@trace_counts.keys.sort_by { |name| -@@trace_ns[name] }.each do |name|
                  s << sprintf("    %-18s %4d calls  %.2f ms\n",
                               name, @@trace_counts[name], @@trace_ns[name] / 1_000_000.0)
                end
              end
              unless @@group_counts.empty?
                s << "  grouped command buffers:\n"
                @@group_counts.keys.sort_by { |name| {-@@group_wait_ns[name], name} }.each do |name|
                  s << sprintf("    %-18s %4d calls  encode %.2f ms  wait %.2f ms  read %.2f ms\n",
                               name, @@group_counts[name],
                               @@group_encode_ns[name] / 1_000_000.0,
                               @@group_wait_ns[name] / 1_000_000.0,
                               @@group_read_ns[name] / 1_000_000.0)
                end
              end
              unless @@matmul_counts.empty?
                s << "  matmul shapes:\n"
                total_weight_bytes = @@matmul_weight_bytes.values.sum
                @@matmul_counts.keys.sort_by { |name| {-@@matmul_weight_bytes[name], name} }.each do |name|
                  pct = total_weight_bytes > 0 ? @@matmul_weight_bytes[name] * 100.0 / total_weight_bytes : 0.0
                  s << sprintf("    %-34s %4d calls  %.2f MiB logical weights  %.2f%%\n",
                               name, @@matmul_counts[name], @@matmul_weight_bytes[name] / 1_048_576.0, pct)
                end
                s << sprintf("    %-34s      total  %.2f MiB logical weights\n",
                             "matmul", total_weight_bytes / 1_048_576.0)
              end
              unless @@conversion_counts.empty?
                s << "  conversion kernels:\n"
                total_conversion_bytes = @@conversion_bytes.values.sum
                @@conversion_counts.keys.sort_by { |name| {-@@conversion_bytes[name], name} }.each do |name|
                  pct = total_conversion_bytes > 0 ? @@conversion_bytes[name] * 100.0 / total_conversion_bytes : 0.0
                  s << sprintf("    %-34s %4d calls  %.2f MiB logical traffic  %.2f%%\n",
                               name, @@conversion_counts[name], @@conversion_bytes[name] / 1_048_576.0, pct)
                end
                s << sprintf("    %-34s      total  %.2f MiB logical traffic\n",
                             "conversion", total_conversion_bytes / 1_048_576.0)
              end
              unless @@matmul_weight_bytes.empty? && @@conversion_bytes.empty?
                total_weight_bytes = @@matmul_weight_bytes.values.sum
                total_conversion_bytes = @@conversion_bytes.values.sum
                total_logical_bytes = total_weight_bytes + total_conversion_bytes
                if total_logical_bytes > 0
                  s << sprintf("  logical traffic mix: matmul %.2f%%  conversion %.2f%%\n",
                               total_weight_bytes * 100.0 / total_logical_bytes,
                               total_conversion_bytes * 100.0 / total_logical_bytes)
                end
              end
              s << sprintf("  cpu_fallback matvecs: %d\n", @@cpu_fallback)
              s << sprintf("  total metal syncs: %d\n", total_syncs)
            end
          end
        end

        # Per-slot tags for `matmul_many` output buffers. Multiple outputs
        # are bound into one compute encoder, so they must not alias.
        # Add more tags here if a batched dispatch ever grows past 8.
        MANY_SLOT_TAGS = [
          :mv_many_out_0,
          :mv_many_out_1,
          :mv_many_out_2,
          :mv_many_out_3,
          :mv_many_out_4,
          :mv_many_out_5,
          :mv_many_out_6,
          :mv_many_out_7,
        ]

        # ── Phase 4.2 scratch pool ─────────────────────────────────────
        # Persistent per-dispatch scratch buffers keyed by (tag, bytes).
        # Each dispatch-site tag names a buffer slot that must not alias
        # another buffer alive in the same command buffer. Sizes vary
        # across call sites and layers, so key includes size; reuse
        # within a key is a cache hit. Buffers live until `clear`.
        module Scratch
          @@pool : Hash({Symbol, Int64}, ML::MetalBuffer) = {} of {Symbol, Int64} => ML::MetalBuffer
          @@pool_s : Hash({String, Int64}, ML::MetalBuffer) = {} of {String, Int64} => ML::MetalBuffer
          @@hits   = 0_i64
          @@misses = 0_i64

          def self.get(tag : Symbol, byte_size : Int64) : ML::MetalBuffer
            # A/B gate — when set, always allocate fresh (emulates pre-4.2).
            if ENV["QWEN35_SCRATCH_OFF"]? == "1"
              @@misses += 1
              return ML::MetalBuffer.new(byte_size)
            end
            key = {tag, byte_size}
            if buf = @@pool[key]?
              @@hits += 1
              return buf
            end
            @@misses += 1
            buf = ML::MetalBuffer.new(byte_size)
            @@pool[key] = buf
            buf
          end

          def self.get(tag : String, byte_size : Int64) : ML::MetalBuffer
            if ENV["QWEN35_SCRATCH_OFF"]? == "1"
              @@misses += 1
              return ML::MetalBuffer.new(byte_size)
            end
            key = {tag, byte_size}
            if buf = @@pool_s[key]?
              @@hits += 1
              return buf
            end
            @@misses += 1
            buf = ML::MetalBuffer.new(byte_size)
            @@pool_s[key] = buf
            buf
          end

          def self.stats : {Int64, Int64}
            {@@hits, @@misses}
          end

          def self.clear : Nil
            @@pool.clear
            @@pool_s.clear
            @@hits = @@misses = 0_i64
          end
        end

        module ConstCache
          @@written : Hash(String, Bool) = {} of String => Bool

          def self.write_once(tag : String, buf : ML::MetalBuffer, data : Array(Float32)) : Nil
            key = "#{tag}:#{buf.handle.address}:#{buf.size}:#{data.to_unsafe.address}:#{data.size}"
            return if @@written[key]?
            buf.write(data)
            @@written[key] = true
          end

          def self.write_zero_f32_once(tag : String, buf : ML::MetalBuffer, count : Int32) : Nil
            key = "#{tag}:#{buf.handle.address}:#{buf.size}:zero:#{count}"
            return if @@written[key]?
            buf.contents.as(Pointer(UInt8)).clear(count.to_i64 * sizeof(Float32))
            @@written[key] = true
          end

          def self.clear : Nil
            @@written.clear
          end
        end

        # Whole-mmap MetalBuffer. Registered once per model load via
        # `register_mmap`. All weights whose `raw` bytes are slices
        # inside this region dispatch against it with a byte offset —
        # true zero-copy on Apple Silicon unified memory.
        @@mmap_base_addr : UInt64 = 0_u64
        @@mmap_size      : Int64  = 0_i64
        @@mmap_buf       : ML::MetalBuffer? = nil

        def self.available? : Bool
          ML::Metal::Device.init!
        end

        # Register the mmap'd weight file as a single zero-copy
        # MetalBuffer. Must be called before `matmul(qw, ...)` if you
        # want zero-copy dispatch. Idempotent on the same region;
        # subsequent calls with a different region replace the buffer
        # (previous one is released).
        def self.register_mmap(base : Pointer(UInt8), size : UInt64) : Nil
          return unless available?
          page = 16384_u64
          raise "mmap base #{base.address} not page-aligned (page=#{page})" unless base.address % page == 0
          # newBufferWithBytesNoCopy also requires the length to be a
          # multiple of page size. mmap'd files are page-rounded on Darwin.
          aligned_size = ((size + page - 1) // page) * page
          if aligned_size.to_i64 > size.to_i64
            # safer to pass a smaller, still page-aligned length that
            # lies entirely within the mmap region
            aligned_size = (size // page) * page
          end
          raise "mmap region too small (size=#{size})" if aligned_size == 0

          if buf = @@mmap_buf
            # Replace previous — release the ObjC wrapper (not the bytes).
            buf.release
            @@mmap_buf = nil
          end

          @@mmap_base_addr = base.address
          @@mmap_size = aligned_size.to_i64
          @@mmap_buf = ML::MetalBuffer.wrap_no_copy(
            base.as(Pointer(Void)),
            @@mmap_size,
          )
          ConstCache.clear
          nil
        end

        # Return (buffer, byte-offset) for the given raw slice if it lies
        # inside the registered mmap region. Otherwise nil — caller must
        # fall back to per-weight upload.
        private def self.mmap_slot_for(raw : Bytes) : {ML::MetalBuffer, Int64}?
          return nil if @@mmap_buf.nil?
          base = @@mmap_base_addr
          size = @@mmap_size
          addr = raw.to_unsafe.address
          return nil if addr < base
          off = (addr - base).to_i64
          return nil if off + raw.size > size
          {@@mmap_buf.not_nil!, off}
        end

        private def self.weight_slot(qw : QuantWeight) : {ML::MetalBuffer, Int64}
          if slot = mmap_slot_for(qw.raw)
            slot
          else
            {qw.fallback_metal_buffer, 0_i64}
          end
        end

        # Lazy compile and cache pipelines on first use.
        private def self.mv_pipeline : ML::Metal::ComputePipeline
          @@mv_pipeline ||= ML::Metal::PipelineCache.get("simd_mv_q4k_f32") {
            ML::Metal::ComputePipeline.new("simd_mv_q4k_f32", GEMM_Q4K_SOURCE)
          }
        end

        private def self.mv_add_pipeline : ML::Metal::ComputePipeline
          @@mv_add_pipeline ||= ML::Metal::PipelineCache.get("simd_mv_q4k_f32_add") {
            ML::Metal::ComputePipeline.new("simd_mv_q4k_f32_add", GEMM_Q4K_SOURCE)
          }
        end

        private def self.mm_pipeline : ML::Metal::ComputePipeline
          @@mm_pipeline ||= ML::Metal::PipelineCache.get("simd_mm_q4k_f32") {
            ML::Metal::ComputePipeline.new("simd_mm_q4k_f32", GEMM_Q4K_SOURCE)
          }
        end

        private def self.mm_h16_pipeline : ML::Metal::ComputePipeline
          @@mm_h16_pipeline ||= ML::Metal::PipelineCache.get("simd_mm_q4k_h16") {
            ML::Metal::ComputePipeline.new("simd_mm_q4k_h16", GEMM_Q4K_SOURCE)
          }
        end

        private def self.mm5_pipeline : ML::Metal::ComputePipeline
          @@mm5_pipeline ||= ML::Metal::PipelineCache.get("qwen35_simd_mm_q5k") {
            ML::Metal::ComputePipeline.new("simd_mm_q5k", GEMM_MM_SOURCE)
          }
        end

        private def self.mm6_pipeline : ML::Metal::ComputePipeline
          @@mm6_pipeline ||= ML::Metal::PipelineCache.get("qwen35_simd_mm_q6k") {
            ML::Metal::ComputePipeline.new("simd_mm_q6k", GEMM_MM_SOURCE)
          }
        end

        private def self.mm5_f32out_pipeline : ML::Metal::ComputePipeline
          @@mm5_f32out_pipeline ||= ML::Metal::PipelineCache.get("qwen35_simd_mm_q5k_f32out") {
            ML::Metal::ComputePipeline.new("simd_mm_q5k_f32out", GEMM_MM_SOURCE)
          }
        end

        private def self.mm6_f32out_pipeline : ML::Metal::ComputePipeline
          @@mm6_f32out_pipeline ||= ML::Metal::PipelineCache.get("qwen35_simd_mm_q6k_f32out") {
            ML::Metal::ComputePipeline.new("simd_mm_q6k_f32out", GEMM_MM_SOURCE)
          }
        end

        private def self.mm6_f32out_add_pipeline : ML::Metal::ComputePipeline
          @@mm6_f32out_add_pipeline ||= ML::Metal::PipelineCache.get("qwen35_simd_mm_q6k_f32out_add") {
            ML::Metal::ComputePipeline.new("simd_mm_q6k_f32out_add", GEMM_MM_SOURCE)
          }
        end

        private def self.mv5_pipeline : ML::Metal::ComputePipeline
          @@mv5_pipeline ||= ML::Metal::PipelineCache.get("simd_mv_q5k_f32") {
            ML::Metal::ComputePipeline.new("simd_mv_q5k_f32", GEMM_Q56K_SOURCE)
          }
        end

        private def self.mv6_pipeline : ML::Metal::ComputePipeline
          @@mv6_pipeline ||= ML::Metal::PipelineCache.get("simd_mv_q6k_f32") {
            ML::Metal::ComputePipeline.new("simd_mv_q6k_f32", GEMM_Q56K_SOURCE)
          }
        end

        private def self.mv6_add_pipeline : ML::Metal::ComputePipeline
          @@mv6_add_pipeline ||= ML::Metal::PipelineCache.get("simd_mv_q6k_f32_add") {
            ML::Metal::ComputePipeline.new("simd_mv_q6k_f32_add", GEMM_Q56K_SOURCE)
          }
        end

        private def self.mv8_pipeline : ML::Metal::ComputePipeline
          @@mv8_pipeline ||= ML::Metal::PipelineCache.get("simd_mv_q8_0_f32") {
            ML::Metal::ComputePipeline.new("simd_mv_q8_0_f32", GEMM_Q56K_SOURCE)
          }
        end

        private def self.mv8_add_pipeline : ML::Metal::ComputePipeline
          @@mv8_add_pipeline ||= ML::Metal::PipelineCache.get("simd_mv_q8_0_f32_add") {
            ML::Metal::ComputePipeline.new("simd_mv_q8_0_f32_add", GEMM_Q56K_SOURCE)
          }
        end

        private def self.mv8_dual_pipeline : ML::Metal::ComputePipeline
          @@mv8_dual_pipeline ||= ML::Metal::PipelineCache.get("simd_mv_q8_0_dual_f32") {
            ML::Metal::ComputePipeline.new("simd_mv_q8_0_dual_f32", GEMM_Q56K_SOURCE)
          }
        end

        private def self.mv6_top1_tiles_pipeline : ML::Metal::ComputePipeline
          @@mv6_top1_tiles_pipeline ||= ML::Metal::PipelineCache.get("simd_mv_q6k_top1_tiles_f32") {
            ML::Metal::ComputePipeline.new("simd_mv_q6k_top1_tiles_f32", GEMM_Q56K_SOURCE)
          }
        end

        private def self.mv6_top1_tiles_batch_pipeline : ML::Metal::ComputePipeline
          @@mv6_top1_tiles_batch_pipeline ||= ML::Metal::PipelineCache.get("simd_mv_q6k_top1_tiles_batch_f32") {
            ML::Metal::ComputePipeline.new("simd_mv_q6k_top1_tiles_batch_f32", GEMM_Q56K_SOURCE)
          }
        end

        private def self.mv8_top1_tiles_pipeline : ML::Metal::ComputePipeline
          @@mv8_top1_tiles_pipeline ||= ML::Metal::PipelineCache.get("simd_mv_q8_0_top1_tiles_f32") {
            ML::Metal::ComputePipeline.new("simd_mv_q8_0_top1_tiles_f32", GEMM_Q56K_SOURCE)
          }
        end

        private def self.top1_reduce_tiles_pipeline : ML::Metal::ComputePipeline
          @@top1_reduce_tiles_pipeline ||= ML::Metal::PipelineCache.get("qwen35_top1_reduce_tiles") {
            ML::Metal::ComputePipeline.new("qwen35_top1_reduce_tiles", GEMM_Q56K_SOURCE)
          }
        end

        private def self.top1_reduce_tiles_batch_pipeline : ML::Metal::ComputePipeline
          @@top1_reduce_tiles_batch_pipeline ||= ML::Metal::PipelineCache.get("qwen35_top1_reduce_tiles_batch") {
            ML::Metal::ComputePipeline.new("qwen35_top1_reduce_tiles_batch", GEMM_Q56K_SOURCE)
          }
        end

        private def self.top1_reduce_f16_rows_pipeline : ML::Metal::ComputePipeline
          @@top1_reduce_f16_rows_pipeline ||= ML::Metal::PipelineCache.get("qwen35_top1_reduce_f16_rows") {
            ML::Metal::ComputePipeline.new("qwen35_top1_reduce_f16_rows", GEMM_Q56K_SOURCE)
          }
        end

        private def self.top2_reduce_f16_rows_pipeline : ML::Metal::ComputePipeline
          @@top2_reduce_f16_rows_pipeline ||= ML::Metal::PipelineCache.get("qwen35_top2_reduce_f16_rows") {
            ML::Metal::ComputePipeline.new("qwen35_top2_reduce_f16_rows", GEMM_Q56K_SOURCE)
          }
        end

        private def self.dn_pipeline : ML::Metal::ComputePipeline
          @@dn_pipeline ||= ML::Metal::PipelineCache.get("delta_net_step") {
            ML::Metal::ComputePipeline.new("delta_net_step", DELTA_NET_SOURCE)
          }
        end

        private def self.dn128_pipeline : ML::Metal::ComputePipeline
          @@dn128_pipeline ||= ML::Metal::PipelineCache.get("delta_net_step_128") {
            ML::Metal::ComputePipeline.new("delta_net_step_128", DELTA_NET_SOURCE)
          }
        end

        private def self.dn128_fused_pipeline : ML::Metal::ComputePipeline
          @@dn128_fused_pipeline ||= ML::Metal::PipelineCache.get("delta_net_step_128_fused") {
            ML::Metal::ComputePipeline.new("delta_net_step_128_fused", DELTA_NET_SOURCE)
          }
        end

        private def self.dn128_fused_post_pipeline : ML::Metal::ComputePipeline
          @@dn128_fused_post_pipeline ||= ML::Metal::PipelineCache.get("delta_net_step_128_fused_post") {
            ML::Metal::ComputePipeline.new("delta_net_step_128_fused_post", DELTA_NET_SOURCE)
          }
        end

        private def self.dn128_chunk_fused_pipeline : ML::Metal::ComputePipeline
          @@dn128_chunk_fused_pipeline ||= ML::Metal::PipelineCache.get("delta_net_chunk_128_fused") {
            ML::Metal::ComputePipeline.new("delta_net_chunk_128_fused", DELTA_NET_SOURCE)
          }
        end

        private def self.dn128_chunk_rowwise_pipeline : ML::Metal::ComputePipeline
          @@dn128_chunk_rowwise_pipeline ||= ML::Metal::PipelineCache.get("delta_net_chunk_128_rowwise") {
            ML::Metal::ComputePipeline.new("delta_net_chunk_128_rowwise", DELTA_NET_SOURCE)
          }
        end

        private def self.dn_128_enabled? : Bool
          ENV["QWEN35_DN_128"]? != "0"
        end

        private def self.dn_chunk_rowwise_enabled?(s : Int32) : Bool
          s == 128 && ENV["QWEN35_DN_CHUNK_ROWWISE_OFF"]? != "1"
        end

        private def self.dn_fused_enabled? : Bool
          dn_128_enabled? && ENV["QWEN35_DN_FUSED"]? != "0"
        end

        private def self.dn_post_fused_enabled? : Bool
          dn_fused_enabled? && ENV["QWEN35_DN_POST_FUSED"]? != "0"
        end

        private def self.active_dn_pipeline : ML::Metal::ComputePipeline
          if dn_fused_enabled?
            dn128_fused_pipeline
          elsif dn_128_enabled?
            dn128_pipeline
          else
            dn_pipeline
          end
        end

        private def self.dn_threadgroup_size : Int32
          dn_fused_enabled? || dn_128_enabled? ? 128 : 32
        end


        private def self.dn_post_pipeline : ML::Metal::ComputePipeline
          @@dn_post_pipeline ||= ML::Metal::PipelineCache.get("delta_net_post_norm_gate") {
            ML::Metal::ComputePipeline.new("delta_net_post_norm_gate", DELTA_NET_SOURCE)
          }
        end

        private def self.dn_post_chunk_pipeline : ML::Metal::ComputePipeline
          @@dn_post_chunk_pipeline ||= ML::Metal::PipelineCache.get("delta_net_post_norm_gate_chunk") {
            ML::Metal::ComputePipeline.new("delta_net_post_norm_gate_chunk", DELTA_NET_SOURCE)
          }
        end

        private def self.attn_pipeline : ML::Metal::ComputePipeline
          @@attn_pipeline ||= ML::Metal::PipelineCache.get("qwen35_attn_decode") {
            ML::Metal::ComputePipeline.new("qwen35_attn_decode", ATTN_DECODE_SOURCE)
          }
        end

        private def self.f32_to_f16_pipeline : ML::Metal::ComputePipeline
          @@f32_to_f16_pipeline ||= ML::Metal::PipelineCache.get("qwen35_f32_to_f16") {
            ML::Metal::ComputePipeline.new("qwen35_f32_to_f16", FFN_SOURCE)
          }
        end

        private def self.f16_to_f32_pipeline : ML::Metal::ComputePipeline
          @@f16_to_f32_pipeline ||= ML::Metal::PipelineCache.get("qwen35_f16_to_f32") {
            ML::Metal::ComputePipeline.new("qwen35_f16_to_f32", FFN_SOURCE)
          }
        end

        private def self.ffn_swiglu_pipeline : ML::Metal::ComputePipeline
          @@ffn_swiglu_pipeline ||= ML::Metal::PipelineCache.get("qwen35_swiglu_mul") {
            ML::Metal::ComputePipeline.new("qwen35_swiglu_mul", FFN_SOURCE)
          }
        end

        private def self.add_rmsnorm_pipeline : ML::Metal::ComputePipeline
          @@add_rmsnorm_pipeline ||= ML::Metal::PipelineCache.get("qwen35_add_rmsnorm") {
            ML::Metal::ComputePipeline.new("qwen35_add_rmsnorm", FFN_SOURCE)
          }
        end

        private def self.add_rmsnorm_rows_pipeline : ML::Metal::ComputePipeline
          @@add_rmsnorm_rows_pipeline ||= ML::Metal::PipelineCache.get("qwen35_add_rmsnorm_rows") {
            ML::Metal::ComputePipeline.new("qwen35_add_rmsnorm_rows", FFN_SOURCE)
          }
        end

        private def self.add_vec_pipeline : ML::Metal::ComputePipeline
          @@add_vec_pipeline ||= ML::Metal::PipelineCache.get("qwen35_add_vec") {
            ML::Metal::ComputePipeline.new("qwen35_add_vec", FFN_SOURCE)
          }
        end

        private def self.rmsnorm_vec_pipeline : ML::Metal::ComputePipeline
          @@rmsnorm_vec_pipeline ||= ML::Metal::PipelineCache.get("qwen35_rmsnorm_vec") {
            ML::Metal::ComputePipeline.new("qwen35_rmsnorm_vec", FFN_SOURCE)
          }
        end

        private def self.rmsnorm_rows_pipeline : ML::Metal::ComputePipeline
          @@rmsnorm_rows_pipeline ||= ML::Metal::PipelineCache.get("qwen35_rmsnorm_rows") {
            ML::Metal::ComputePipeline.new("qwen35_rmsnorm_rows", FFN_SOURCE)
          }
        end

        private def self.recurrent_ab_pipeline : ML::Metal::ComputePipeline
          @@recurrent_ab_pipeline ||= ML::Metal::PipelineCache.get("qwen35_recurrent_ab") {
            ML::Metal::ComputePipeline.new("qwen35_recurrent_ab", RECURRENT_SOURCE)
          }
        end

        private def self.recurrent_ab_chunk_pipeline : ML::Metal::ComputePipeline
          @@recurrent_ab_chunk_pipeline ||= ML::Metal::PipelineCache.get("qwen35_recurrent_ab_chunk") {
            ML::Metal::ComputePipeline.new("qwen35_recurrent_ab_chunk", RECURRENT_SOURCE)
          }
        end

        private def self.recurrent_conv_pipeline : ML::Metal::ComputePipeline
          @@recurrent_conv_pipeline ||= ML::Metal::PipelineCache.get("qwen35_recurrent_conv") {
            ML::Metal::ComputePipeline.new("qwen35_recurrent_conv", RECURRENT_SOURCE)
          }
        end

        private def self.recurrent_shift_pipeline : ML::Metal::ComputePipeline
          @@recurrent_shift_pipeline ||= ML::Metal::PipelineCache.get("qwen35_recurrent_shift") {
            ML::Metal::ComputePipeline.new("qwen35_recurrent_shift", RECURRENT_SOURCE)
          }
        end

        private def self.recurrent_conv_shift_pipeline : ML::Metal::ComputePipeline
          @@recurrent_conv_shift_pipeline ||= ML::Metal::PipelineCache.get("qwen35_recurrent_conv_shift") {
            ML::Metal::ComputePipeline.new("qwen35_recurrent_conv_shift", RECURRENT_SOURCE)
          }
        end

        private def self.recurrent_conv_shift_chunk_pipeline : ML::Metal::ComputePipeline
          @@recurrent_conv_shift_chunk_pipeline ||= ML::Metal::PipelineCache.get("qwen35_recurrent_conv_shift_chunk") {
            ML::Metal::ComputePipeline.new("qwen35_recurrent_conv_shift_chunk", RECURRENT_SOURCE)
          }
        end

        private def self.recurrent_conv_shift_chunk_h16_pipeline : ML::Metal::ComputePipeline
          @@recurrent_conv_shift_chunk_h16_pipeline ||= ML::Metal::PipelineCache.get("qwen35_recurrent_conv_shift_chunk_h16") {
            ML::Metal::ComputePipeline.new("qwen35_recurrent_conv_shift_chunk_h16", RECURRENT_SOURCE)
          }
        end

        private def self.l2_heads_pipeline : ML::Metal::ComputePipeline
          @@l2_heads_pipeline ||= ML::Metal::PipelineCache.get("qwen35_l2_heads") {
            ML::Metal::ComputePipeline.new("qwen35_l2_heads", RECURRENT_SOURCE)
          }
        end

        private def self.l2_heads_chunk_pipeline : ML::Metal::ComputePipeline
          @@l2_heads_chunk_pipeline ||= ML::Metal::PipelineCache.get("qwen35_l2_heads_chunk") {
            ML::Metal::ComputePipeline.new("qwen35_l2_heads_chunk", RECURRENT_SOURCE)
          }
        end

        private def self.split_qgate_pipeline : ML::Metal::ComputePipeline
          @@split_qgate_pipeline ||= ML::Metal::PipelineCache.get("qwen35_split_qgate") {
            ML::Metal::ComputePipeline.new("qwen35_split_qgate", FULLATTN_SOURCE)
          }
        end

        private def self.split_qgate_rows_pipeline : ML::Metal::ComputePipeline
          @@split_qgate_rows_pipeline ||= ML::Metal::PipelineCache.get("qwen35_split_qgate_rows") {
            ML::Metal::ComputePipeline.new("qwen35_split_qgate_rows", FULLATTN_SOURCE)
          }
        end

        private def self.rmsnorm_heads_pipeline : ML::Metal::ComputePipeline
          @@rmsnorm_heads_pipeline ||= ML::Metal::PipelineCache.get("qwen35_rmsnorm_heads") {
            ML::Metal::ComputePipeline.new("qwen35_rmsnorm_heads", FULLATTN_SOURCE)
          }
        end

        private def self.rmsnorm_heads_rows_pipeline : ML::Metal::ComputePipeline
          @@rmsnorm_heads_rows_pipeline ||= ML::Metal::PipelineCache.get("qwen35_rmsnorm_heads_rows") {
            ML::Metal::ComputePipeline.new("qwen35_rmsnorm_heads_rows", FULLATTN_SOURCE)
          }
        end

        private def self.rope_partial_pipeline : ML::Metal::ComputePipeline
          @@rope_partial_pipeline ||= ML::Metal::PipelineCache.get("qwen35_rope_partial") {
            ML::Metal::ComputePipeline.new("qwen35_rope_partial", FULLATTN_SOURCE)
          }
        end

        private def self.rope_partial_rows_pipeline : ML::Metal::ComputePipeline
          @@rope_partial_rows_pipeline ||= ML::Metal::PipelineCache.get("qwen35_rope_partial_rows") {
            ML::Metal::ComputePipeline.new("qwen35_rope_partial_rows", FULLATTN_SOURCE)
          }
        end

        private def self.kv_write_pipeline : ML::Metal::ComputePipeline
          @@kv_write_pipeline ||= ML::Metal::PipelineCache.get("qwen35_kv_write") {
            ML::Metal::ComputePipeline.new("qwen35_kv_write", FULLATTN_SOURCE)
          }
        end

        private def self.kv_write_rows_pipeline : ML::Metal::ComputePipeline
          @@kv_write_rows_pipeline ||= ML::Metal::PipelineCache.get("qwen35_kv_write_rows") {
            ML::Metal::ComputePipeline.new("qwen35_kv_write_rows", FULLATTN_SOURCE)
          }
        end

        private def self.attn_rows_pipeline : ML::Metal::ComputePipeline
          @@attn_rows_pipeline ||= ML::Metal::PipelineCache.get("qwen35_attn_decode_rows") {
            ML::Metal::ComputePipeline.new("qwen35_attn_decode_rows", FULLATTN_SOURCE)
          }
        end

        private def self.gemv_pipeline_for(qw : QuantWeight) : ML::Metal::ComputePipeline?
          case qw.type
          when .q4_k? then mv_pipeline
          when .q5_k? then mv5_pipeline
          when .q6_k? then mv6_pipeline
          when .q8_0? then mv8_pipeline
          else             nil
          end
        end

        private def self.gemv_add_pipeline_for(qw : QuantWeight) : ML::Metal::ComputePipeline?
          case qw.type
          when .q4_k? then mv_add_pipeline
          when .q6_k? then mv6_add_pipeline
          when .q8_0? then mv8_add_pipeline
          else             nil
          end
        end

        private def self.gemv_rows_per_tg_for(pipeline : ML::Metal::ComputePipeline) : Int32
          case pipeline
          when .same?(mv5_pipeline)
            MV_Q5_NSG * MV_Q5_NR0
          when .same?(mv6_pipeline), .same?(mv6_add_pipeline)
            MV_Q6_NSG * MV_Q6_NR0
          when .same?(mv8_pipeline), .same?(mv8_add_pipeline)
            MV_Q8_NSG * MV_Q8_NR0
          else
            MV_Q4_NSG * MV_Q4_NR0
          end
        end

        private def self.gemv_threads_per_tg_for(pipeline : ML::Metal::ComputePipeline) : Int32
          case pipeline
          when .same?(mv8_pipeline), .same?(mv8_add_pipeline), .same?(mv8_top1_tiles_pipeline)
            MV_Q8_NSG * 32
          else
            64
          end
        end

        private def self.gemv_profile_quant(pipeline : ML::Metal::ComputePipeline) : {String, Int32, Int32}
          case pipeline
          when .same?(mv5_pipeline)
            {"Q5_K", Q5K_BLOCK_BYTES, QK_K}
          when .same?(mv6_pipeline), .same?(mv6_add_pipeline), .same?(mv6_top1_tiles_pipeline)
            {"Q6_K", Q6K_BLOCK_BYTES, QK_K}
          when .same?(mv8_pipeline), .same?(mv8_add_pipeline), .same?(mv8_top1_tiles_pipeline)
            {"Q8_0", Q8_0_BLOCK_BYTES, Q8_0_QK}
          else
            {"Q4_K", Q4K_BLOCK_BYTES, QK_K}
          end
        end

        private def self.encode_gemv(enc : ML::Metal::ComputeEncoder,
                                     pipeline : ML::Metal::ComputePipeline,
                                     x_buf : ML::MetalBuffer,
                                     out_buf : ML::MetalBuffer,
                                     w_buf : ML::MetalBuffer,
                                     w_offset : Int64,
                                     in_dim : Int32,
                                     out_dim : Int32,
                                     batch : Int32 = 1,
                                     profile_shape : Bool = true) : Nil
          if profile_shape
            quant_name, block_bytes, block_elems = gemv_profile_quant(pipeline)
            blocks_per_row = (in_dim + block_elems - 1) // block_elems
            weight_bytes = out_dim.to_i64 * blocks_per_row.to_i64 * block_bytes.to_i64
            Profile.bump_matmul_shape("gemv #{quant_name} #{in_dim}x#{out_dim} b#{batch}", weight_bytes)
          end
          enc.set_pipeline(pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
          enc.set_buffer(x_buf, 1)
          enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32,  3)
          enc.set_value(out_dim.to_u32, 4)
          enc.set_value(batch.to_u32,   5)
          rows_per_tg = gemv_rows_per_tg_for(pipeline)
          grid = {(out_dim + rows_per_tg - 1) // rows_per_tg, batch, 1}
          enc.dispatch_threadgroups(grid, {gemv_threads_per_tg_for(pipeline), 1, 1})
        end

        private def self.encode_gemv_add(enc : ML::Metal::ComputeEncoder,
                                         pipeline : ML::Metal::ComputePipeline,
                                         x_buf : ML::MetalBuffer,
                                         residual_buf : ML::MetalBuffer,
                                         out_buf : ML::MetalBuffer,
                                         w_buf : ML::MetalBuffer,
                                         w_offset : Int64,
                                         in_dim : Int32,
                                         out_dim : Int32,
                                         batch : Int32 = 1,
                                         profile_shape : Bool = true) : Nil
          if profile_shape
            quant_name, block_bytes, block_elems = gemv_profile_quant(pipeline)
            blocks_per_row = (in_dim + block_elems - 1) // block_elems
            weight_bytes = out_dim.to_i64 * blocks_per_row.to_i64 * block_bytes.to_i64
            Profile.bump_matmul_shape("gemv_add #{quant_name} #{in_dim}x#{out_dim} b#{batch}", weight_bytes)
          end
          enc.set_pipeline(pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
          enc.set_buffer(x_buf, 1)
          enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32, 3)
          enc.set_value(out_dim.to_u32, 4)
          enc.set_value(batch.to_u32, 5)
          enc.set_buffer(residual_buf, 6)
          rows_per_tg = gemv_rows_per_tg_for(pipeline)
          grid = {(out_dim + rows_per_tg - 1) // rows_per_tg, batch, 1}
          enc.dispatch_threadgroups(grid, {gemv_threads_per_tg_for(pipeline), 1, 1})
        end

        private def self.encode_gemv_q8_dual(enc : ML::Metal::ComputeEncoder,
                                             x_buf : ML::MetalBuffer,
                                             gate_out_buf : ML::MetalBuffer,
                                             up_out_buf : ML::MetalBuffer,
                                             gate_w_buf : ML::MetalBuffer,
                                             gate_w_offset : Int64,
                                             up_w_buf : ML::MetalBuffer,
                                             up_w_offset : Int64,
                                             in_dim : Int32,
                                             out_dim : Int32,
                                             batch : Int32 = 1,
                                             profile_shape : Bool = true) : Nil
          if profile_shape
            blocks_per_row = (in_dim + Q8_0_QK - 1) // Q8_0_QK
            weight_bytes = 2_i64 * out_dim.to_i64 * blocks_per_row.to_i64 * Q8_0_BLOCK_BYTES.to_i64
            Profile.bump_matmul_shape("gemv Q8_0 dual #{in_dim}x#{out_dim} b#{batch}", weight_bytes)
          end
          enc.set_pipeline(mv8_dual_pipeline)
          enc.set_buffer(gate_w_buf, 0, ML::Metal::BufferAccess::Read, offset: gate_w_offset)
          enc.set_buffer(up_w_buf, 1, ML::Metal::BufferAccess::Read, offset: up_w_offset)
          enc.set_buffer(x_buf, 2)
          enc.set_buffer(gate_out_buf, 3, ML::Metal::BufferAccess::Write)
          enc.set_buffer(up_out_buf, 4, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32,  5)
          enc.set_value(out_dim.to_u32, 6)
          enc.set_value(batch.to_u32,   7)
          rows_per_tg = MV_Q8_NSG * MV_Q8_NR0
          grid = {(out_dim + rows_per_tg - 1) // rows_per_tg, batch, 1}
          enc.dispatch_threadgroups(grid, {MV_Q8_NSG * 32, 1, 1})
        end

        private def self.encode_gemv_input_offset(enc : ML::Metal::ComputeEncoder,
                                                  pipeline : ML::Metal::ComputePipeline,
                                                  x_buf : ML::MetalBuffer,
                                                  x_offset : Int64,
                                                  out_buf : ML::MetalBuffer,
                                                  w_buf : ML::MetalBuffer,
                                                  w_offset : Int64,
                                                  in_dim : Int32,
                                                  out_dim : Int32) : Nil
          enc.set_pipeline(pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
          enc.set_buffer(x_buf, 1, ML::Metal::BufferAccess::Read, offset: x_offset)
          enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32,  3)
          enc.set_value(out_dim.to_u32, 4)
          enc.set_value(1_u32,          5)
          rows_per_tg = gemv_rows_per_tg_for(pipeline)
          enc.dispatch_threadgroups({(out_dim + rows_per_tg - 1) // rows_per_tg, 1, 1}, {gemv_threads_per_tg_for(pipeline), 1, 1})
        end

        private def self.encode_q4k_gemm(enc : ML::Metal::ComputeEncoder,
                                         x_buf : ML::MetalBuffer,
                                         out_buf : ML::MetalBuffer,
                                         w_buf : ML::MetalBuffer,
                                         w_offset : Int64,
                                         in_dim : Int32,
                                         out_dim : Int32,
                                         batch : Int32) : Nil
          enc.set_pipeline(mm_pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
          enc.set_buffer(x_buf, 1)
          enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32,  3)
          enc.set_value(out_dim.to_u32, 4)
          enc.set_value(batch.to_u32,   5)
          enc.set_threadgroup_memory(MM_SHMEM, 0)
          grid = {
            (batch   + MM_NR1 - 1) // MM_NR1,
            (out_dim + MM_NR0 - 1) // MM_NR0,
            1,
          }
          enc.dispatch_threadgroups(grid, {MM_TG, 1, 1})
        end

        private def self.encode_q4k_gemm_h16(enc : ML::Metal::ComputeEncoder,
                                             x_buf : ML::MetalBuffer,
                                             out_buf : ML::MetalBuffer,
                                             w_buf : ML::MetalBuffer,
                                             w_offset : Int64,
                                             in_dim : Int32,
                                             out_dim : Int32,
                                             batch : Int32) : Nil
          x16_buf = Scratch.get(:mm4_x16, (batch * in_dim).to_i64 * 2_i64)

          Profile.bump_conversion("f32_to_f16 q4_gemm_input #{in_dim} b#{batch}", (batch * in_dim).to_i64 * 6_i64)
          enc.set_pipeline(f32_to_f16_pipeline)
          enc.set_buffer(x_buf, 0)
          enc.set_buffer(x16_buf, 1, ML::Metal::BufferAccess::Write)
          enc.set_value((batch * in_dim).to_u32, 2)
          enc.dispatch_1d(batch * in_dim, 256)

          enc.set_pipeline(mm_h16_pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
          enc.set_buffer(x16_buf, 1)
          enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32, 3)
          enc.set_value(out_dim.to_u32, 4)
          enc.set_value(batch.to_u32, 5)
          enc.set_threadgroup_memory(MM_SHMEM, 0)
          grid = {
            (batch   + MM_NR1 - 1) // MM_NR1,
            (out_dim + MM_NR0 - 1) // MM_NR0,
            1,
          }
          enc.dispatch_threadgroups(grid, {MM_TG, 1, 1})
        end

        private def self.encode_q4k_gemm_h16_from_h16(enc : ML::Metal::ComputeEncoder,
                                                      x16_buf : ML::MetalBuffer,
                                                      out_buf : ML::MetalBuffer,
                                                      w_buf : ML::MetalBuffer,
                                                      w_offset : Int64,
                                                      in_dim : Int32,
                                                      out_dim : Int32,
                                                      batch : Int32) : Nil
          enc.set_pipeline(mm_h16_pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
          enc.set_buffer(x16_buf, 1)
          enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32, 3)
          enc.set_value(out_dim.to_u32, 4)
          enc.set_value(batch.to_u32, 5)
          enc.set_threadgroup_memory(MM_SHMEM, 0)
          grid = {
            (batch   + MM_NR1 - 1) // MM_NR1,
            (out_dim + MM_NR0 - 1) // MM_NR0,
            1,
          }
          enc.dispatch_threadgroups(grid, {MM_TG, 1, 1})
        end

        private def self.encode_q4k_gemm_h16_pair(enc : ML::Metal::ComputeEncoder,
                                                  x_buf : ML::MetalBuffer,
                                                  out_a_buf : ML::MetalBuffer,
                                                  out_b_buf : ML::MetalBuffer,
                                                  w_a_buf : ML::MetalBuffer,
                                                  w_a_offset : Int64,
                                                  w_b_buf : ML::MetalBuffer,
                                                  w_b_offset : Int64,
                                                  in_dim : Int32,
                                                  out_dim : Int32,
                                                  batch : Int32) : Nil
          x16_buf = Scratch.get(:mm4_pair_x16, (batch * in_dim).to_i64 * 2_i64)

          Profile.bump_conversion("f32_to_f16 q4_pair_input #{in_dim} b#{batch}", (batch * in_dim).to_i64 * 6_i64)
          enc.set_pipeline(f32_to_f16_pipeline)
          enc.set_buffer(x_buf, 0)
          enc.set_buffer(x16_buf, 1, ML::Metal::BufferAccess::Write)
          enc.set_value((batch * in_dim).to_u32, 2)
          enc.dispatch_1d(batch * in_dim, 256)

          encode_q4k_gemm_h16_from_h16(enc, x16_buf, out_a_buf, w_a_buf, w_a_offset, in_dim, out_dim, batch)
          encode_q4k_gemm_h16_from_h16(enc, x16_buf, out_b_buf, w_b_buf, w_b_offset, in_dim, out_dim, batch)
        end

        private def self.encode_q56k_gemm_f32(enc : ML::Metal::ComputeEncoder,
                                              pipeline : ML::Metal::ComputePipeline,
                                              x_buf : ML::MetalBuffer,
                                              out_buf : ML::MetalBuffer,
                                              w_buf : ML::MetalBuffer,
                                              w_offset : Int64,
                                              in_dim : Int32,
                                              out_dim : Int32,
                                              batch : Int32) : Nil
          x16_buf = Scratch.get(:mm56_x16, (batch * in_dim).to_i64 * 2_i64)

          Profile.bump_conversion("f32_to_f16 q56_gemm_input #{in_dim} b#{batch}", (batch * in_dim).to_i64 * 6_i64)
          enc.set_pipeline(f32_to_f16_pipeline)
          enc.set_buffer(x_buf, 0)
          enc.set_buffer(x16_buf, 1, ML::Metal::BufferAccess::Write)
          enc.set_value((batch * in_dim).to_u32, 2)
          enc.dispatch_1d(batch * in_dim, 256)

          enc.set_pipeline(pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
          enc.set_buffer(x16_buf, 1)
          enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32, 3)
          enc.set_value(out_dim.to_u32, 4)
          enc.set_value(batch.to_u32, 5)
          enc.set_threadgroup_memory(MM_SHMEM, 0)
          grid = {
            (batch   + MM_NR1 - 1) // MM_NR1,
            (out_dim + MM_NR0 - 1) // MM_NR0,
            1,
          }
          enc.dispatch_threadgroups(grid, {MM_TG, 1, 1})
        end

        private def self.encode_q56k_gemm_f32_add(enc : ML::Metal::ComputeEncoder,
                                                  pipeline : ML::Metal::ComputePipeline,
                                                  x_buf : ML::MetalBuffer,
                                                  residual_buf : ML::MetalBuffer,
                                                  out_buf : ML::MetalBuffer,
                                                  w_buf : ML::MetalBuffer,
                                                  w_offset : Int64,
                                                  in_dim : Int32,
                                                  out_dim : Int32,
                                                  batch : Int32) : Nil
          x16_buf = Scratch.get(:mm56_x16, (batch * in_dim).to_i64 * 2_i64)

          Profile.bump_conversion("f32_to_f16 q56_gemm_input #{in_dim} b#{batch}", (batch * in_dim).to_i64 * 6_i64)
          enc.set_pipeline(f32_to_f16_pipeline)
          enc.set_buffer(x_buf, 0)
          enc.set_buffer(x16_buf, 1, ML::Metal::BufferAccess::Write)
          enc.set_value((batch * in_dim).to_u32, 2)
          enc.dispatch_1d(batch * in_dim, 256)

          enc.set_pipeline(pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
          enc.set_buffer(x16_buf, 1)
          enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32, 3)
          enc.set_value(out_dim.to_u32, 4)
          enc.set_value(batch.to_u32, 5)
          enc.set_buffer(residual_buf, 6)
          enc.set_threadgroup_memory(MM_SHMEM, 0)
          grid = {
            (batch   + MM_NR1 - 1) // MM_NR1,
            (out_dim + MM_NR0 - 1) // MM_NR0,
            1,
          }
          enc.dispatch_threadgroups(grid, {MM_TG, 1, 1})
        end

        private def self.encode_q56k_gemm_h16(enc : ML::Metal::ComputeEncoder,
                                              pipeline : ML::Metal::ComputePipeline,
                                              x_buf : ML::MetalBuffer,
                                              out16_buf : ML::MetalBuffer,
                                              w_buf : ML::MetalBuffer,
                                              w_offset : Int64,
                                              in_dim : Int32,
                                              out_dim : Int32,
                                              batch : Int32) : Nil
          x16_buf = Scratch.get(:mm56_x16, (batch * in_dim).to_i64 * 2_i64)

          Profile.bump_conversion("f32_to_f16 q56_h16_input #{in_dim} b#{batch}", (batch * in_dim).to_i64 * 6_i64)
          enc.set_pipeline(f32_to_f16_pipeline)
          enc.set_buffer(x_buf, 0)
          enc.set_buffer(x16_buf, 1, ML::Metal::BufferAccess::Write)
          enc.set_value((batch * in_dim).to_u32, 2)
          enc.dispatch_1d(batch * in_dim, 256)

          encode_q56k_gemm_h16_from_h16(enc, pipeline, x16_buf, out16_buf, w_buf, w_offset, in_dim, out_dim, batch)
        end

        private def self.encode_q56k_gemm_h16_from_h16(enc : ML::Metal::ComputeEncoder,
                                                       pipeline : ML::Metal::ComputePipeline,
                                                       x16_buf : ML::MetalBuffer,
                                                       out16_buf : ML::MetalBuffer,
                                                       w_buf : ML::MetalBuffer,
                                                       w_offset : Int64,
                                                       in_dim : Int32,
                                                       out_dim : Int32,
                                                       batch : Int32) : Nil
          bias_buf = Scratch.get("mm56_bias_#{out_dim}", out_dim.to_i64 * sizeof(Float32))
          ConstCache.write_zero_f32_once("mm56_bias_#{out_dim}", bias_buf, out_dim)

          enc.set_pipeline(pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
          enc.set_buffer(x16_buf, 1)
          enc.set_buffer(bias_buf, 2)
          enc.set_buffer(out16_buf, 3, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32, 4)
          enc.set_value(out_dim.to_u32, 5)
          enc.set_value(batch.to_u32, 6)
          enc.set_value(0_u32, 7)
          enc.set_threadgroup_memory(MM_SHMEM, 0)
          grid = {
            (batch   + MM_NR1 - 1) // MM_NR1,
            (out_dim + MM_NR0 - 1) // MM_NR0,
            1,
          }
          enc.dispatch_threadgroups(grid, {MM_TG, 1, 1})
        end

        private def self.encode_matmul(enc : ML::Metal::ComputeEncoder,
                                       gemv_pipeline : ML::Metal::ComputePipeline,
                                       qw : QuantWeight,
                                       x_buf : ML::MetalBuffer,
                                       out_buf : ML::MetalBuffer,
                                       w_buf : ML::MetalBuffer,
                                       w_offset : Int64,
                                       in_dim : Int32,
                                       out_dim : Int32,
                                       batch : Int32) : Nil
          force_small_q4_gemv = small_q4_gemv_enabled? && qw.type.q4_k? && out_dim <= 64
          use_q4_h16 = q4_h16_gemm_enabled? && qw.type.q4_k? && batch > GEMM_BATCH_THRESHOLD && !force_small_q4_gemv
          route = if use_q4_h16
                    "q4_h16_gemm"
                  elsif qw.type.q4_k? && batch > GEMM_BATCH_THRESHOLD && !force_small_q4_gemv
                    "q4_gemm"
                  elsif q56_batch_gemm_enabled? && qw.type.q5_k? && batch > GEMM_BATCH_THRESHOLD
                    "q5_gemm"
                  elsif q56_batch_gemm_enabled? && qw.type.q6_k? && batch > GEMM_BATCH_THRESHOLD
                    "q6_gemm"
                  else
                    "gemv"
                  end
          Profile.bump_matmul_shape("#{route} #{qw.type.name} #{in_dim}x#{out_dim} b#{batch}", qw.raw.size.to_i64)

          if use_q4_h16
            encode_q4k_gemm_h16(enc, x_buf, out_buf, w_buf, w_offset, in_dim, out_dim, batch)
          elsif qw.type.q4_k? && batch > GEMM_BATCH_THRESHOLD && !force_small_q4_gemv
            encode_q4k_gemm(enc, x_buf, out_buf, w_buf, w_offset, in_dim, out_dim, batch)
          elsif q56_batch_gemm_enabled? && qw.type.q5_k? && batch > GEMM_BATCH_THRESHOLD
            encode_q56k_gemm_f32(enc, mm5_f32out_pipeline, x_buf, out_buf, w_buf, w_offset, in_dim, out_dim, batch)
          elsif q56_batch_gemm_enabled? && qw.type.q6_k? && batch > GEMM_BATCH_THRESHOLD
            encode_q56k_gemm_f32(enc, mm6_f32out_pipeline, x_buf, out_buf, w_buf, w_offset, in_dim, out_dim, batch)
          else
            encode_gemv(enc, gemv_pipeline, x_buf, out_buf, w_buf, w_offset, in_dim, out_dim, batch, profile_shape: false)
          end
        end

        private def self.encode_matmul_add(enc : ML::Metal::ComputeEncoder,
                                           gemv_pipeline : ML::Metal::ComputePipeline,
                                           qw : QuantWeight,
                                           x_buf : ML::MetalBuffer,
                                           residual_buf : ML::MetalBuffer,
                                           out_buf : ML::MetalBuffer,
                                           w_buf : ML::MetalBuffer,
                                           w_offset : Int64,
                                           in_dim : Int32,
                                           out_dim : Int32,
                                           batch : Int32) : Bool
          route = if q56_batch_gemm_enabled? && qw.type.q6_k? && batch > GEMM_BATCH_THRESHOLD
                    "q6_gemm_add"
                  elsif batch <= GEMM_BATCH_THRESHOLD && (add_pipe = gemv_add_pipeline_for(qw))
                    "gemv_add"
                  else
                    return false
                  end
          Profile.bump_matmul_shape("#{route} #{qw.type.name} #{in_dim}x#{out_dim} b#{batch}", qw.raw.size.to_i64)

          if q56_batch_gemm_enabled? && qw.type.q6_k? && batch > GEMM_BATCH_THRESHOLD
            encode_q56k_gemm_f32_add(enc, mm6_f32out_add_pipeline, x_buf, residual_buf, out_buf, w_buf, w_offset, in_dim, out_dim, batch)
          elsif batch <= GEMM_BATCH_THRESHOLD && (add_pipe = gemv_add_pipeline_for(qw))
            encode_gemv_add(enc, add_pipe, x_buf, residual_buf, out_buf, w_buf, w_offset, in_dim, out_dim, batch, profile_shape: false)
          else
            return false
          end
          true
        end

        private def self.encode_rmsnorm_vec(enc : ML::Metal::ComputeEncoder,
                                            x_buf : ML::MetalBuffer,
                                            weight_buf : ML::MetalBuffer,
                                            out_buf : ML::MetalBuffer,
                                            count : Int32,
                                            eps : Float32) : Nil
          enc.set_pipeline(rmsnorm_vec_pipeline)
          enc.set_buffer(x_buf, 0)
          enc.set_buffer(weight_buf, 1)
          enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_value(count.to_u32, 3)
          enc.set_value(eps, 4)
          enc.dispatch_threadgroups({1, 1, 1}, {256, 1, 1})
        end

        private def self.encode_add_rmsnorm(enc : ML::Metal::ComputeEncoder,
                                            x_buf : ML::MetalBuffer,
                                            y_buf : ML::MetalBuffer,
                                            weight_buf : ML::MetalBuffer,
                                            residual_buf : ML::MetalBuffer,
                                            normed_buf : ML::MetalBuffer,
                                            count : Int32,
                                            eps : Float32) : Nil
          enc.set_pipeline(add_rmsnorm_pipeline)
          enc.set_buffer(x_buf, 0)
          enc.set_buffer(y_buf, 1)
          enc.set_buffer(weight_buf, 2)
          enc.set_buffer(residual_buf, 3, ML::Metal::BufferAccess::Write)
          enc.set_buffer(normed_buf, 4, ML::Metal::BufferAccess::Write)
          enc.set_value(count.to_u32, 5)
          enc.set_value(eps, 6)
          enc.dispatch_threadgroups({1, 1, 1}, {256, 1, 1})
        end

        private def self.encode_rmsnorm_rows(enc : ML::Metal::ComputeEncoder,
                                             x_buf : ML::MetalBuffer,
                                             weight_buf : ML::MetalBuffer,
                                             out_buf : ML::MetalBuffer,
                                             dim : Int32,
                                             n_rows : Int32,
                                             eps : Float32) : Nil
          enc.set_pipeline(rmsnorm_rows_pipeline)
          enc.set_buffer(x_buf, 0)
          enc.set_buffer(weight_buf, 1)
          enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_value(dim.to_u32, 3)
          enc.set_value(n_rows.to_u32, 4)
          enc.set_value(eps, 5)
          enc.dispatch_threadgroups({n_rows, 1, 1}, {256, 1, 1})
        end

        private def self.encode_add_rmsnorm_rows(enc : ML::Metal::ComputeEncoder,
                                                 x_buf : ML::MetalBuffer,
                                                 y_buf : ML::MetalBuffer,
                                                 weight_buf : ML::MetalBuffer,
                                                 residual_buf : ML::MetalBuffer,
                                                 normed_buf : ML::MetalBuffer,
                                                 dim : Int32,
                                                 n_rows : Int32,
                                                 eps : Float32) : Nil
          enc.set_pipeline(add_rmsnorm_rows_pipeline)
          enc.set_buffer(x_buf, 0)
          enc.set_buffer(y_buf, 1)
          enc.set_buffer(weight_buf, 2)
          enc.set_buffer(residual_buf, 3, ML::Metal::BufferAccess::Write)
          enc.set_buffer(normed_buf, 4, ML::Metal::BufferAccess::Write)
          enc.set_value(dim.to_u32, 5)
          enc.set_value(n_rows.to_u32, 6)
          enc.set_value(eps, 7)
          enc.dispatch_threadgroups({n_rows, 1, 1}, {256, 1, 1})
        end

        # Apple Silicon uses unified memory for our buffers, so hot decode
        # paths can read directly from `contents` instead of bouncing
        # through the bridge's gs_buffer_read copy helper.
        private def self.read_shared_f32(buf : ML::MetalBuffer, count : Int32) : Array(Float32)
          ptr = buf.contents.as(Pointer(Float32))
          Array(Float32).build(count) do |dst|
            Slice.new(dst, count).copy_from(Slice.new(ptr, count))
            count
          end
        end

        private def self.write_shared_f16(buf : ML::MetalBuffer, values : Array(Float32)) : Nil
          ptr = buf.contents.as(Pointer(UInt16))
          values.each_with_index do |value, i|
            ptr[i] = f32_to_f16(value)
          end
        end

        private def self.read_shared_f16(buf : ML::MetalBuffer, count : Int32) : Array(Float32)
          ptr = buf.contents.as(Pointer(UInt16))
          Array(Float32).new(count) { |i| Dequant.fp16_to_f32(ptr[i]) }
        end

        private def self.f32_to_f16(f : Float32) : UInt16
          bits = f.unsafe_as(UInt32)
          sign = (bits >> 16) & 0x8000_u32
          exp = ((bits >> 23) & 0xff).to_i32 - 127 + 15
          mant = (bits >> 13) & 0x03ff_u32
          if exp <= 0
            sign.to_u16
          elsif exp >= 31
            (sign | 0x7c00_u32).to_u16
          else
            (sign | (exp.to_u32 << 10) | mant).to_u16
          end
        end

        private def self.q56_batch_gemm_enabled? : Bool
          ENV["QWEN35_Q56K_BATCH_GEMM_OFF"]? != "1"
        end

        private def self.swiglu_inplace_enabled? : Bool
          ENV["QWEN35_SWIGLU_INPLACE_OFF"]? != "1"
        end

        private def self.decode_swiglu_inplace_enabled? : Bool
          ENV["QWEN35_DECODE_SWIGLU_INPLACE"]? == "1"
        end

        private def self.ffn_down_add_fused_enabled? : Bool
          ENV["QWEN35_FFN_DOWN_ADD_FUSED_OFF"]? != "1"
        end

        private def self.prefill_ffn_down_add_fused_enabled? : Bool
          ENV["QWEN35_PREFILL_FFN_DOWN_ADD_FUSED"]? == "1"
        end

        private def self.small_q4_gemv_enabled? : Bool
          ENV["QWEN35_SMALL_Q4_GEMV_OFF"]? != "1"
        end

        private def self.q4_h16_gemm_enabled? : Bool
          ENV["QWEN35_Q4K_H16_GEMM_OFF"]? != "1"
        end

        private def self.q5_qkv_h16_conv_enabled? : Bool
          ENV["QWEN35_Q5_QKV_H16_CONV_OFF"]? != "1"
        end

        private def self.rec_proj_shared_h16_enabled? : Bool
          ENV["QWEN35_REC_PROJ_SHARED_H16_OFF"]? != "1"
        end

        private def self.q4_pair_h16_gemm_enabled? : Bool
          ENV["QWEN35_Q4K_PAIR_H16_GEMM_OFF"]? != "1"
        end

        private def self.q4_pair_h16_min_batch : Int32
          (ENV["QWEN35_Q4K_PAIR_H16_MIN_BATCH"]? || Q4_PAIR_H16_MIN_BATCH.to_s).to_i32
        end

        private def self.q8_dual_gemv_enabled? : Bool
          ENV["QWEN35_Q8_DUAL_GEMV_OFF"]? != "1"
        end

        private def self.q8_alpha_beta_dual_gemv_enabled? : Bool
          q8_dual_gemv_enabled? && ENV["QWEN35_Q8_ALPHA_BETA_DUAL_GEMV_OFF"]? != "1"
        end

        private def self.q8_kv_dual_gemv_enabled? : Bool
          q8_dual_gemv_enabled? && ENV["QWEN35_Q8_KV_DUAL_GEMV_OFF"]? != "1"
        end

        private def self.q8_dual_gemv_candidate?(gate_qw : QuantWeight,
                                                 up_qw : QuantWeight,
                                                 batch : Int32 = 1) : Bool
          q8_dual_gemv_enabled? && batch == 1 &&
            gate_qw.type.q8_0? && up_qw.type.q8_0? &&
            gate_qw.in_dim == up_qw.in_dim &&
            gate_qw.out_dim == up_qw.out_dim
        end

        private def self.q8_alpha_beta_dual_gemv_candidate?(alpha_qw : QuantWeight,
                                                            beta_qw : QuantWeight,
                                                            batch : Int32 = 1) : Bool
          q8_alpha_beta_dual_gemv_enabled? && batch == 1 &&
            alpha_qw.type.q8_0? && beta_qw.type.q8_0? &&
            alpha_qw.in_dim == beta_qw.in_dim &&
            alpha_qw.out_dim == beta_qw.out_dim
        end

        private def self.q8_kv_dual_gemv_candidate?(k_qw : QuantWeight,
                                                    v_qw : QuantWeight,
                                                    batch : Int32 = 1) : Bool
          q8_kv_dual_gemv_enabled? && batch == 1 &&
            k_qw.type.q8_0? && v_qw.type.q8_0? &&
            k_qw.in_dim == v_qw.in_dim &&
            k_qw.out_dim == v_qw.out_dim
        end

        private def self.q4_pair_h16_gemm_candidate?(gate_qw : QuantWeight,
                                                     up_qw : QuantWeight,
                                                     batch : Int32) : Bool
          q4_pair_h16_gemm_enabled? && q4_h16_gemm_enabled? &&
            batch >= q4_pair_h16_min_batch &&
            gate_qw.type.q4_k? && up_qw.type.q4_k? &&
            gate_qw.in_dim == up_qw.in_dim &&
            gate_qw.out_dim == up_qw.out_dim
        end

        private def self.read_shared_top1(id_buf : ML::MetalBuffer, value_buf : ML::MetalBuffer) : Array(Float32)
          id = id_buf.contents.as(Pointer(UInt32)).value
          value = value_buf.contents.as(Pointer(Float32)).value
          [id.to_f32, value]
        end

        private def self.read_shared_top1_rows(id_buf : ML::MetalBuffer, value_buf : ML::MetalBuffer, rows : Int32) : Array({Int32, Float32})
          ids = id_buf.contents.as(Pointer(UInt32))
          values = value_buf.contents.as(Pointer(Float32))
          Array({Int32, Float32}).new(rows) do |i|
            {ids[i].to_i32, values[i]}
          end
        end

        private def self.read_shared_top2_rows(top_id_buf : ML::MetalBuffer,
                                               top_value_buf : ML::MetalBuffer,
                                               second_id_buf : ML::MetalBuffer,
                                               second_value_buf : ML::MetalBuffer,
                                               rows : Int32) : Array({Int32, Float32, Int32, Float32})
          top_ids = top_id_buf.contents.as(Pointer(UInt32))
          top_values = top_value_buf.contents.as(Pointer(Float32))
          second_ids = second_id_buf.contents.as(Pointer(UInt32))
          second_values = second_value_buf.contents.as(Pointer(Float32))
          Array({Int32, Float32, Int32, Float32}).new(rows) do |i|
            {top_ids[i].to_i32, top_values[i], second_ids[i].to_i32, second_values[i]}
          end
        end

        private def self.head_top1_fused_enabled? : Bool
          ENV["QWEN35_HEAD_TOP1_FUSED"]? != "0"
        end

        private def self.wave_fast_command_buffer_enabled? : Bool
          ENV["QWEN35_WAVE_FAST_CMD"]? != "0"
        end

        private def self.recurrent_conv_shift_fused_enabled? : Bool
          ENV["QWEN35_REC_CONVSHIFT_FUSED"]? == "1"
        end

        private def self.wave_chunk_layers : Int32
          raw = ENV["QWEN35_WAVE_CHUNK_LAYERS"]?
          return 2 unless raw
          value = raw.to_i? || 0
          value > 0 ? value : 0
        end

        private def self.can_use_head_top1_fused?(output_qw : QuantWeight) : Bool
          head_top1_fused_enabled? &&
            ((output_qw.type.q6_k? && output_qw.in_dim % QK_K == 0) ||
              (output_qw.type.q8_0? && output_qw.in_dim % Q8_0_QK == 0))
        end

        # Gated attention decode on Metal. CPU side prepares Q (post-rmsnorm,
        # post-RoPE) and gate (raw, kernel applies sigmoid); K/V are already
        # appended to the per-layer k_cache_buf/v_cache_buf at row `pos`.
        # Returns the gated attention output `attn_o` as `Array(Float32)`
        # of length `n_head * head_dim`, ready for the output projection.
        def self.attn_decode(q : Array(Float32),
                             gate : Array(Float32),
                             k_cache_buf : ML::MetalBuffer,
                             v_cache_buf : ML::MetalBuffer,
                             pos : Int32, n_head : Int32, n_head_kv : Int32,
                             head_dim : Int32, heads_per_group : Int32,
                             scale : Float32) : Array(Float32)
          ML::Metal::Device.init!

          t0 = Time.instant if Profile.enabled?
          q_dim    = n_head * head_dim
          q_bytes  = q_dim.to_i64 * sizeof(Float32)
          q_buf    = Scratch.get(:attn_q,    q_bytes)
          gate_buf = Scratch.get(:attn_gate, q_bytes)
          out_buf  = Scratch.get(:attn_out,  q_bytes)
          q_buf.write(q)
          gate_buf.write(gate)

          cache_len = pos + 1

          cmd = ML::Metal::CommandBuffer.new
          enc = ML::Metal::ComputeEncoder.new(cmd)
          enc.set_pipeline(attn_pipeline)
          enc.set_buffer(q_buf,         0)
          enc.set_buffer(gate_buf,      1)
          enc.set_buffer(k_cache_buf,   2)
          enc.set_buffer(v_cache_buf,   3)
          enc.set_buffer(out_buf,       4, ML::Metal::BufferAccess::Write)
          enc.set_value(cache_len.to_u32,       5)
          enc.set_value(n_head.to_u32,          6)
          enc.set_value(n_head_kv.to_u32,       7)
          enc.set_value(head_dim.to_u32,        8)
          enc.set_value(heads_per_group.to_u32, 9)
          enc.set_value(scale,                 10)
          enc.dispatch_threadgroups({n_head, 1, 1}, {32, 1, 1})
          enc.end_encoding
          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_f32(out_buf, q_dim)
          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_attn(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          result
        end

        # Fused attention decode + output projection in a single command
        # buffer. This removes one CPU round-trip from the full-attention
        # layers: we keep attn_o GPU-resident and immediately feed it into
        # the quantized output projection.
        #
        # Returns nil when `out_qw` is not one of the Metal-routable K-quants.
        def self.attn_decode_project(q : Array(Float32),
                                     gate : Array(Float32),
                                     k_cache_buf : ML::MetalBuffer,
                                     v_cache_buf : ML::MetalBuffer,
                                     out_qw : QuantWeight,
                                     pos : Int32, n_head : Int32, n_head_kv : Int32,
                                     head_dim : Int32, heads_per_group : Int32,
                                     scale : Float32) : Array(Float32)?
          pipeline = gemv_pipeline_for(out_qw)
          return nil if pipeline.nil?

          ML::Metal::Device.init!

          q_dim = n_head * head_dim
          raise "attn output projection in_dim mismatch: expected #{q_dim}, got #{out_qw.in_dim}" unless out_qw.in_dim == q_dim

          t0 = Time.instant if Profile.enabled?
          q_bytes   = q_dim.to_i64 * sizeof(Float32)
          out_bytes = out_qw.out_dim.to_i64 * sizeof(Float32)
          q_buf     = Scratch.get(:attn_q,         q_bytes)
          gate_buf  = Scratch.get(:attn_gate,      q_bytes)
          attn_buf  = Scratch.get(:attn_proj_mid,  q_bytes)
          proj_buf  = Scratch.get(:attn_proj_out,  out_bytes)
          q_buf.write(q)
          gate_buf.write(gate)

          cache_len = pos + 1
          w_buf, w_off = if slot = mmap_slot_for(out_qw.raw)
                           slot
                         else
                           {out_qw.fallback_metal_buffer, 0_i64}
                         end

          cmd = ML::Metal::CommandBuffer.new

          attn_enc = ML::Metal::ComputeEncoder.new(cmd)
          attn_enc.set_pipeline(attn_pipeline)
          attn_enc.set_buffer(q_buf,         0)
          attn_enc.set_buffer(gate_buf,      1)
          attn_enc.set_buffer(k_cache_buf,   2)
          attn_enc.set_buffer(v_cache_buf,   3)
          attn_enc.set_buffer(attn_buf,      4, ML::Metal::BufferAccess::Write)
          attn_enc.set_value(cache_len.to_u32,       5)
          attn_enc.set_value(n_head.to_u32,          6)
          attn_enc.set_value(n_head_kv.to_u32,       7)
          attn_enc.set_value(head_dim.to_u32,        8)
          attn_enc.set_value(heads_per_group.to_u32, 9)
          attn_enc.set_value(scale,                 10)
          attn_enc.dispatch_threadgroups({n_head, 1, 1}, {32, 1, 1})
          attn_enc.end_encoding

          proj_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(proj_enc, pipeline.not_nil!, attn_buf, proj_buf, w_buf, w_off, out_qw.in_dim, out_qw.out_dim)
          proj_enc.end_encoding

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_f32(proj_buf, out_qw.out_dim)
          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_attn(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          result
        end

        # DeltaNet / GatedDeltaRule step on Metal.
        #
        # `state_buf` holds `h_v * s * s` floats in layout [h, d2, d1]
        # and is updated in place. `ghead[h]` is the pre-computed decay
        # multiplier (caller does `exp(softplus(...) * ssm_a[h])`);
        # `beta[h]` is already sigmoid'd.
        #
        # Returns the output `y` as `Array(Float32)` of length `h_v * s`.
        # NOTE: uploads inputs, dispatches, downloads output each call.
        # State stays GPU-resident across calls via `state_buf`.
        def self.delta_net_step(state_buf : ML::MetalBuffer,
                                q_conv : Array(Float32),
                                k_conv : Array(Float32),
                                v_conv : Array(Float32),
                                ghead : Array(Float32),
                                beta : Array(Float32),
                                h_k : Int32, h_v : Int32, s : Int32,
                                scale : Float32) : Array(Float32)
          ML::Metal::Device.init!

          t0 = Time.instant if Profile.enabled?
          q_buf  = Scratch.get(:dn_q,   q_conv.size.to_i64 * sizeof(Float32))
          k_buf  = Scratch.get(:dn_k,   k_conv.size.to_i64 * sizeof(Float32))
          v_buf  = Scratch.get(:dn_v,   v_conv.size.to_i64 * sizeof(Float32))
          g_buf  = Scratch.get(:dn_g,   ghead.size.to_i64  * sizeof(Float32))
          b_buf  = Scratch.get(:dn_b,   beta.size.to_i64   * sizeof(Float32))
          out_buf = Scratch.get(:dn_out, (h_v * s).to_i64  * sizeof(Float32))

          q_buf.write(q_conv)
          k_buf.write(k_conv)
          v_buf.write(v_conv)
          g_buf.write(ghead)
          b_buf.write(beta)

          cmd = ML::Metal::CommandBuffer.new
          enc = ML::Metal::ComputeEncoder.new(cmd)
          enc.set_pipeline(active_dn_pipeline)
          enc.set_buffer(state_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          enc.set_buffer(q_buf,     1)
          enc.set_buffer(k_buf,     2)
          enc.set_buffer(v_buf,     3)
          enc.set_buffer(g_buf,     4)
          enc.set_buffer(b_buf,     5)
          enc.set_buffer(out_buf,   6, ML::Metal::BufferAccess::Write)
          enc.set_value(h_k.to_u32,  7)
          enc.set_value(h_v.to_u32,  8)
          enc.set_value(s.to_u32,    9)
          enc.set_value(scale,      10)
          enc.dispatch_threadgroups({h_v, 1, 1}, {dn_threadgroup_size, 1, 1})
          enc.end_encoding
          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_f32(out_buf, h_v * s)
          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_dn(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          result
        end

        # Multi-token DeltaNet scan on Metal.
        #
        # Inputs are token-major:
        #   q/k    [n_tokens, h_k, s]
        #   v/out  [n_tokens, h_v, s]
        #   g/beta [n_tokens, h_v]
        #
        # This is a prefill building block: the recurrent scan over tokens is
        # still exact and serial, but it runs inside one dispatch per layer/head
        # chunk instead of launching one DeltaNet kernel per prompt token.
        def self.delta_net_chunk(state_buf : ML::MetalBuffer,
                                 q_conv : Array(Float32),
                                 k_conv : Array(Float32),
                                 v_conv : Array(Float32),
                                 ghead : Array(Float32),
                                 beta : Array(Float32),
                                 h_k : Int32, h_v : Int32, s : Int32,
                                 n_tokens : Int32,
                                 scale : Float32) : Array(Float32)
          ML::Metal::Device.init!
          raise "delta_net_chunk n_tokens must be positive" unless n_tokens > 0
          raise "delta_net_chunk q size mismatch" unless q_conv.size == n_tokens * h_k * s
          raise "delta_net_chunk k size mismatch" unless k_conv.size == n_tokens * h_k * s
          raise "delta_net_chunk v size mismatch" unless v_conv.size == n_tokens * h_v * s
          raise "delta_net_chunk g size mismatch" unless ghead.size == n_tokens * h_v
          raise "delta_net_chunk beta size mismatch" unless beta.size == n_tokens * h_v

          t0 = Time.instant if Profile.enabled?
          q_buf   = Scratch.get(:dn_chunk_q,   q_conv.size.to_i64 * sizeof(Float32))
          k_buf   = Scratch.get(:dn_chunk_k,   k_conv.size.to_i64 * sizeof(Float32))
          v_buf   = Scratch.get(:dn_chunk_v,   v_conv.size.to_i64 * sizeof(Float32))
          g_buf   = Scratch.get(:dn_chunk_g,   ghead.size.to_i64  * sizeof(Float32))
          b_buf   = Scratch.get(:dn_chunk_b,   beta.size.to_i64   * sizeof(Float32))
          out_buf = Scratch.get(:dn_chunk_out, (n_tokens * h_v * s).to_i64 * sizeof(Float32))

          q_buf.write(q_conv)
          k_buf.write(k_conv)
          v_buf.write(v_conv)
          g_buf.write(ghead)
          b_buf.write(beta)

          cmd = ML::Metal::CommandBuffer.new
          enc = ML::Metal::ComputeEncoder.new(cmd)
          use_rowwise = dn_chunk_rowwise_enabled?(s)
          enc.set_pipeline(use_rowwise ? dn128_chunk_rowwise_pipeline : dn128_chunk_fused_pipeline)
          enc.set_buffer(state_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          enc.set_buffer(q_buf,     1)
          enc.set_buffer(k_buf,     2)
          enc.set_buffer(v_buf,     3)
          enc.set_buffer(g_buf,     4)
          enc.set_buffer(b_buf,     5)
          enc.set_buffer(out_buf,   6, ML::Metal::BufferAccess::Write)
          enc.set_value(h_k.to_u32,       7)
          enc.set_value(h_v.to_u32,       8)
          enc.set_value(s.to_u32,         9)
          enc.set_value(scale,           10)
          enc.set_value(n_tokens.to_u32, 11)
          if use_rowwise
            enc.dispatch_threadgroups({(s + 3) // 4, h_v, 1}, {32, 4, 1})
          else
            enc.dispatch_threadgroups({h_v, 1, 1}, {128, 1, 1})
          end
          enc.end_encoding

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_f32(out_buf, n_tokens * h_v * s)
          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_dn(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          result
        end

        # Multi-token recurrent prep for Qwen35 prefill chunks.
        #
        # `qkv_mixed`, `alpha`, and `beta` are token-major outputs from the
        # recurrent input projections. The method updates `conv_state_buf`
        # exactly as repeated single-token `qwen35_recurrent_conv_shift` would,
        # applies L2 normalization to Q/K heads, transforms alpha/beta into
        # DeltaNet g/beta, and returns token-major arrays ready for
        # `delta_net_chunk`.
        def self.recurrent_prep_chunk(conv_state_buf : ML::MetalBuffer,
                                      qkv_mixed : Array(Float32),
                                      alpha : Array(Float32),
                                      beta : Array(Float32),
                                      ssm_conv1d : Array(Float32),
                                      ssm_dt_bias : Array(Float32),
                                      ssm_a : Array(Float32),
                                      h_k : Int32, h_v : Int32, s : Int32,
                                      conv_k : Int32,
                                      n_tokens : Int32,
                                      eps : Float32)
          ML::Metal::Device.init!
          qkv_dim = 2 * h_k * s + h_v * s
          q_dim = h_k * s
          v_dim = h_v * s
          raise "recurrent_prep_chunk n_tokens must be positive" unless n_tokens > 0
          raise "recurrent_prep_chunk qkv size mismatch" unless qkv_mixed.size == n_tokens * qkv_dim
          raise "recurrent_prep_chunk alpha size mismatch" unless alpha.size == n_tokens * h_v
          raise "recurrent_prep_chunk beta size mismatch" unless beta.size == n_tokens * h_v
          raise "recurrent_prep_chunk conv1d size mismatch" unless ssm_conv1d.size == qkv_dim * conv_k

          qkv_buf = Scratch.get(:rec_chunk_qkv, qkv_mixed.size.to_i64 * sizeof(Float32))
          conv_w_buf = Scratch.get(:rec_chunk_conv_w, ssm_conv1d.size.to_i64 * sizeof(Float32))
          q_buf = Scratch.get(:rec_chunk_q, (n_tokens * q_dim).to_i64 * sizeof(Float32))
          k_buf = Scratch.get(:rec_chunk_k, (n_tokens * q_dim).to_i64 * sizeof(Float32))
          v_buf = Scratch.get(:rec_chunk_v, (n_tokens * v_dim).to_i64 * sizeof(Float32))
          alpha_buf = Scratch.get(:rec_chunk_alpha, alpha.size.to_i64 * sizeof(Float32))
          beta_buf = Scratch.get(:rec_chunk_beta, beta.size.to_i64 * sizeof(Float32))
          dt_bias_buf = Scratch.get(:rec_chunk_dt_bias, ssm_dt_bias.size.to_i64 * sizeof(Float32))
          ssm_a_buf = Scratch.get(:rec_chunk_ssm_a, ssm_a.size.to_i64 * sizeof(Float32))
          g_buf = Scratch.get(:rec_chunk_g, (n_tokens * h_v).to_i64 * sizeof(Float32))

          qkv_buf.write(qkv_mixed)
          conv_w_buf.write(ssm_conv1d)
          alpha_buf.write(alpha)
          beta_buf.write(beta)
          dt_bias_buf.write(ssm_dt_bias)
          ssm_a_buf.write(ssm_a)

          cmd = ML::Metal::CommandBuffer.new

          conv_enc = ML::Metal::ComputeEncoder.new(cmd)
          conv_enc.set_pipeline(recurrent_conv_shift_chunk_pipeline)
          conv_enc.set_buffer(conv_state_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          conv_enc.set_buffer(qkv_buf,        1)
          conv_enc.set_buffer(conv_w_buf,     2)
          conv_enc.set_buffer(q_buf,          3, ML::Metal::BufferAccess::Write)
          conv_enc.set_buffer(k_buf,          4, ML::Metal::BufferAccess::Write)
          conv_enc.set_buffer(v_buf,          5, ML::Metal::BufferAccess::Write)
          conv_enc.set_value(h_k.to_u32,      6)
          conv_enc.set_value(h_v.to_u32,      7)
          conv_enc.set_value(s.to_u32,        8)
          conv_enc.set_value(conv_k.to_u32,   9)
          conv_enc.set_value(n_tokens.to_u32, 10)
          conv_enc.dispatch_1d(qkv_dim, 256)
          conv_enc.end_encoding

          qnorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          qnorm_enc.set_pipeline(l2_heads_chunk_pipeline)
          qnorm_enc.set_buffer(q_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          qnorm_enc.set_value(h_k.to_u32, 1)
          qnorm_enc.set_value(s.to_u32,   2)
          qnorm_enc.set_value(eps,        3)
          qnorm_enc.dispatch_threadgroups({h_k, n_tokens, 1}, {32, 1, 1})
          qnorm_enc.end_encoding

          knorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          knorm_enc.set_pipeline(l2_heads_chunk_pipeline)
          knorm_enc.set_buffer(k_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          knorm_enc.set_value(h_k.to_u32, 1)
          knorm_enc.set_value(s.to_u32,   2)
          knorm_enc.set_value(eps,        3)
          knorm_enc.dispatch_threadgroups({h_k, n_tokens, 1}, {32, 1, 1})
          knorm_enc.end_encoding

          ab_enc = ML::Metal::ComputeEncoder.new(cmd)
          ab_enc.set_pipeline(recurrent_ab_chunk_pipeline)
          ab_enc.set_buffer(alpha_buf,   0)
          ab_enc.set_buffer(beta_buf,    1, ML::Metal::BufferAccess::ReadWrite)
          ab_enc.set_buffer(dt_bias_buf, 2)
          ab_enc.set_buffer(ssm_a_buf,   3)
          ab_enc.set_buffer(g_buf,       4, ML::Metal::BufferAccess::Write)
          ab_enc.set_value(h_v.to_u32,       5)
          ab_enc.set_value(n_tokens.to_u32,  6)
          ab_enc.dispatch_1d(n_tokens * h_v, 64)
          ab_enc.end_encoding

          cmd.commit
          cmd.wait

          {
            read_shared_f32(q_buf, n_tokens * q_dim),
            read_shared_f32(k_buf, n_tokens * q_dim),
            read_shared_f32(v_buf, n_tokens * v_dim),
            read_shared_f32(g_buf, n_tokens * h_v),
            read_shared_f32(beta_buf, n_tokens * h_v),
          }
        end

        # Fused recurrent route:
        #   delta_net_step -> post RMSNorm*SiLU gate -> ssm_out matvec
        #
        # Keeps the DeltaNet output on GPU and only reads back the final
        # projected vector. Returns nil when `out_qw` is not Metal-routable.
        def self.delta_net_project(state_buf : ML::MetalBuffer,
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
          pipeline = gemv_pipeline_for(out_qw)
          return nil if pipeline.nil?

          ML::Metal::Device.init!

          inner_dim = h_v * s
          raise "delta_net output projection in_dim mismatch: expected #{inner_dim}, got #{out_qw.in_dim}" unless out_qw.in_dim == inner_dim
          raise "delta_net z size mismatch: expected #{inner_dim}, got #{z.size}" unless z.size == inner_dim
          raise "delta_net ssm_norm size mismatch: expected #{s}, got #{ssm_norm.size}" unless ssm_norm.size == s

          t0 = Time.instant if Profile.enabled?
          q_buf    = Scratch.get(:dn_q,         q_conv.size.to_i64 * sizeof(Float32))
          k_buf    = Scratch.get(:dn_k,         k_conv.size.to_i64 * sizeof(Float32))
          v_buf    = Scratch.get(:dn_v,         v_conv.size.to_i64 * sizeof(Float32))
          g_buf    = Scratch.get(:dn_g,         ghead.size.to_i64  * sizeof(Float32))
          b_buf    = Scratch.get(:dn_b,         beta.size.to_i64   * sizeof(Float32))
          z_buf    = Scratch.get(:dn_z,         z.size.to_i64      * sizeof(Float32))
          norm_buf = Scratch.get(:dn_norm,      ssm_norm.size.to_i64 * sizeof(Float32))
          mid_buf  = Scratch.get(:dn_out,       inner_dim.to_i64   * sizeof(Float32))
          proj_buf = Scratch.get(:dn_proj_out,  out_qw.out_dim.to_i64 * sizeof(Float32))

          q_buf.write(q_conv)
          k_buf.write(k_conv)
          v_buf.write(v_conv)
          g_buf.write(ghead)
          b_buf.write(beta)
          z_buf.write(z)
          norm_buf.write(ssm_norm)

          w_buf, w_off = if slot = mmap_slot_for(out_qw.raw)
                           slot
                         else
                           {out_qw.fallback_metal_buffer, 0_i64}
                         end

          cmd = ML::Metal::CommandBuffer.new

          dn_enc = ML::Metal::ComputeEncoder.new(cmd)
          dn_enc.set_pipeline(active_dn_pipeline)
          dn_enc.set_buffer(state_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          dn_enc.set_buffer(q_buf,     1)
          dn_enc.set_buffer(k_buf,     2)
          dn_enc.set_buffer(v_buf,     3)
          dn_enc.set_buffer(g_buf,     4)
          dn_enc.set_buffer(b_buf,     5)
          dn_enc.set_buffer(mid_buf,   6, ML::Metal::BufferAccess::Write)
          dn_enc.set_value(h_k.to_u32,  7)
          dn_enc.set_value(h_v.to_u32,  8)
          dn_enc.set_value(s.to_u32,    9)
          dn_enc.set_value(scale,      10)
          dn_enc.dispatch_threadgroups({h_v, 1, 1}, {dn_threadgroup_size, 1, 1})
          dn_enc.end_encoding

          post_enc = ML::Metal::ComputeEncoder.new(cmd)
          post_enc.set_pipeline(dn_post_pipeline)
          post_enc.set_buffer(mid_buf,  0, ML::Metal::BufferAccess::ReadWrite)
          post_enc.set_buffer(z_buf,    1)
          post_enc.set_buffer(norm_buf, 2)
          post_enc.set_value(h_v.to_u32, 3)
          post_enc.set_value(s.to_u32,   4)
          post_enc.set_value(eps,        5)
          post_enc.dispatch_threadgroups({h_v, 1, 1}, {32, 1, 1})
          post_enc.end_encoding

          proj_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(proj_enc, pipeline.not_nil!, mid_buf, proj_buf, w_buf, w_off, out_qw.in_dim, out_qw.out_dim)
          proj_enc.end_encoding

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_f32(proj_buf, out_qw.out_dim)
          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_dn(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          result
        end

        private def self.delta_net_project_buf(state_buf : ML::MetalBuffer,
                                               q_buf : ML::MetalBuffer,
                                               k_buf : ML::MetalBuffer,
                                               v_buf : ML::MetalBuffer,
                                               g_buf : ML::MetalBuffer,
                                               beta_buf : ML::MetalBuffer,
                                               z_buf : ML::MetalBuffer,
                                               norm_buf : ML::MetalBuffer,
                                               out_qw : QuantWeight,
                                               h_k : Int32, h_v : Int32, s : Int32,
                                               scale : Float32,
                                               eps : Float32) : Array(Float32)?
          pipeline = gemv_pipeline_for(out_qw)
          return nil if pipeline.nil?

          inner_dim = h_v * s
          raise "delta_net output projection in_dim mismatch: expected #{inner_dim}, got #{out_qw.in_dim}" unless out_qw.in_dim == inner_dim

          t0 = Time.instant if Profile.enabled?
          mid_buf  = Scratch.get(:dn_out,      inner_dim.to_i64 * sizeof(Float32))
          proj_buf = Scratch.get(:dn_proj_out, out_qw.out_dim.to_i64 * sizeof(Float32))

          w_buf, w_off = if slot = mmap_slot_for(out_qw.raw)
                           slot
                         else
                           {out_qw.fallback_metal_buffer, 0_i64}
                         end

          cmd = ML::Metal::CommandBuffer.new

          dn_enc = ML::Metal::ComputeEncoder.new(cmd)
          dn_enc.set_pipeline(active_dn_pipeline)
          dn_enc.set_buffer(state_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          dn_enc.set_buffer(q_buf,     1)
          dn_enc.set_buffer(k_buf,     2)
          dn_enc.set_buffer(v_buf,     3)
          dn_enc.set_buffer(g_buf,     4)
          dn_enc.set_buffer(beta_buf,  5)
          dn_enc.set_buffer(mid_buf,   6, ML::Metal::BufferAccess::Write)
          dn_enc.set_value(h_k.to_u32,  7)
          dn_enc.set_value(h_v.to_u32,  8)
          dn_enc.set_value(s.to_u32,    9)
          dn_enc.set_value(scale,      10)
          dn_enc.dispatch_threadgroups({h_v, 1, 1}, {dn_threadgroup_size, 1, 1})
          dn_enc.end_encoding

          post_enc = ML::Metal::ComputeEncoder.new(cmd)
          post_enc.set_pipeline(dn_post_pipeline)
          post_enc.set_buffer(mid_buf,   0, ML::Metal::BufferAccess::ReadWrite)
          post_enc.set_buffer(z_buf,     1)
          post_enc.set_buffer(norm_buf,  2)
          post_enc.set_value(h_v.to_u32, 3)
          post_enc.set_value(s.to_u32,   4)
          post_enc.set_value(eps,        5)
          post_enc.dispatch_threadgroups({h_v, 1, 1}, {32, 1, 1})
          post_enc.end_encoding

          proj_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(proj_enc, pipeline.not_nil!, mid_buf, proj_buf, w_buf, w_off, out_qw.in_dim, out_qw.out_dim)
          proj_enc.end_encoding

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_f32(proj_buf, out_qw.out_dim)
          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_dn(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          result
        end

        # Fused FFN route:
        #   gate_proj + up_proj -> swiglu combine -> down_proj
        # in one command buffer, with only the final projection read back.
        def self.ffn_project(x : Array(Float32),
                             gate_qw : QuantWeight,
                             up_qw : QuantWeight,
                             down_qw : QuantWeight) : Array(Float32)?
          gate_pipe = gemv_pipeline_for(gate_qw)
          up_pipe   = gemv_pipeline_for(up_qw)
          down_pipe = gemv_pipeline_for(down_qw)
          return nil if gate_pipe.nil? || up_pipe.nil? || down_pipe.nil?

          ML::Metal::Device.init!

          hidden_dim = x.size
          ffn_dim = gate_qw.out_dim
          raise "ffn gate in_dim mismatch: expected #{hidden_dim}, got #{gate_qw.in_dim}" unless gate_qw.in_dim == hidden_dim
          raise "ffn up shape mismatch" unless up_qw.in_dim == hidden_dim && up_qw.out_dim == ffn_dim
          raise "ffn down shape mismatch" unless down_qw.in_dim == ffn_dim

          t0 = Time.instant if Profile.enabled?
          x_buf      = Scratch.get(:ffn_x,       hidden_dim.to_i64 * sizeof(Float32))
          gate_buf   = Scratch.get(:ffn_gate,    ffn_dim.to_i64    * sizeof(Float32))
          up_buf     = Scratch.get(:ffn_up,      ffn_dim.to_i64    * sizeof(Float32))
          comb_buf   = Scratch.get(:ffn_comb,    ffn_dim.to_i64    * sizeof(Float32))
          out_buf    = Scratch.get(:ffn_out,     down_qw.out_dim.to_i64 * sizeof(Float32))
          x_buf.write(x)

          gate_w_buf, gate_w_off = if slot = mmap_slot_for(gate_qw.raw)
                                     slot
                                   else
                                     {gate_qw.fallback_metal_buffer, 0_i64}
                                   end
          up_w_buf, up_w_off = if slot = mmap_slot_for(up_qw.raw)
                                 slot
                               else
                                 {up_qw.fallback_metal_buffer, 0_i64}
                               end
          down_w_buf, down_w_off = if slot = mmap_slot_for(down_qw.raw)
                                     slot
                                   else
                                     {down_qw.fallback_metal_buffer, 0_i64}
                                   end

          cmd = ML::Metal::CommandBuffer.new

          proj_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(proj_enc, gate_pipe.not_nil!, x_buf, gate_buf, gate_w_buf, gate_w_off, gate_qw.in_dim, gate_qw.out_dim)
          encode_gemv(proj_enc, up_pipe.not_nil!, x_buf, up_buf, up_w_buf, up_w_off, up_qw.in_dim, up_qw.out_dim)
          proj_enc.end_encoding

          swiglu_enc = ML::Metal::ComputeEncoder.new(cmd)
          swiglu_enc.set_pipeline(ffn_swiglu_pipeline)
          swiglu_enc.set_buffer(gate_buf, 0)
          swiglu_enc.set_buffer(up_buf,   1)
          swiglu_enc.set_buffer(comb_buf, 2, ML::Metal::BufferAccess::Write)
          swiglu_enc.set_value(ffn_dim.to_u32, 3)
          swiglu_enc.dispatch_1d(ffn_dim, 256)
          swiglu_enc.end_encoding

          down_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(down_enc, down_pipe.not_nil!, comb_buf, out_buf, down_w_buf, down_w_off, down_qw.in_dim, down_qw.out_dim)
          down_enc.end_encoding

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_f32(out_buf, down_qw.out_dim)
          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_gemv(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          result
        end

        # Full recurrent attention projection on GPU:
        #   qkv/z/alpha/beta GEMVs -> alpha/beta transform -> conv -> L2 ->
        #   DeltaNet/post-norm/out projection.
        def self.recurrent_attn_project(x : Array(Float32),
                                        conv_state_buf : ML::MetalBuffer,
                                        ssm_state_buf : ML::MetalBuffer,
                                        attn_qkv_qw : QuantWeight,
                                        attn_gate_qw : QuantWeight,
                                        ssm_alpha_qw : QuantWeight,
                                        ssm_beta_qw : QuantWeight,
                                        ssm_conv1d : Array(Float32),
                                        ssm_dt_bias : Array(Float32),
                                        ssm_a : Array(Float32),
                                        ssm_norm : Array(Float32),
                                        ssm_out_qw : QuantWeight,
                                        h_k : Int32, h_v : Int32, s : Int32,
                                        conv_k : Int32,
                                        eps : Float32) : Array(Float32)?
          qkv_pipe   = gemv_pipeline_for(attn_qkv_qw)
          gate_pipe  = gemv_pipeline_for(attn_gate_qw)
          alpha_pipe = gemv_pipeline_for(ssm_alpha_qw)
          beta_pipe  = gemv_pipeline_for(ssm_beta_qw)
          return nil if qkv_pipe.nil? || gate_pipe.nil? || alpha_pipe.nil? || beta_pipe.nil?

          ML::Metal::Device.init!

          qkv_dim = 2 * h_k * s + h_v * s
          d_inner = h_v * s
          scale = (1.0 / Math.sqrt(s.to_f64)).to_f32

          x_buf       = Scratch.get(:rec_x,         x.size.to_i64 * sizeof(Float32))
          qkv_buf     = Scratch.get(:rec_qkv,       qkv_dim.to_i64 * sizeof(Float32))
          z_buf       = Scratch.get(:rec_z,         d_inner.to_i64 * sizeof(Float32))
          alpha_buf   = Scratch.get(:rec_alpha,     h_v.to_i64 * sizeof(Float32))
          beta_buf    = Scratch.get(:rec_beta,      h_v.to_i64 * sizeof(Float32))
          g_buf       = Scratch.get(:rec_g,         h_v.to_i64 * sizeof(Float32))
          conv_w_buf  = Scratch.get(:rec_conv_w,    ssm_conv1d.size.to_i64 * sizeof(Float32))
          dt_bias_buf = Scratch.get(:rec_dt_bias,   ssm_dt_bias.size.to_i64 * sizeof(Float32))
          ssm_a_buf   = Scratch.get(:rec_ssm_a,     ssm_a.size.to_i64 * sizeof(Float32))
          norm_buf    = Scratch.get(:rec_norm,      ssm_norm.size.to_i64 * sizeof(Float32))
          q_buf       = Scratch.get(:rec_q,         (h_k * s).to_i64 * sizeof(Float32))
          k_buf       = Scratch.get(:rec_k,         (h_k * s).to_i64 * sizeof(Float32))
          v_buf       = Scratch.get(:rec_v,         d_inner.to_i64 * sizeof(Float32))
          mid_buf     = Scratch.get(:dn_out,        d_inner.to_i64 * sizeof(Float32))
          proj_buf    = Scratch.get(:dn_proj_out,   ssm_out_qw.out_dim.to_i64 * sizeof(Float32))

          x_buf.write(x)
          conv_w_buf.write(ssm_conv1d)
          dt_bias_buf.write(ssm_dt_bias)
          ssm_a_buf.write(ssm_a)
          norm_buf.write(ssm_norm)

          qkv_w_buf, qkv_w_off = if slot = mmap_slot_for(attn_qkv_qw.raw)
                                   slot
                                 else
                                   {attn_qkv_qw.fallback_metal_buffer, 0_i64}
                                 end
          gate_w_buf, gate_w_off = if slot = mmap_slot_for(attn_gate_qw.raw)
                                     slot
                                   else
                                     {attn_gate_qw.fallback_metal_buffer, 0_i64}
                                   end
          alpha_w_buf, alpha_w_off = if slot = mmap_slot_for(ssm_alpha_qw.raw)
                                       slot
                                     else
                                       {ssm_alpha_qw.fallback_metal_buffer, 0_i64}
                                     end
          beta_w_buf, beta_w_off = if slot = mmap_slot_for(ssm_beta_qw.raw)
                                     slot
                                   else
                                     {ssm_beta_qw.fallback_metal_buffer, 0_i64}
                                   end
          out_w_buf, out_w_off = if slot = mmap_slot_for(ssm_out_qw.raw)
                                   slot
                                 else
                                   {ssm_out_qw.fallback_metal_buffer, 0_i64}
                                 end

          t0 = Time.instant if Profile.enabled?
          cmd = ML::Metal::CommandBuffer.new

          proj_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(proj_enc, qkv_pipe.not_nil!,   x_buf, qkv_buf,   qkv_w_buf,   qkv_w_off,   attn_qkv_qw.in_dim,  attn_qkv_qw.out_dim)
          encode_gemv(proj_enc, gate_pipe.not_nil!,  x_buf, z_buf,     gate_w_buf,  gate_w_off,  attn_gate_qw.in_dim, attn_gate_qw.out_dim)
          encode_gemv(proj_enc, alpha_pipe.not_nil!, x_buf, alpha_buf, alpha_w_buf, alpha_w_off, ssm_alpha_qw.in_dim, ssm_alpha_qw.out_dim)
          encode_gemv(proj_enc, beta_pipe.not_nil!,  x_buf, beta_buf,  beta_w_buf,  beta_w_off,  ssm_beta_qw.in_dim,  ssm_beta_qw.out_dim)
          proj_enc.end_encoding

          conv_enc = ML::Metal::ComputeEncoder.new(cmd)
          conv_enc.set_pipeline(recurrent_conv_pipeline)
          conv_enc.set_buffer(conv_state_buf, 0)
          conv_enc.set_buffer(qkv_buf,        1)
          conv_enc.set_buffer(conv_w_buf,     2)
          conv_enc.set_buffer(q_buf,          3, ML::Metal::BufferAccess::Write)
          conv_enc.set_buffer(k_buf,          4, ML::Metal::BufferAccess::Write)
          conv_enc.set_buffer(v_buf,          5, ML::Metal::BufferAccess::Write)
          conv_enc.set_value(h_k.to_u32,      6)
          conv_enc.set_value(h_v.to_u32,      7)
          conv_enc.set_value(s.to_u32,        8)
          conv_enc.set_value(conv_k.to_u32,   9)
          conv_enc.dispatch_1d(qkv_dim, 256)
          conv_enc.end_encoding

          shift_enc = ML::Metal::ComputeEncoder.new(cmd)
          shift_enc.set_pipeline(recurrent_shift_pipeline)
          shift_enc.set_buffer(conv_state_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          shift_enc.set_buffer(qkv_buf,        1)
          shift_enc.set_value(qkv_dim.to_u32,  2)
          shift_enc.set_value(conv_k.to_u32,   3)
          shift_enc.dispatch_1d(qkv_dim, 256)
          shift_enc.end_encoding

          qnorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          qnorm_enc.set_pipeline(l2_heads_pipeline)
          qnorm_enc.set_buffer(q_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          qnorm_enc.set_value(s.to_u32, 1)
          qnorm_enc.set_value(eps, 2)
          qnorm_enc.dispatch_threadgroups({h_k, 1, 1}, {32, 1, 1})
          qnorm_enc.end_encoding

          knorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          knorm_enc.set_pipeline(l2_heads_pipeline)
          knorm_enc.set_buffer(k_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          knorm_enc.set_value(s.to_u32, 1)
          knorm_enc.set_value(eps, 2)
          knorm_enc.dispatch_threadgroups({h_k, 1, 1}, {32, 1, 1})
          knorm_enc.end_encoding

          ab_enc = ML::Metal::ComputeEncoder.new(cmd)
          ab_enc.set_pipeline(recurrent_ab_pipeline)
          ab_enc.set_buffer(alpha_buf,   0)
          ab_enc.set_buffer(beta_buf,    1, ML::Metal::BufferAccess::ReadWrite)
          ab_enc.set_buffer(dt_bias_buf, 2)
          ab_enc.set_buffer(ssm_a_buf,   3)
          ab_enc.set_buffer(g_buf,       4, ML::Metal::BufferAccess::Write)
          ab_enc.set_value(h_v.to_u32,   5)
          ab_enc.dispatch_1d(h_v, 32)
          ab_enc.end_encoding

          dn_enc = ML::Metal::ComputeEncoder.new(cmd)
          dn_enc.set_pipeline(active_dn_pipeline)
          dn_enc.set_buffer(ssm_state_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          dn_enc.set_buffer(q_buf,         1)
          dn_enc.set_buffer(k_buf,         2)
          dn_enc.set_buffer(v_buf,         3)
          dn_enc.set_buffer(g_buf,         4)
          dn_enc.set_buffer(beta_buf,      5)
          dn_enc.set_buffer(mid_buf,       6, ML::Metal::BufferAccess::Write)
          dn_enc.set_value(h_k.to_u32,     7)
          dn_enc.set_value(h_v.to_u32,     8)
          dn_enc.set_value(s.to_u32,       9)
          dn_enc.set_value(scale,         10)
          dn_enc.dispatch_threadgroups({h_v, 1, 1}, {dn_threadgroup_size, 1, 1})
          dn_enc.end_encoding

          post_enc = ML::Metal::ComputeEncoder.new(cmd)
          post_enc.set_pipeline(dn_post_pipeline)
          post_enc.set_buffer(mid_buf,   0, ML::Metal::BufferAccess::ReadWrite)
          post_enc.set_buffer(z_buf,     1)
          post_enc.set_buffer(norm_buf,  2)
          post_enc.set_value(h_v.to_u32, 3)
          post_enc.set_value(s.to_u32,   4)
          post_enc.set_value(eps,        5)
          post_enc.dispatch_threadgroups({h_v, 1, 1}, {32, 1, 1})
          post_enc.end_encoding

          out_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(out_enc, gemv_pipeline_for(ssm_out_qw).not_nil!, mid_buf, proj_buf, out_w_buf, out_w_off, ssm_out_qw.in_dim, ssm_out_qw.out_dim)
          out_enc.end_encoding

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_f32(proj_buf, ssm_out_qw.out_dim)
          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_dn(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          result
        end

        # Fused recurrent layer route:
        #   recurrent attention projection ->
        #   residual add + post-attn RMSNorm ->
        #   SwiGLU FFN ->
        #   final residual add
        # in one command buffer, with only the final layer output read back.
        def self.recurrent_layer_project(inp : Array(Float32),
                                         cur : Array(Float32),
                                         conv_state_buf : ML::MetalBuffer,
                                         ssm_state_buf : ML::MetalBuffer,
                                         attn_qkv_qw : QuantWeight,
                                         attn_gate_qw : QuantWeight,
                                         ssm_alpha_qw : QuantWeight,
                                         ssm_beta_qw : QuantWeight,
                                         ssm_conv1d : Array(Float32),
                                         ssm_dt_bias : Array(Float32),
                                         ssm_a : Array(Float32),
                                         ssm_norm : Array(Float32),
                                         ssm_out_qw : QuantWeight,
                                         post_attention_norm : Array(Float32),
                                         ffn_gate_qw : QuantWeight,
                                         ffn_up_qw : QuantWeight,
                                         ffn_down_qw : QuantWeight,
                                         h_k : Int32, h_v : Int32, s : Int32,
                                         conv_k : Int32,
                                         eps : Float32) : Array(Float32)?
          qkv_pipe   = gemv_pipeline_for(attn_qkv_qw)
          gate_pipe  = gemv_pipeline_for(attn_gate_qw)
          alpha_pipe = gemv_pipeline_for(ssm_alpha_qw)
          beta_pipe  = gemv_pipeline_for(ssm_beta_qw)
          out_pipe   = gemv_pipeline_for(ssm_out_qw)
          ffn_gate_pipe = gemv_pipeline_for(ffn_gate_qw)
          ffn_up_pipe   = gemv_pipeline_for(ffn_up_qw)
          ffn_down_pipe = gemv_pipeline_for(ffn_down_qw)
          return nil if qkv_pipe.nil? || gate_pipe.nil? || alpha_pipe.nil? || beta_pipe.nil? ||
                        out_pipe.nil? || ffn_gate_pipe.nil? || ffn_up_pipe.nil? || ffn_down_pipe.nil?

          ML::Metal::Device.init!

          hidden_dim = inp.size
          qkv_dim = 2 * h_k * s + h_v * s
          d_inner = h_v * s
          ffn_dim = ffn_gate_qw.out_dim
          scale = (1.0 / Math.sqrt(s.to_f64)).to_f32

          inp_buf      = Scratch.get(:recl_inp,        hidden_dim.to_i64 * sizeof(Float32))
          cur_buf      = Scratch.get(:recl_cur,        cur.size.to_i64 * sizeof(Float32))
          qkv_buf      = Scratch.get(:recl_qkv,        qkv_dim.to_i64 * sizeof(Float32))
          z_buf        = Scratch.get(:recl_z,          d_inner.to_i64 * sizeof(Float32))
          alpha_buf    = Scratch.get(:recl_alpha,      h_v.to_i64 * sizeof(Float32))
          beta_buf     = Scratch.get(:recl_beta,       h_v.to_i64 * sizeof(Float32))
          g_buf        = Scratch.get(:recl_g,          h_v.to_i64 * sizeof(Float32))
          conv_w_buf   = Scratch.get(:recl_conv_w,     ssm_conv1d.size.to_i64 * sizeof(Float32))
          dt_bias_buf  = Scratch.get(:recl_dt_bias,    ssm_dt_bias.size.to_i64 * sizeof(Float32))
          ssm_a_buf    = Scratch.get(:recl_ssm_a,      ssm_a.size.to_i64 * sizeof(Float32))
          ssm_norm_buf = Scratch.get(:recl_ssm_norm,   ssm_norm.size.to_i64 * sizeof(Float32))
          q_buf        = Scratch.get(:recl_q,          (h_k * s).to_i64 * sizeof(Float32))
          k_buf        = Scratch.get(:recl_k,          (h_k * s).to_i64 * sizeof(Float32))
          v_buf        = Scratch.get(:recl_v,          d_inner.to_i64 * sizeof(Float32))
          attn_mid_buf = Scratch.get(:recl_attn_mid,   d_inner.to_i64 * sizeof(Float32))
          attn_out_buf = Scratch.get(:recl_attn_out,   ssm_out_qw.out_dim.to_i64 * sizeof(Float32))
          post_norm_buf = Scratch.get(:recl_postnorm_w, post_attention_norm.size.to_i64 * sizeof(Float32))
          residual_buf = Scratch.get(:recl_residual,   hidden_dim.to_i64 * sizeof(Float32))
          normed_buf   = Scratch.get(:recl_normed,     hidden_dim.to_i64 * sizeof(Float32))
          ffn_gate_buf = Scratch.get(:recl_ffn_gate,   ffn_dim.to_i64 * sizeof(Float32))
          ffn_up_buf   = Scratch.get(:recl_ffn_up,     ffn_dim.to_i64 * sizeof(Float32))
          ffn_comb_buf = Scratch.get(:recl_ffn_comb,   ffn_dim.to_i64 * sizeof(Float32))
          ffn_out_buf  = Scratch.get(:recl_ffn_out,    ffn_down_qw.out_dim.to_i64 * sizeof(Float32))
          out_buf      = Scratch.get(:recl_out,        hidden_dim.to_i64 * sizeof(Float32))

          inp_buf.write(inp)
          cur_buf.write(cur)
          conv_w_buf.write(ssm_conv1d)
          dt_bias_buf.write(ssm_dt_bias)
          ssm_a_buf.write(ssm_a)
          ssm_norm_buf.write(ssm_norm)
          post_norm_buf.write(post_attention_norm)

          qkv_w_buf, qkv_w_off = if slot = mmap_slot_for(attn_qkv_qw.raw)
                                   slot
                                 else
                                   {attn_qkv_qw.fallback_metal_buffer, 0_i64}
                                 end
          gate_w_buf, gate_w_off = if slot = mmap_slot_for(attn_gate_qw.raw)
                                     slot
                                   else
                                     {attn_gate_qw.fallback_metal_buffer, 0_i64}
                                   end
          alpha_w_buf, alpha_w_off = if slot = mmap_slot_for(ssm_alpha_qw.raw)
                                       slot
                                     else
                                       {ssm_alpha_qw.fallback_metal_buffer, 0_i64}
                                     end
          beta_w_buf, beta_w_off = if slot = mmap_slot_for(ssm_beta_qw.raw)
                                     slot
                                   else
                                     {ssm_beta_qw.fallback_metal_buffer, 0_i64}
                                   end
          out_w_buf, out_w_off = if slot = mmap_slot_for(ssm_out_qw.raw)
                                   slot
                                 else
                                   {ssm_out_qw.fallback_metal_buffer, 0_i64}
                                 end
          ffn_gate_w_buf, ffn_gate_w_off = if slot = mmap_slot_for(ffn_gate_qw.raw)
                                             slot
                                           else
                                             {ffn_gate_qw.fallback_metal_buffer, 0_i64}
                                           end
          ffn_up_w_buf, ffn_up_w_off = if slot = mmap_slot_for(ffn_up_qw.raw)
                                         slot
                                       else
                                         {ffn_up_qw.fallback_metal_buffer, 0_i64}
                                       end
          ffn_down_w_buf, ffn_down_w_off = if slot = mmap_slot_for(ffn_down_qw.raw)
                                             slot
                                           else
                                             {ffn_down_qw.fallback_metal_buffer, 0_i64}
                                           end

          t0 = Time.instant if Profile.enabled?
          cmd = ML::Metal::CommandBuffer.new

          proj_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(proj_enc, qkv_pipe.not_nil!,   cur_buf, qkv_buf,   qkv_w_buf,   qkv_w_off,   attn_qkv_qw.in_dim,  attn_qkv_qw.out_dim)
          encode_gemv(proj_enc, gate_pipe.not_nil!,  cur_buf, z_buf,     gate_w_buf,  gate_w_off,  attn_gate_qw.in_dim, attn_gate_qw.out_dim)
          if q8_alpha_beta_dual_gemv_candidate?(ssm_alpha_qw, ssm_beta_qw)
            encode_gemv_q8_dual(proj_enc, cur_buf, alpha_buf, beta_buf,
              alpha_w_buf, alpha_w_off, beta_w_buf, beta_w_off,
              ssm_alpha_qw.in_dim, ssm_alpha_qw.out_dim)
          else
            encode_gemv(proj_enc, alpha_pipe.not_nil!, cur_buf, alpha_buf, alpha_w_buf, alpha_w_off, ssm_alpha_qw.in_dim, ssm_alpha_qw.out_dim)
            encode_gemv(proj_enc, beta_pipe.not_nil!,  cur_buf, beta_buf,  beta_w_buf,  beta_w_off,  ssm_beta_qw.in_dim,  ssm_beta_qw.out_dim)
          end
          proj_enc.end_encoding

          conv_enc = ML::Metal::ComputeEncoder.new(cmd)
          conv_enc.set_pipeline(recurrent_conv_pipeline)
          conv_enc.set_buffer(conv_state_buf, 0)
          conv_enc.set_buffer(qkv_buf,        1)
          conv_enc.set_buffer(conv_w_buf,     2)
          conv_enc.set_buffer(q_buf,          3, ML::Metal::BufferAccess::Write)
          conv_enc.set_buffer(k_buf,          4, ML::Metal::BufferAccess::Write)
          conv_enc.set_buffer(v_buf,          5, ML::Metal::BufferAccess::Write)
          conv_enc.set_value(h_k.to_u32,      6)
          conv_enc.set_value(h_v.to_u32,      7)
          conv_enc.set_value(s.to_u32,        8)
          conv_enc.set_value(conv_k.to_u32,   9)
          conv_enc.dispatch_1d(qkv_dim, 256)
          conv_enc.end_encoding

          shift_enc = ML::Metal::ComputeEncoder.new(cmd)
          shift_enc.set_pipeline(recurrent_shift_pipeline)
          shift_enc.set_buffer(conv_state_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          shift_enc.set_buffer(qkv_buf,        1)
          shift_enc.set_value(qkv_dim.to_u32,  2)
          shift_enc.set_value(conv_k.to_u32,   3)
          shift_enc.dispatch_1d(qkv_dim, 256)
          shift_enc.end_encoding

          qnorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          qnorm_enc.set_pipeline(l2_heads_pipeline)
          qnorm_enc.set_buffer(q_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          qnorm_enc.set_value(s.to_u32, 1)
          qnorm_enc.set_value(eps, 2)
          qnorm_enc.dispatch_threadgroups({h_k, 1, 1}, {32, 1, 1})
          qnorm_enc.end_encoding

          knorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          knorm_enc.set_pipeline(l2_heads_pipeline)
          knorm_enc.set_buffer(k_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          knorm_enc.set_value(s.to_u32, 1)
          knorm_enc.set_value(eps, 2)
          knorm_enc.dispatch_threadgroups({h_k, 1, 1}, {32, 1, 1})
          knorm_enc.end_encoding

          ab_enc = ML::Metal::ComputeEncoder.new(cmd)
          ab_enc.set_pipeline(recurrent_ab_pipeline)
          ab_enc.set_buffer(alpha_buf,   0)
          ab_enc.set_buffer(beta_buf,    1, ML::Metal::BufferAccess::ReadWrite)
          ab_enc.set_buffer(dt_bias_buf, 2)
          ab_enc.set_buffer(ssm_a_buf,   3)
          ab_enc.set_buffer(g_buf,       4, ML::Metal::BufferAccess::Write)
          ab_enc.set_value(h_v.to_u32,   5)
          ab_enc.dispatch_1d(h_v, 32)
          ab_enc.end_encoding

          dn_enc = ML::Metal::ComputeEncoder.new(cmd)
          dn_enc.set_pipeline(active_dn_pipeline)
          dn_enc.set_buffer(ssm_state_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          dn_enc.set_buffer(q_buf,         1)
          dn_enc.set_buffer(k_buf,         2)
          dn_enc.set_buffer(v_buf,         3)
          dn_enc.set_buffer(g_buf,         4)
          dn_enc.set_buffer(beta_buf,      5)
          dn_enc.set_buffer(attn_mid_buf,  6, ML::Metal::BufferAccess::Write)
          dn_enc.set_value(h_k.to_u32,     7)
          dn_enc.set_value(h_v.to_u32,     8)
          dn_enc.set_value(s.to_u32,       9)
          dn_enc.set_value(scale,         10)
          dn_enc.dispatch_threadgroups({h_v, 1, 1}, {dn_threadgroup_size, 1, 1})
          dn_enc.end_encoding

          post_enc = ML::Metal::ComputeEncoder.new(cmd)
          post_enc.set_pipeline(dn_post_pipeline)
          post_enc.set_buffer(attn_mid_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          post_enc.set_buffer(z_buf,        1)
          post_enc.set_buffer(ssm_norm_buf, 2)
          post_enc.set_value(h_v.to_u32,    3)
          post_enc.set_value(s.to_u32,      4)
          post_enc.set_value(eps,           5)
          post_enc.dispatch_threadgroups({h_v, 1, 1}, {32, 1, 1})
          post_enc.end_encoding

          out_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(out_enc, out_pipe.not_nil!, attn_mid_buf, attn_out_buf, out_w_buf, out_w_off, ssm_out_qw.in_dim, ssm_out_qw.out_dim)
          out_enc.end_encoding

          addnorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          addnorm_enc.set_pipeline(add_rmsnorm_pipeline)
          addnorm_enc.set_buffer(inp_buf,       0)
          addnorm_enc.set_buffer(attn_out_buf,  1)
          addnorm_enc.set_buffer(post_norm_buf, 2)
          addnorm_enc.set_buffer(residual_buf,  3, ML::Metal::BufferAccess::Write)
          addnorm_enc.set_buffer(normed_buf,    4, ML::Metal::BufferAccess::Write)
          addnorm_enc.set_value(hidden_dim.to_u32, 5)
          addnorm_enc.set_value(eps,               6)
          addnorm_enc.dispatch_threadgroups({1, 1, 1}, {256, 1, 1})
          addnorm_enc.end_encoding

          ffn_proj_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(ffn_proj_enc, ffn_gate_pipe.not_nil!, normed_buf, ffn_gate_buf, ffn_gate_w_buf, ffn_gate_w_off, ffn_gate_qw.in_dim, ffn_gate_qw.out_dim)
          encode_gemv(ffn_proj_enc, ffn_up_pipe.not_nil!,   normed_buf, ffn_up_buf,   ffn_up_w_buf,   ffn_up_w_off,   ffn_up_qw.in_dim,   ffn_up_qw.out_dim)
          ffn_proj_enc.end_encoding

          swiglu_enc = ML::Metal::ComputeEncoder.new(cmd)
          swiglu_enc.set_pipeline(ffn_swiglu_pipeline)
          swiglu_enc.set_buffer(ffn_gate_buf, 0)
          swiglu_enc.set_buffer(ffn_up_buf,   1)
          swiglu_enc.set_buffer(ffn_comb_buf, 2, ML::Metal::BufferAccess::Write)
          swiglu_enc.set_value(ffn_dim.to_u32, 3)
          swiglu_enc.dispatch_1d(ffn_dim, 256)
          swiglu_enc.end_encoding

          ffn_down_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(ffn_down_enc, ffn_down_pipe.not_nil!, ffn_comb_buf, ffn_out_buf, ffn_down_w_buf, ffn_down_w_off, ffn_down_qw.in_dim, ffn_down_qw.out_dim)
          ffn_down_enc.end_encoding

          add_enc = ML::Metal::ComputeEncoder.new(cmd)
          add_enc.set_pipeline(add_vec_pipeline)
          add_enc.set_buffer(residual_buf, 0)
          add_enc.set_buffer(ffn_out_buf,  1)
          add_enc.set_buffer(out_buf,      2, ML::Metal::BufferAccess::Write)
          add_enc.set_value(hidden_dim.to_u32, 3)
          add_enc.dispatch_1d(hidden_dim, 256)
          add_enc.end_encoding

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_f32(out_buf, hidden_dim)
          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_dn(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          result
        end

        # Multi-token recurrent layer route for prefill chunks.
        #
        # Keeps the whole recurrent layer on GPU for a token-major chunk:
        # row RMSNorm -> batched qkv/z/alpha/beta projections -> chunked
        # conv/L2/alpha-beta -> chunked DeltaNet -> batched ssm_out ->
        # row add+RMSNorm -> batched FFN -> final residual add.
        #
        # This is an exact primitive for recurrent layers. It still reads the
        # final chunk output back because the surrounding prefill engine is not
        # yet a full Metal-side wave.
        def self.recurrent_layer_chunk_project(inp : Array(Float32),
                                               conv_state_buf : ML::MetalBuffer,
                                               ssm_state_buf : ML::MetalBuffer,
                                               attn_norm : Array(Float32),
                                               attn_qkv_qw : QuantWeight,
                                               attn_gate_qw : QuantWeight,
                                               ssm_alpha_qw : QuantWeight,
                                               ssm_beta_qw : QuantWeight,
                                               ssm_conv1d : Array(Float32),
                                               ssm_dt_bias : Array(Float32),
                                               ssm_a : Array(Float32),
                                               ssm_norm : Array(Float32),
                                               ssm_out_qw : QuantWeight,
                                               post_attention_norm : Array(Float32),
                                               ffn_gate_qw : QuantWeight,
                                               ffn_up_qw : QuantWeight,
                                               ffn_down_qw : QuantWeight,
                                               h_k : Int32, h_v : Int32, s : Int32,
                                               conv_k : Int32,
                                               n_tokens : Int32,
                                               eps : Float32) : Array(Float32)?
          qkv_pipe   = gemv_pipeline_for(attn_qkv_qw)
          gate_pipe  = gemv_pipeline_for(attn_gate_qw)
          alpha_pipe = gemv_pipeline_for(ssm_alpha_qw)
          beta_pipe  = gemv_pipeline_for(ssm_beta_qw)
          out_pipe   = gemv_pipeline_for(ssm_out_qw)
          ffn_gate_pipe = gemv_pipeline_for(ffn_gate_qw)
          ffn_up_pipe   = gemv_pipeline_for(ffn_up_qw)
          ffn_down_pipe = gemv_pipeline_for(ffn_down_qw)
          return nil if qkv_pipe.nil? || gate_pipe.nil? || alpha_pipe.nil? || beta_pipe.nil? ||
                        out_pipe.nil? || ffn_gate_pipe.nil? || ffn_up_pipe.nil? || ffn_down_pipe.nil?
          return nil unless n_tokens > 0

          ML::Metal::Device.init!

          hidden_dim = attn_qkv_qw.in_dim
          qkv_dim = 2 * h_k * s + h_v * s
          d_inner = h_v * s
          ffn_dim = ffn_gate_qw.out_dim
          scale = (1.0 / Math.sqrt(s.to_f64)).to_f32
          raise "recurrent_layer_chunk input size mismatch" unless inp.size == n_tokens * hidden_dim

          inp_buf       = Scratch.get(:rec_chunk_layer_inp,       inp.size.to_i64 * sizeof(Float32))
          norm_w_buf    = Scratch.get(:rec_chunk_layer_norm_w,    attn_norm.size.to_i64 * sizeof(Float32))
          cur_buf       = Scratch.get(:rec_chunk_layer_cur,       inp.size.to_i64 * sizeof(Float32))
          qkv_buf       = Scratch.get(:rec_chunk_layer_qkv,       (n_tokens * qkv_dim).to_i64 * sizeof(Float32))
          z_buf         = Scratch.get(:rec_chunk_layer_z,         (n_tokens * d_inner).to_i64 * sizeof(Float32))
          alpha_buf     = Scratch.get(:rec_chunk_layer_alpha,     (n_tokens * h_v).to_i64 * sizeof(Float32))
          beta_buf      = Scratch.get(:rec_chunk_layer_beta,      (n_tokens * h_v).to_i64 * sizeof(Float32))
          g_buf         = Scratch.get(:rec_chunk_layer_g,         (n_tokens * h_v).to_i64 * sizeof(Float32))
          conv_w_buf    = Scratch.get(:rec_chunk_layer_conv_w,    ssm_conv1d.size.to_i64 * sizeof(Float32))
          dt_bias_buf   = Scratch.get(:rec_chunk_layer_dt_bias,   ssm_dt_bias.size.to_i64 * sizeof(Float32))
          ssm_a_buf     = Scratch.get(:rec_chunk_layer_ssm_a,     ssm_a.size.to_i64 * sizeof(Float32))
          ssm_norm_buf  = Scratch.get(:rec_chunk_layer_ssm_norm,  ssm_norm.size.to_i64 * sizeof(Float32))
          q_buf         = Scratch.get(:rec_chunk_layer_q,         (n_tokens * h_k * s).to_i64 * sizeof(Float32))
          k_buf         = Scratch.get(:rec_chunk_layer_k,         (n_tokens * h_k * s).to_i64 * sizeof(Float32))
          v_buf         = Scratch.get(:rec_chunk_layer_v,         (n_tokens * d_inner).to_i64 * sizeof(Float32))
          attn_mid_buf  = Scratch.get(:rec_chunk_layer_mid,       (n_tokens * d_inner).to_i64 * sizeof(Float32))
          attn_out_buf  = Scratch.get(:rec_chunk_layer_attn_out,  (n_tokens * ssm_out_qw.out_dim).to_i64 * sizeof(Float32))
          post_w_buf    = Scratch.get(:rec_chunk_layer_post_w,    post_attention_norm.size.to_i64 * sizeof(Float32))
          residual_buf  = Scratch.get(:rec_chunk_layer_residual,  inp.size.to_i64 * sizeof(Float32))
          normed_buf    = Scratch.get(:rec_chunk_layer_normed,    inp.size.to_i64 * sizeof(Float32))
          ffn_gate_buf  = Scratch.get(:rec_chunk_layer_ffn_gate,  (n_tokens * ffn_dim).to_i64 * sizeof(Float32))
          ffn_up_buf    = Scratch.get(:rec_chunk_layer_ffn_up,    (n_tokens * ffn_dim).to_i64 * sizeof(Float32))
          ffn_comb_buf  = Scratch.get(:rec_chunk_layer_ffn_comb,  (n_tokens * ffn_dim).to_i64 * sizeof(Float32))
          ffn_out_buf   = Scratch.get(:rec_chunk_layer_ffn_out,   (n_tokens * ffn_down_qw.out_dim).to_i64 * sizeof(Float32))
          out_buf       = Scratch.get(:rec_chunk_layer_out,       inp.size.to_i64 * sizeof(Float32))

          inp_buf.write(inp)
          norm_w_buf.write(attn_norm)
          conv_w_buf.write(ssm_conv1d)
          dt_bias_buf.write(ssm_dt_bias)
          ssm_a_buf.write(ssm_a)
          ssm_norm_buf.write(ssm_norm)
          post_w_buf.write(post_attention_norm)

          qkv_w_buf, qkv_w_off = weight_slot(attn_qkv_qw)
          gate_w_buf, gate_w_off = weight_slot(attn_gate_qw)
          alpha_w_buf, alpha_w_off = weight_slot(ssm_alpha_qw)
          beta_w_buf, beta_w_off = weight_slot(ssm_beta_qw)
          out_w_buf, out_w_off = weight_slot(ssm_out_qw)
          ffn_gate_w_buf, ffn_gate_w_off = weight_slot(ffn_gate_qw)
          ffn_up_w_buf, ffn_up_w_off = weight_slot(ffn_up_qw)
          ffn_down_w_buf, ffn_down_w_off = weight_slot(ffn_down_qw)

          t0 = Time.instant if Profile.enabled?
          cmd = ML::Metal::CommandBuffer.new

          norm_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_rmsnorm_rows(norm_enc, inp_buf, norm_w_buf, cur_buf, hidden_dim, n_tokens, eps)
          norm_enc.end_encoding

          proj_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_matmul(proj_enc, qkv_pipe.not_nil!,   attn_qkv_qw,  cur_buf, qkv_buf,   qkv_w_buf,   qkv_w_off,   attn_qkv_qw.in_dim,  attn_qkv_qw.out_dim,  n_tokens)
          encode_matmul(proj_enc, gate_pipe.not_nil!,  attn_gate_qw, cur_buf, z_buf,     gate_w_buf,  gate_w_off,  attn_gate_qw.in_dim, attn_gate_qw.out_dim, n_tokens)
          encode_matmul(proj_enc, alpha_pipe.not_nil!, ssm_alpha_qw, cur_buf, alpha_buf, alpha_w_buf, alpha_w_off, ssm_alpha_qw.in_dim, ssm_alpha_qw.out_dim, n_tokens)
          encode_matmul(proj_enc, beta_pipe.not_nil!,  ssm_beta_qw,  cur_buf, beta_buf,  beta_w_buf,  beta_w_off,  ssm_beta_qw.in_dim,  ssm_beta_qw.out_dim,  n_tokens)
          proj_enc.end_encoding

          conv_enc = ML::Metal::ComputeEncoder.new(cmd)
          conv_enc.set_pipeline(recurrent_conv_shift_chunk_pipeline)
          conv_enc.set_buffer(conv_state_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          conv_enc.set_buffer(qkv_buf,        1)
          conv_enc.set_buffer(conv_w_buf,     2)
          conv_enc.set_buffer(q_buf,          3, ML::Metal::BufferAccess::Write)
          conv_enc.set_buffer(k_buf,          4, ML::Metal::BufferAccess::Write)
          conv_enc.set_buffer(v_buf,          5, ML::Metal::BufferAccess::Write)
          conv_enc.set_value(h_k.to_u32,      6)
          conv_enc.set_value(h_v.to_u32,      7)
          conv_enc.set_value(s.to_u32,        8)
          conv_enc.set_value(conv_k.to_u32,   9)
          conv_enc.set_value(n_tokens.to_u32, 10)
          conv_enc.dispatch_1d(qkv_dim, 256)
          conv_enc.end_encoding

          qnorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          qnorm_enc.set_pipeline(l2_heads_chunk_pipeline)
          qnorm_enc.set_buffer(q_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          qnorm_enc.set_value(h_k.to_u32, 1)
          qnorm_enc.set_value(s.to_u32,   2)
          qnorm_enc.set_value(eps,        3)
          qnorm_enc.dispatch_threadgroups({h_k, n_tokens, 1}, {32, 1, 1})
          qnorm_enc.end_encoding

          knorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          knorm_enc.set_pipeline(l2_heads_chunk_pipeline)
          knorm_enc.set_buffer(k_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          knorm_enc.set_value(h_k.to_u32, 1)
          knorm_enc.set_value(s.to_u32,   2)
          knorm_enc.set_value(eps,        3)
          knorm_enc.dispatch_threadgroups({h_k, n_tokens, 1}, {32, 1, 1})
          knorm_enc.end_encoding

          ab_enc = ML::Metal::ComputeEncoder.new(cmd)
          ab_enc.set_pipeline(recurrent_ab_chunk_pipeline)
          ab_enc.set_buffer(alpha_buf,   0)
          ab_enc.set_buffer(beta_buf,    1, ML::Metal::BufferAccess::ReadWrite)
          ab_enc.set_buffer(dt_bias_buf, 2)
          ab_enc.set_buffer(ssm_a_buf,   3)
          ab_enc.set_buffer(g_buf,       4, ML::Metal::BufferAccess::Write)
          ab_enc.set_value(h_v.to_u32,       5)
          ab_enc.set_value(n_tokens.to_u32,  6)
          ab_enc.dispatch_1d(n_tokens * h_v, 64)
          ab_enc.end_encoding

          dn_enc = ML::Metal::ComputeEncoder.new(cmd)
          use_dn_rowwise = dn_chunk_rowwise_enabled?(s)
          dn_enc.set_pipeline(use_dn_rowwise ? dn128_chunk_rowwise_pipeline : dn128_chunk_fused_pipeline)
          dn_enc.set_buffer(ssm_state_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          dn_enc.set_buffer(q_buf,         1)
          dn_enc.set_buffer(k_buf,         2)
          dn_enc.set_buffer(v_buf,         3)
          dn_enc.set_buffer(g_buf,         4)
          dn_enc.set_buffer(beta_buf,      5)
          dn_enc.set_buffer(attn_mid_buf,  6, ML::Metal::BufferAccess::Write)
          dn_enc.set_value(h_k.to_u32,     7)
          dn_enc.set_value(h_v.to_u32,     8)
          dn_enc.set_value(s.to_u32,       9)
          dn_enc.set_value(scale,         10)
          dn_enc.set_value(n_tokens.to_u32, 11)
          if use_dn_rowwise
            dn_enc.dispatch_threadgroups({(s + 3) // 4, h_v, 1}, {32, 4, 1})
          else
            dn_enc.dispatch_threadgroups({h_v, 1, 1}, {128, 1, 1})
          end
          dn_enc.end_encoding

          post_enc = ML::Metal::ComputeEncoder.new(cmd)
          post_enc.set_pipeline(dn_post_chunk_pipeline)
          post_enc.set_buffer(attn_mid_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          post_enc.set_buffer(z_buf,        1)
          post_enc.set_buffer(ssm_norm_buf, 2)
          post_enc.set_value(h_v.to_u32,    3)
          post_enc.set_value(s.to_u32,      4)
          post_enc.set_value(eps,           5)
          post_enc.set_value(n_tokens.to_u32, 6)
          post_enc.dispatch_threadgroups({h_v, n_tokens, 1}, {32, 1, 1})
          post_enc.end_encoding

          out_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_matmul(out_enc, out_pipe.not_nil!, ssm_out_qw, attn_mid_buf, attn_out_buf, out_w_buf, out_w_off, ssm_out_qw.in_dim, ssm_out_qw.out_dim, n_tokens)
          out_enc.end_encoding

          addnorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_add_rmsnorm_rows(addnorm_enc, inp_buf, attn_out_buf, post_w_buf, residual_buf, normed_buf, hidden_dim, n_tokens, eps)
          addnorm_enc.end_encoding

          ffn_proj_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_matmul(ffn_proj_enc, ffn_gate_pipe.not_nil!, ffn_gate_qw, normed_buf, ffn_gate_buf, ffn_gate_w_buf, ffn_gate_w_off, ffn_gate_qw.in_dim, ffn_gate_qw.out_dim, n_tokens)
          encode_matmul(ffn_proj_enc, ffn_up_pipe.not_nil!,   ffn_up_qw,   normed_buf, ffn_up_buf,   ffn_up_w_buf,   ffn_up_w_off,   ffn_up_qw.in_dim,   ffn_up_qw.out_dim,   n_tokens)
          ffn_proj_enc.end_encoding

          swiglu_enc = ML::Metal::ComputeEncoder.new(cmd)
          swiglu_enc.set_pipeline(ffn_swiglu_pipeline)
          swiglu_enc.set_buffer(ffn_gate_buf, 0)
          swiglu_enc.set_buffer(ffn_up_buf,   1)
          swiglu_enc.set_buffer(ffn_comb_buf, 2, ML::Metal::BufferAccess::Write)
          swiglu_enc.set_value((n_tokens * ffn_dim).to_u32, 3)
          swiglu_enc.dispatch_1d(n_tokens * ffn_dim, 256)
          swiglu_enc.end_encoding

          fused_down_add = false
          if prefill_ffn_down_add_fused_enabled?
            ffn_down_enc = ML::Metal::ComputeEncoder.new(cmd)
            fused_down_add = encode_matmul_add(ffn_down_enc, ffn_down_pipe.not_nil!, ffn_down_qw, ffn_comb_buf, residual_buf, out_buf, ffn_down_w_buf, ffn_down_w_off, ffn_down_qw.in_dim, ffn_down_qw.out_dim, n_tokens)
            ffn_down_enc.end_encoding
          end

          unless fused_down_add
            ffn_down_enc = ML::Metal::ComputeEncoder.new(cmd)
            encode_matmul(ffn_down_enc, ffn_down_pipe.not_nil!, ffn_down_qw, ffn_comb_buf, ffn_out_buf, ffn_down_w_buf, ffn_down_w_off, ffn_down_qw.in_dim, ffn_down_qw.out_dim, n_tokens)
            ffn_down_enc.end_encoding

            add_enc = ML::Metal::ComputeEncoder.new(cmd)
            add_enc.set_pipeline(add_vec_pipeline)
            add_enc.set_buffer(residual_buf, 0)
            add_enc.set_buffer(ffn_out_buf,  1)
            add_enc.set_buffer(out_buf,      2, ML::Metal::BufferAccess::Write)
            add_enc.set_value((n_tokens * hidden_dim).to_u32, 3)
            add_enc.dispatch_1d(n_tokens * hidden_dim, 256)
            add_enc.end_encoding
          end

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_f32(out_buf, n_tokens * hidden_dim)
          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_dn(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          result
        end

        # GPU-resident run of consecutive recurrent prefill layers.
        #
        # This keeps the token-major hidden matrix on Metal across recurrent
        # layers, removing per-layer hidden readback/upload within runs between
        # full-attention layers. It intentionally reuses the same exact kernels
        # as `recurrent_layer_chunk_project`.
        def self.recurrent_layer_chunk_project_many(inp : Array(Float32),
                                                    conv_state_bufs : Array(ML::MetalBuffer),
                                                    ssm_state_bufs : Array(ML::MetalBuffer),
                                                    layers : Array(Qwen35RecurrentWeights),
                                                    h_k : Int32, h_v : Int32, s : Int32,
                                                    conv_k : Int32,
                                                    n_tokens : Int32,
                                                    eps : Float32,
                                                    profile_label : String = "rec_chunk_many") : Array(Float32)?
          return nil unless n_tokens > 0
          return nil if layers.empty?

          layers.each do |lw|
            qkv_pipe = gemv_pipeline_for(lw.attn_qkv_qw)
            gate_pipe = gemv_pipeline_for(lw.attn_gate_qw)
            alpha_pipe = gemv_pipeline_for(lw.ssm_alpha_qw)
            beta_pipe = gemv_pipeline_for(lw.ssm_beta_qw)
            out_pipe = gemv_pipeline_for(lw.ssm_out_qw)
            ffn_gate_pipe = gemv_pipeline_for(lw.ffn_gate_qw)
            ffn_up_pipe = gemv_pipeline_for(lw.ffn_up_qw)
            ffn_down_pipe = gemv_pipeline_for(lw.ffn_down_qw)
            return nil if qkv_pipe.nil? || gate_pipe.nil? || alpha_pipe.nil? || beta_pipe.nil? ||
                          out_pipe.nil? || ffn_gate_pipe.nil? || ffn_up_pipe.nil? || ffn_down_pipe.nil?
          end

          ML::Metal::Device.init!

          hidden_dim = layers.first.attn_qkv_qw.in_dim
          qkv_dim = 2 * h_k * s + h_v * s
          d_inner = h_v * s
          ffn_dim = layers.first.ffn_gate_qw.out_dim
          scale = (1.0 / Math.sqrt(s.to_f64)).to_f32
          raise "recurrent_layer_chunk_many input size mismatch" unless inp.size == n_tokens * hidden_dim
          raise "recurrent_layer_chunk_many state size mismatch" unless conv_state_bufs.size == layers.size && ssm_state_bufs.size == layers.size

          src_buf = Scratch.get(:rec_chunk_many_hidden_a, inp.size.to_i64 * sizeof(Float32))
          dst_buf = Scratch.get(:rec_chunk_many_hidden_b, inp.size.to_i64 * sizeof(Float32))
          cur_buf = Scratch.get(:rec_chunk_many_cur, inp.size.to_i64 * sizeof(Float32))
          qkv_buf = Scratch.get(:rec_chunk_many_qkv, (n_tokens * qkv_dim).to_i64 * sizeof(Float32))
          qkv_h16_buf = Scratch.get(:rec_chunk_many_qkv_h16, (n_tokens * qkv_dim).to_i64 * 2_i64)
          z_buf = Scratch.get(:rec_chunk_many_z, (n_tokens * d_inner).to_i64 * sizeof(Float32))
          alpha_buf = Scratch.get(:rec_chunk_many_alpha, (n_tokens * h_v).to_i64 * sizeof(Float32))
          beta_buf = Scratch.get(:rec_chunk_many_beta, (n_tokens * h_v).to_i64 * sizeof(Float32))
          g_buf = Scratch.get(:rec_chunk_many_g, (n_tokens * h_v).to_i64 * sizeof(Float32))
          q_buf = Scratch.get(:rec_chunk_many_q, (n_tokens * h_k * s).to_i64 * sizeof(Float32))
          k_buf = Scratch.get(:rec_chunk_many_k, (n_tokens * h_k * s).to_i64 * sizeof(Float32))
          v_buf = Scratch.get(:rec_chunk_many_v, (n_tokens * d_inner).to_i64 * sizeof(Float32))
          attn_mid_buf = Scratch.get(:rec_chunk_many_mid, (n_tokens * d_inner).to_i64 * sizeof(Float32))
          attn_out_buf = Scratch.get(:rec_chunk_many_attn_out, (n_tokens * hidden_dim).to_i64 * sizeof(Float32))
          residual_buf = Scratch.get(:rec_chunk_many_residual, inp.size.to_i64 * sizeof(Float32))
          normed_buf = Scratch.get(:rec_chunk_many_normed, inp.size.to_i64 * sizeof(Float32))
          ffn_gate_buf = Scratch.get(:rec_chunk_many_ffn_gate, (n_tokens * ffn_dim).to_i64 * sizeof(Float32))
          ffn_up_buf = Scratch.get(:rec_chunk_many_ffn_up, (n_tokens * ffn_dim).to_i64 * sizeof(Float32))
          ffn_comb_buf = Scratch.get(:rec_chunk_many_ffn_comb, (n_tokens * ffn_dim).to_i64 * sizeof(Float32))
          ffn_out_buf = Scratch.get(:rec_chunk_many_ffn_out, (n_tokens * hidden_dim).to_i64 * sizeof(Float32))

          src_buf.write(inp)

          t0 = Time.instant if Profile.enabled?
          cmd = ML::Metal::CommandBuffer.new

          layers.each_with_index do |lw, local_i|
            tag = "rec_chunk_many_#{local_i}_#{lw.attn_qkv_qw.raw.to_unsafe.address}"
            norm_w_buf = Scratch.get("#{tag}_norm_w", lw.attn_norm.size.to_i64 * sizeof(Float32))
            conv_w_buf = Scratch.get("#{tag}_conv_w", lw.ssm_conv1d.size.to_i64 * sizeof(Float32))
            dt_bias_buf = Scratch.get("#{tag}_dt_bias", lw.ssm_dt_bias.size.to_i64 * sizeof(Float32))
            ssm_a_buf = Scratch.get("#{tag}_ssm_a", lw.ssm_a.size.to_i64 * sizeof(Float32))
            ssm_norm_buf = Scratch.get("#{tag}_ssm_norm", lw.ssm_norm.size.to_i64 * sizeof(Float32))
            post_w_buf = Scratch.get("#{tag}_post_w", lw.post_attention_norm.size.to_i64 * sizeof(Float32))
            ConstCache.write_once("#{tag}_norm_w", norm_w_buf, lw.attn_norm)
            ConstCache.write_once("#{tag}_conv_w", conv_w_buf, lw.ssm_conv1d)
            ConstCache.write_once("#{tag}_dt_bias", dt_bias_buf, lw.ssm_dt_bias)
            ConstCache.write_once("#{tag}_ssm_a", ssm_a_buf, lw.ssm_a)
            ConstCache.write_once("#{tag}_ssm_norm", ssm_norm_buf, lw.ssm_norm)
            ConstCache.write_once("#{tag}_post_w", post_w_buf, lw.post_attention_norm)

            qkv_w_buf, qkv_w_off = weight_slot(lw.attn_qkv_qw)
            gate_w_buf, gate_w_off = weight_slot(lw.attn_gate_qw)
            alpha_w_buf, alpha_w_off = weight_slot(lw.ssm_alpha_qw)
            beta_w_buf, beta_w_off = weight_slot(lw.ssm_beta_qw)
            out_w_buf, out_w_off = weight_slot(lw.ssm_out_qw)
            ffn_gate_w_buf, ffn_gate_w_off = weight_slot(lw.ffn_gate_qw)
            ffn_up_w_buf, ffn_up_w_off = weight_slot(lw.ffn_up_qw)
            ffn_down_w_buf, ffn_down_w_off = weight_slot(lw.ffn_down_qw)

            norm_enc = ML::Metal::ComputeEncoder.new(cmd)
            encode_rmsnorm_rows(norm_enc, src_buf, norm_w_buf, cur_buf, hidden_dim, n_tokens, eps)
            norm_enc.end_encoding

            Profile.trace("prefill.rec.proj") do
              proj_enc = ML::Metal::ComputeEncoder.new(cmd)
              qkv_h16 = q5_qkv_h16_conv_enabled? && q56_batch_gemm_enabled? && lw.attn_qkv_qw.type.q5_k? && n_tokens > GEMM_BATCH_THRESHOLD
              shared_h16 = rec_proj_shared_h16_enabled? && qkv_h16 && q4_h16_gemm_enabled? &&
                           lw.attn_gate_qw.type.q4_k? && n_tokens > GEMM_BATCH_THRESHOLD
              if shared_h16
                proj_x16_buf = Scratch.get(:rec_chunk_layer_proj_x16, (n_tokens * lw.attn_qkv_qw.in_dim).to_i64 * 2_i64)
                Profile.bump_conversion("f32_to_f16 rec_proj_shared_input #{lw.attn_qkv_qw.in_dim} b#{n_tokens}", (n_tokens * lw.attn_qkv_qw.in_dim).to_i64 * 6_i64)
                proj_enc.set_pipeline(f32_to_f16_pipeline)
                proj_enc.set_buffer(cur_buf, 0)
                proj_enc.set_buffer(proj_x16_buf, 1, ML::Metal::BufferAccess::Write)
                proj_enc.set_value((n_tokens * lw.attn_qkv_qw.in_dim).to_u32, 2)
                proj_enc.dispatch_1d(n_tokens * lw.attn_qkv_qw.in_dim, 256)

                Profile.bump_matmul_shape("q5_h16_gemm #{lw.attn_qkv_qw.type.name} #{lw.attn_qkv_qw.in_dim}x#{lw.attn_qkv_qw.out_dim} b#{n_tokens}", lw.attn_qkv_qw.raw.size.to_i64)
                encode_q56k_gemm_h16_from_h16(proj_enc, mm5_pipeline, proj_x16_buf, qkv_h16_buf, qkv_w_buf, qkv_w_off, lw.attn_qkv_qw.in_dim, lw.attn_qkv_qw.out_dim, n_tokens)
                Profile.bump_matmul_shape("q4_h16_gemm #{lw.attn_gate_qw.type.name} #{lw.attn_gate_qw.in_dim}x#{lw.attn_gate_qw.out_dim} b#{n_tokens}", lw.attn_gate_qw.raw.size.to_i64)
                encode_q4k_gemm_h16_from_h16(proj_enc, proj_x16_buf, z_buf, gate_w_buf, gate_w_off, lw.attn_gate_qw.in_dim, lw.attn_gate_qw.out_dim, n_tokens)
              elsif qkv_h16
                Profile.bump_matmul_shape("q5_h16_gemm #{lw.attn_qkv_qw.type.name} #{lw.attn_qkv_qw.in_dim}x#{lw.attn_qkv_qw.out_dim} b#{n_tokens}", lw.attn_qkv_qw.raw.size.to_i64)
                encode_q56k_gemm_h16(proj_enc, mm5_pipeline, cur_buf, qkv_h16_buf, qkv_w_buf, qkv_w_off, lw.attn_qkv_qw.in_dim, lw.attn_qkv_qw.out_dim, n_tokens)
                encode_matmul(proj_enc, gemv_pipeline_for(lw.attn_gate_qw).not_nil!, lw.attn_gate_qw, cur_buf, z_buf, gate_w_buf, gate_w_off, lw.attn_gate_qw.in_dim, lw.attn_gate_qw.out_dim, n_tokens)
              else
                encode_matmul(proj_enc, gemv_pipeline_for(lw.attn_qkv_qw).not_nil!, lw.attn_qkv_qw, cur_buf, qkv_buf, qkv_w_buf, qkv_w_off, lw.attn_qkv_qw.in_dim, lw.attn_qkv_qw.out_dim, n_tokens)
                encode_matmul(proj_enc, gemv_pipeline_for(lw.attn_gate_qw).not_nil!, lw.attn_gate_qw, cur_buf, z_buf, gate_w_buf, gate_w_off, lw.attn_gate_qw.in_dim, lw.attn_gate_qw.out_dim, n_tokens)
              end
              encode_matmul(proj_enc, gemv_pipeline_for(lw.ssm_alpha_qw).not_nil!, lw.ssm_alpha_qw, cur_buf, alpha_buf, alpha_w_buf, alpha_w_off, lw.ssm_alpha_qw.in_dim, lw.ssm_alpha_qw.out_dim, n_tokens)
              encode_matmul(proj_enc, gemv_pipeline_for(lw.ssm_beta_qw).not_nil!, lw.ssm_beta_qw, cur_buf, beta_buf, beta_w_buf, beta_w_off, lw.ssm_beta_qw.in_dim, lw.ssm_beta_qw.out_dim, n_tokens)
              proj_enc.end_encoding
            end

            conv_enc = ML::Metal::ComputeEncoder.new(cmd)
            qkv_h16 = q5_qkv_h16_conv_enabled? && q56_batch_gemm_enabled? && lw.attn_qkv_qw.type.q5_k? && n_tokens > GEMM_BATCH_THRESHOLD
            conv_enc.set_pipeline(qkv_h16 ? recurrent_conv_shift_chunk_h16_pipeline : recurrent_conv_shift_chunk_pipeline)
            conv_enc.set_buffer(conv_state_bufs[local_i], 0, ML::Metal::BufferAccess::ReadWrite)
            conv_enc.set_buffer(qkv_h16 ? qkv_h16_buf : qkv_buf, 1)
            conv_enc.set_buffer(conv_w_buf, 2)
            conv_enc.set_buffer(q_buf, 3, ML::Metal::BufferAccess::Write)
            conv_enc.set_buffer(k_buf, 4, ML::Metal::BufferAccess::Write)
            conv_enc.set_buffer(v_buf, 5, ML::Metal::BufferAccess::Write)
            conv_enc.set_value(h_k.to_u32, 6)
            conv_enc.set_value(h_v.to_u32, 7)
            conv_enc.set_value(s.to_u32, 8)
            conv_enc.set_value(conv_k.to_u32, 9)
            conv_enc.set_value(n_tokens.to_u32, 10)
            conv_enc.dispatch_1d(qkv_dim, 256)
            conv_enc.end_encoding

            qnorm_enc = ML::Metal::ComputeEncoder.new(cmd)
            qnorm_enc.set_pipeline(l2_heads_chunk_pipeline)
            qnorm_enc.set_buffer(q_buf, 0, ML::Metal::BufferAccess::ReadWrite)
            qnorm_enc.set_value(h_k.to_u32, 1)
            qnorm_enc.set_value(s.to_u32, 2)
            qnorm_enc.set_value(eps, 3)
            qnorm_enc.dispatch_threadgroups({h_k, n_tokens, 1}, {32, 1, 1})
            qnorm_enc.end_encoding

            knorm_enc = ML::Metal::ComputeEncoder.new(cmd)
            knorm_enc.set_pipeline(l2_heads_chunk_pipeline)
            knorm_enc.set_buffer(k_buf, 0, ML::Metal::BufferAccess::ReadWrite)
            knorm_enc.set_value(h_k.to_u32, 1)
            knorm_enc.set_value(s.to_u32, 2)
            knorm_enc.set_value(eps, 3)
            knorm_enc.dispatch_threadgroups({h_k, n_tokens, 1}, {32, 1, 1})
            knorm_enc.end_encoding

            ab_enc = ML::Metal::ComputeEncoder.new(cmd)
            ab_enc.set_pipeline(recurrent_ab_chunk_pipeline)
            ab_enc.set_buffer(alpha_buf, 0)
            ab_enc.set_buffer(beta_buf, 1, ML::Metal::BufferAccess::ReadWrite)
            ab_enc.set_buffer(dt_bias_buf, 2)
            ab_enc.set_buffer(ssm_a_buf, 3)
            ab_enc.set_buffer(g_buf, 4, ML::Metal::BufferAccess::Write)
            ab_enc.set_value(h_v.to_u32, 5)
            ab_enc.set_value(n_tokens.to_u32, 6)
            ab_enc.dispatch_1d(n_tokens * h_v, 64)
            ab_enc.end_encoding

            dn_enc = ML::Metal::ComputeEncoder.new(cmd)
            use_dn_rowwise = dn_chunk_rowwise_enabled?(s)
            dn_enc.set_pipeline(use_dn_rowwise ? dn128_chunk_rowwise_pipeline : dn128_chunk_fused_pipeline)
            dn_enc.set_buffer(ssm_state_bufs[local_i], 0, ML::Metal::BufferAccess::ReadWrite)
            dn_enc.set_buffer(q_buf, 1)
            dn_enc.set_buffer(k_buf, 2)
            dn_enc.set_buffer(v_buf, 3)
            dn_enc.set_buffer(g_buf, 4)
            dn_enc.set_buffer(beta_buf, 5)
            dn_enc.set_buffer(attn_mid_buf, 6, ML::Metal::BufferAccess::Write)
            dn_enc.set_value(h_k.to_u32, 7)
            dn_enc.set_value(h_v.to_u32, 8)
            dn_enc.set_value(s.to_u32, 9)
            dn_enc.set_value(scale, 10)
            dn_enc.set_value(n_tokens.to_u32, 11)
            if use_dn_rowwise
              dn_enc.dispatch_threadgroups({(s + 3) // 4, h_v, 1}, {32, 4, 1})
            else
              dn_enc.dispatch_threadgroups({h_v, 1, 1}, {128, 1, 1})
            end
            dn_enc.end_encoding

            post_enc = ML::Metal::ComputeEncoder.new(cmd)
            post_enc.set_pipeline(dn_post_chunk_pipeline)
            post_enc.set_buffer(attn_mid_buf, 0, ML::Metal::BufferAccess::ReadWrite)
            post_enc.set_buffer(z_buf, 1)
            post_enc.set_buffer(ssm_norm_buf, 2)
            post_enc.set_value(h_v.to_u32, 3)
            post_enc.set_value(s.to_u32, 4)
            post_enc.set_value(eps, 5)
            post_enc.set_value(n_tokens.to_u32, 6)
            post_enc.dispatch_threadgroups({h_v, n_tokens, 1}, {32, 1, 1})
            post_enc.end_encoding

            Profile.trace("prefill.rec.o_proj") do
              out_enc = ML::Metal::ComputeEncoder.new(cmd)
              encode_matmul(out_enc, gemv_pipeline_for(lw.ssm_out_qw).not_nil!, lw.ssm_out_qw, attn_mid_buf, attn_out_buf, out_w_buf, out_w_off, lw.ssm_out_qw.in_dim, lw.ssm_out_qw.out_dim, n_tokens)
              out_enc.end_encoding
            end

            addnorm_enc = ML::Metal::ComputeEncoder.new(cmd)
            encode_add_rmsnorm_rows(addnorm_enc, src_buf, attn_out_buf, post_w_buf, residual_buf, normed_buf, hidden_dim, n_tokens, eps)
            addnorm_enc.end_encoding

            Profile.trace("prefill.rec.ffn_upgate") do
              ffn_proj_enc = ML::Metal::ComputeEncoder.new(cmd)
              pair_q4 = q4_pair_h16_gemm_candidate?(lw.ffn_gate_qw, lw.ffn_up_qw, n_tokens)
              if pair_q4
                Profile.bump_matmul_shape("q4_h16_gemm #{lw.ffn_gate_qw.type.name} #{lw.ffn_gate_qw.in_dim}x#{lw.ffn_gate_qw.out_dim} b#{n_tokens}", lw.ffn_gate_qw.raw.size.to_i64)
                Profile.bump_matmul_shape("q4_h16_gemm #{lw.ffn_up_qw.type.name} #{lw.ffn_up_qw.in_dim}x#{lw.ffn_up_qw.out_dim} b#{n_tokens}", lw.ffn_up_qw.raw.size.to_i64)
                encode_q4k_gemm_h16_pair(ffn_proj_enc, normed_buf, ffn_gate_buf, ffn_up_buf, ffn_gate_w_buf, ffn_gate_w_off, ffn_up_w_buf, ffn_up_w_off, lw.ffn_gate_qw.in_dim, lw.ffn_gate_qw.out_dim, n_tokens)
              else
                encode_matmul(ffn_proj_enc, gemv_pipeline_for(lw.ffn_gate_qw).not_nil!, lw.ffn_gate_qw, normed_buf, ffn_gate_buf, ffn_gate_w_buf, ffn_gate_w_off, lw.ffn_gate_qw.in_dim, lw.ffn_gate_qw.out_dim, n_tokens)
                encode_matmul(ffn_proj_enc, gemv_pipeline_for(lw.ffn_up_qw).not_nil!, lw.ffn_up_qw, normed_buf, ffn_up_buf, ffn_up_w_buf, ffn_up_w_off, lw.ffn_up_qw.in_dim, lw.ffn_up_qw.out_dim, n_tokens)
              end
              ffn_proj_enc.end_encoding
            end

            swiglu_enc = ML::Metal::ComputeEncoder.new(cmd)
            ffn_act_buf = swiglu_inplace_enabled? ? ffn_up_buf : ffn_comb_buf
            swiglu_enc.set_pipeline(ffn_swiglu_pipeline)
            swiglu_enc.set_buffer(ffn_gate_buf, 0)
            swiglu_enc.set_buffer(ffn_up_buf, 1)
            swiglu_enc.set_buffer(ffn_act_buf, 2, ML::Metal::BufferAccess::Write)
            swiglu_enc.set_value((n_tokens * ffn_dim).to_u32, 3)
            swiglu_enc.dispatch_1d(n_tokens * ffn_dim, 256)
            swiglu_enc.end_encoding

            fused_down_add = false
            if prefill_ffn_down_add_fused_enabled?
              Profile.trace("prefill.rec.ffn_down_add") do
                ffn_down_enc = ML::Metal::ComputeEncoder.new(cmd)
                fused_down_add = encode_matmul_add(ffn_down_enc, gemv_pipeline_for(lw.ffn_down_qw).not_nil!, lw.ffn_down_qw, ffn_act_buf, residual_buf, dst_buf, ffn_down_w_buf, ffn_down_w_off, lw.ffn_down_qw.in_dim, lw.ffn_down_qw.out_dim, n_tokens)
                ffn_down_enc.end_encoding
              end
            end

            unless fused_down_add
              Profile.trace("prefill.rec.ffn_down") do
                ffn_down_enc = ML::Metal::ComputeEncoder.new(cmd)
                encode_matmul(ffn_down_enc, gemv_pipeline_for(lw.ffn_down_qw).not_nil!, lw.ffn_down_qw, ffn_act_buf, ffn_out_buf, ffn_down_w_buf, ffn_down_w_off, lw.ffn_down_qw.in_dim, lw.ffn_down_qw.out_dim, n_tokens)
                ffn_down_enc.end_encoding
              end

              add_enc = ML::Metal::ComputeEncoder.new(cmd)
              add_enc.set_pipeline(add_vec_pipeline)
              add_enc.set_buffer(residual_buf, 0)
              add_enc.set_buffer(ffn_out_buf, 1)
              add_enc.set_buffer(dst_buf, 2, ML::Metal::BufferAccess::Write)
              add_enc.set_value((n_tokens * hidden_dim).to_u32, 3)
              add_enc.dispatch_1d(n_tokens * hidden_dim, 256)
              add_enc.end_encoding
            end

            src_buf, dst_buf = dst_buf, src_buf
          end

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_f32(src_buf, n_tokens * hidden_dim)
          if Profile.enabled?
            t_read = Time.instant
            encode_ns = (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64
            wait_ns = (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64
            read_ns = (t_read - t_wait.not_nil!).total_nanoseconds.to_i64
            Profile.bump_dn(encode_ns, wait_ns, read_ns)
            Profile.bump_group(profile_label, encode_ns, wait_ns, read_ns)
          end
          result
        end

        # Full-attention route with GPU prep:
        #   q/k/v projections -> split q+gate -> q/k RMSNorm -> RoPE ->
        #   KV write -> attention -> output projection, all on GPU.
        def self.full_attn_project(x : Array(Float32),
                                   q_qw : QuantWeight,
                                   k_qw : QuantWeight,
                                   v_qw : QuantWeight,
                                   q_norm : Array(Float32),
                                   k_norm : Array(Float32),
                                   out_qw : QuantWeight,
                                   k_cache_buf : ML::MetalBuffer,
                                   v_cache_buf : ML::MetalBuffer,
                                   pos : Int32,
                                   n_head : Int32,
                                   n_head_kv : Int32,
                                   head_dim : Int32,
                                   rope_dim_count : Int32,
                                   heads_per_group : Int32,
                                   rope_freq_base : Float32,
                                   scale : Float32) : Array(Float32)?
          q_pipe = gemv_pipeline_for(q_qw)
          k_pipe = gemv_pipeline_for(k_qw)
          v_pipe = gemv_pipeline_for(v_qw)
          out_pipe = gemv_pipeline_for(out_qw)
          return nil if q_pipe.nil? || k_pipe.nil? || v_pipe.nil? || out_pipe.nil?

          ML::Metal::Device.init!

          q_dim = n_head * head_dim
          kv_dim = n_head_kv * head_dim
          x_buf      = Scratch.get(:fattn_x,      x.size.to_i64 * sizeof(Float32))
          qfull_buf  = Scratch.get(:fattn_qfull,  q_qw.out_dim.to_i64 * sizeof(Float32))
          q_buf      = Scratch.get(:fattn_q,      q_dim.to_i64 * sizeof(Float32))
          gate_buf   = Scratch.get(:fattn_gate,   q_dim.to_i64 * sizeof(Float32))
          k_buf      = Scratch.get(:fattn_k,      kv_dim.to_i64 * sizeof(Float32))
          v_buf      = Scratch.get(:fattn_v,      kv_dim.to_i64 * sizeof(Float32))
          attn_buf   = Scratch.get(:fattn_attn,   q_dim.to_i64 * sizeof(Float32))
          out_buf    = Scratch.get(:fattn_out,    out_qw.out_dim.to_i64 * sizeof(Float32))
          qnorm_buf  = Scratch.get(:fattn_qnorm,  q_norm.size.to_i64 * sizeof(Float32))
          knorm_buf  = Scratch.get(:fattn_knorm,  k_norm.size.to_i64 * sizeof(Float32))
          x_buf.write(x)
          qnorm_buf.write(q_norm)
          knorm_buf.write(k_norm)

          q_w_buf, q_w_off = if slot = mmap_slot_for(q_qw.raw)
                               slot
                             else
                               {q_qw.fallback_metal_buffer, 0_i64}
                             end
          k_w_buf, k_w_off = if slot = mmap_slot_for(k_qw.raw)
                               slot
                             else
                               {k_qw.fallback_metal_buffer, 0_i64}
                             end
          v_w_buf, v_w_off = if slot = mmap_slot_for(v_qw.raw)
                               slot
                             else
                               {v_qw.fallback_metal_buffer, 0_i64}
                             end
          out_w_buf, out_w_off = if slot = mmap_slot_for(out_qw.raw)
                                   slot
                                 else
                                   {out_qw.fallback_metal_buffer, 0_i64}
                                 end

          t0 = Time.instant if Profile.enabled?
          cmd = ML::Metal::CommandBuffer.new

          proj_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(proj_enc, q_pipe.not_nil!, x_buf, qfull_buf, q_w_buf, q_w_off, q_qw.in_dim, q_qw.out_dim)
          if q8_kv_dual_gemv_candidate?(k_qw, v_qw)
            encode_gemv_q8_dual(proj_enc, x_buf, k_buf, v_buf,
              k_w_buf, k_w_off, v_w_buf, v_w_off, k_qw.in_dim, k_qw.out_dim)
          else
            encode_gemv(proj_enc, k_pipe.not_nil!, x_buf, k_buf,     k_w_buf, k_w_off, k_qw.in_dim, k_qw.out_dim)
            encode_gemv(proj_enc, v_pipe.not_nil!, x_buf, v_buf,     v_w_buf, v_w_off, v_qw.in_dim, v_qw.out_dim)
          end
          proj_enc.end_encoding

          split_enc = ML::Metal::ComputeEncoder.new(cmd)
          split_enc.set_pipeline(split_qgate_pipeline)
          split_enc.set_buffer(qfull_buf, 0)
          split_enc.set_buffer(q_buf,     1, ML::Metal::BufferAccess::Write)
          split_enc.set_buffer(gate_buf,  2, ML::Metal::BufferAccess::Write)
          split_enc.set_value(n_head.to_u32,   3)
          split_enc.set_value(head_dim.to_u32, 4)
          split_enc.dispatch_1d(q_dim, 256)
          split_enc.end_encoding

          qnorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          qnorm_enc.set_pipeline(rmsnorm_heads_pipeline)
          qnorm_enc.set_buffer(q_buf,     0, ML::Metal::BufferAccess::ReadWrite)
          qnorm_enc.set_buffer(qnorm_buf, 1)
          qnorm_enc.set_value(head_dim.to_u32, 2)
          qnorm_enc.set_value(1.0e-6_f32, 3)
          qnorm_enc.dispatch_threadgroups({n_head, 1, 1}, {32, 1, 1})
          qnorm_enc.end_encoding

          knorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          knorm_enc.set_pipeline(rmsnorm_heads_pipeline)
          knorm_enc.set_buffer(k_buf,     0, ML::Metal::BufferAccess::ReadWrite)
          knorm_enc.set_buffer(knorm_buf, 1)
          knorm_enc.set_value(head_dim.to_u32, 2)
          knorm_enc.set_value(1.0e-6_f32, 3)
          knorm_enc.dispatch_threadgroups({n_head_kv, 1, 1}, {32, 1, 1})
          knorm_enc.end_encoding

          qrope_enc = ML::Metal::ComputeEncoder.new(cmd)
          qrope_enc.set_pipeline(rope_partial_pipeline)
          qrope_enc.set_buffer(q_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          qrope_enc.set_value(head_dim.to_u32,       1)
          qrope_enc.set_value(rope_dim_count.to_u32, 2)
          qrope_enc.set_value(pos.to_u32,            3)
          qrope_enc.set_value(rope_freq_base,        4)
          qrope_enc.dispatch_threadgroups({n_head, 1, 1}, {32, 1, 1})
          qrope_enc.end_encoding

          krope_enc = ML::Metal::ComputeEncoder.new(cmd)
          krope_enc.set_pipeline(rope_partial_pipeline)
          krope_enc.set_buffer(k_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          krope_enc.set_value(head_dim.to_u32,       1)
          krope_enc.set_value(rope_dim_count.to_u32, 2)
          krope_enc.set_value(pos.to_u32,            3)
          krope_enc.set_value(rope_freq_base,        4)
          krope_enc.dispatch_threadgroups({n_head_kv, 1, 1}, {32, 1, 1})
          krope_enc.end_encoding

          kvwrite_enc = ML::Metal::ComputeEncoder.new(cmd)
          kvwrite_enc.set_pipeline(kv_write_pipeline)
          kvwrite_enc.set_buffer(k_buf,       0)
          kvwrite_enc.set_buffer(v_buf,       1)
          kvwrite_enc.set_buffer(k_cache_buf, 2, ML::Metal::BufferAccess::ReadWrite)
          kvwrite_enc.set_buffer(v_cache_buf, 3, ML::Metal::BufferAccess::ReadWrite)
          kvwrite_enc.set_value((pos * kv_dim).to_u32, 4)
          kvwrite_enc.set_value(kv_dim.to_u32,         5)
          kvwrite_enc.dispatch_1d(kv_dim, 256)
          kvwrite_enc.end_encoding

          attn_enc = ML::Metal::ComputeEncoder.new(cmd)
          attn_enc.set_pipeline(attn_pipeline)
          attn_enc.set_buffer(q_buf,         0)
          attn_enc.set_buffer(gate_buf,      1)
          attn_enc.set_buffer(k_cache_buf,   2)
          attn_enc.set_buffer(v_cache_buf,   3)
          attn_enc.set_buffer(attn_buf,      4, ML::Metal::BufferAccess::Write)
          attn_enc.set_value((pos + 1).to_u32,       5)
          attn_enc.set_value(n_head.to_u32,          6)
          attn_enc.set_value(n_head_kv.to_u32,       7)
          attn_enc.set_value(head_dim.to_u32,        8)
          attn_enc.set_value(heads_per_group.to_u32, 9)
          attn_enc.set_value(scale,                 10)
          attn_enc.dispatch_threadgroups({n_head, 1, 1}, {32, 1, 1})
          attn_enc.end_encoding

          outproj_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(outproj_enc, out_pipe.not_nil!, attn_buf, out_buf, out_w_buf, out_w_off, out_qw.in_dim, out_qw.out_dim)
          outproj_enc.end_encoding

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_f32(out_buf, out_qw.out_dim)
          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_attn(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          result
        end

        # Fused full-attention layer route:
        #   q/k/v projections -> split/norm/rope -> kv write -> attention ->
        #   output projection -> residual add + post-attn RMSNorm ->
        #   SwiGLU FFN -> final residual add
        # in one command buffer, with only the final layer output read back.
        def self.full_attn_layer_project(inp : Array(Float32),
                                         cur : Array(Float32),
                                         q_qw : QuantWeight,
                                         k_qw : QuantWeight,
                                         v_qw : QuantWeight,
                                         q_norm : Array(Float32),
                                         k_norm : Array(Float32),
                                         out_qw : QuantWeight,
                                         k_cache_buf : ML::MetalBuffer,
                                         v_cache_buf : ML::MetalBuffer,
                                         post_attention_norm : Array(Float32),
                                         ffn_gate_qw : QuantWeight,
                                         ffn_up_qw : QuantWeight,
                                         ffn_down_qw : QuantWeight,
                                         pos : Int32,
                                         n_head : Int32,
                                         n_head_kv : Int32,
                                         head_dim : Int32,
                                         rope_dim_count : Int32,
                                         heads_per_group : Int32,
                                         rope_freq_base : Float32,
                                         eps : Float32,
                                         scale : Float32) : Array(Float32)?
          q_pipe = gemv_pipeline_for(q_qw)
          k_pipe = gemv_pipeline_for(k_qw)
          v_pipe = gemv_pipeline_for(v_qw)
          out_pipe = gemv_pipeline_for(out_qw)
          ffn_gate_pipe = gemv_pipeline_for(ffn_gate_qw)
          ffn_up_pipe = gemv_pipeline_for(ffn_up_qw)
          ffn_down_pipe = gemv_pipeline_for(ffn_down_qw)
          return nil if q_pipe.nil? || k_pipe.nil? || v_pipe.nil? || out_pipe.nil? ||
                        ffn_gate_pipe.nil? || ffn_up_pipe.nil? || ffn_down_pipe.nil?

          ML::Metal::Device.init!

          hidden_dim = inp.size
          q_dim = n_head * head_dim
          kv_dim = n_head_kv * head_dim
          ffn_dim = ffn_gate_qw.out_dim

          inp_buf = Scratch.get(:fsl_inp, hidden_dim.to_i64 * sizeof(Float32))
          cur_buf = Scratch.get(:fsl_cur, cur.size.to_i64 * sizeof(Float32))
          qfull_buf = Scratch.get(:fsl_qfull, q_qw.out_dim.to_i64 * sizeof(Float32))
          q_buf = Scratch.get(:fsl_q, q_dim.to_i64 * sizeof(Float32))
          gate_buf = Scratch.get(:fsl_gate, q_dim.to_i64 * sizeof(Float32))
          k_buf = Scratch.get(:fsl_k, kv_dim.to_i64 * sizeof(Float32))
          v_buf = Scratch.get(:fsl_v, kv_dim.to_i64 * sizeof(Float32))
          attn_buf = Scratch.get(:fsl_attn, q_dim.to_i64 * sizeof(Float32))
          attn_out_buf = Scratch.get(:fsl_attn_out, out_qw.out_dim.to_i64 * sizeof(Float32))
          qnorm_buf = Scratch.get(:fsl_qnorm, q_norm.size.to_i64 * sizeof(Float32))
          knorm_buf = Scratch.get(:fsl_knorm, k_norm.size.to_i64 * sizeof(Float32))
          post_norm_buf = Scratch.get(:fsl_postnorm_w, post_attention_norm.size.to_i64 * sizeof(Float32))
          residual_buf = Scratch.get(:fsl_residual, hidden_dim.to_i64 * sizeof(Float32))
          normed_buf = Scratch.get(:fsl_normed, hidden_dim.to_i64 * sizeof(Float32))
          ffn_gate_buf = Scratch.get(:fsl_ffn_gate, ffn_dim.to_i64 * sizeof(Float32))
          ffn_up_buf = Scratch.get(:fsl_ffn_up, ffn_dim.to_i64 * sizeof(Float32))
          ffn_comb_buf = Scratch.get(:fsl_ffn_comb, ffn_dim.to_i64 * sizeof(Float32))
          ffn_out_buf = Scratch.get(:fsl_ffn_out, ffn_down_qw.out_dim.to_i64 * sizeof(Float32))
          out_buf = Scratch.get(:fsl_out, hidden_dim.to_i64 * sizeof(Float32))

          inp_buf.write(inp)
          cur_buf.write(cur)
          qnorm_buf.write(q_norm)
          knorm_buf.write(k_norm)
          post_norm_buf.write(post_attention_norm)

          q_w_buf, q_w_off = if slot = mmap_slot_for(q_qw.raw)
                               slot
                             else
                               {q_qw.fallback_metal_buffer, 0_i64}
                             end
          k_w_buf, k_w_off = if slot = mmap_slot_for(k_qw.raw)
                               slot
                             else
                               {k_qw.fallback_metal_buffer, 0_i64}
                             end
          v_w_buf, v_w_off = if slot = mmap_slot_for(v_qw.raw)
                               slot
                             else
                               {v_qw.fallback_metal_buffer, 0_i64}
                             end
          out_w_buf, out_w_off = if slot = mmap_slot_for(out_qw.raw)
                                   slot
                                 else
                                   {out_qw.fallback_metal_buffer, 0_i64}
                                 end
          ffn_gate_w_buf, ffn_gate_w_off = if slot = mmap_slot_for(ffn_gate_qw.raw)
                                             slot
                                           else
                                             {ffn_gate_qw.fallback_metal_buffer, 0_i64}
                                           end
          ffn_up_w_buf, ffn_up_w_off = if slot = mmap_slot_for(ffn_up_qw.raw)
                                         slot
                                       else
                                         {ffn_up_qw.fallback_metal_buffer, 0_i64}
                                       end
          ffn_down_w_buf, ffn_down_w_off = if slot = mmap_slot_for(ffn_down_qw.raw)
                                             slot
                                           else
                                             {ffn_down_qw.fallback_metal_buffer, 0_i64}
                                           end

          t0 = Time.instant if Profile.enabled?
          cmd = ML::Metal::CommandBuffer.new

          proj_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(proj_enc, q_pipe.not_nil!, cur_buf, qfull_buf, q_w_buf, q_w_off, q_qw.in_dim, q_qw.out_dim)
          if q8_kv_dual_gemv_candidate?(k_qw, v_qw)
            encode_gemv_q8_dual(proj_enc, cur_buf, k_buf, v_buf,
              k_w_buf, k_w_off, v_w_buf, v_w_off, k_qw.in_dim, k_qw.out_dim)
          else
            encode_gemv(proj_enc, k_pipe.not_nil!, cur_buf, k_buf, k_w_buf, k_w_off, k_qw.in_dim, k_qw.out_dim)
            encode_gemv(proj_enc, v_pipe.not_nil!, cur_buf, v_buf, v_w_buf, v_w_off, v_qw.in_dim, v_qw.out_dim)
          end
          proj_enc.end_encoding

          split_enc = ML::Metal::ComputeEncoder.new(cmd)
          split_enc.set_pipeline(split_qgate_pipeline)
          split_enc.set_buffer(qfull_buf, 0)
          split_enc.set_buffer(q_buf, 1, ML::Metal::BufferAccess::Write)
          split_enc.set_buffer(gate_buf, 2, ML::Metal::BufferAccess::Write)
          split_enc.set_value(n_head.to_u32, 3)
          split_enc.set_value(head_dim.to_u32, 4)
          split_enc.dispatch_1d(q_dim, 256)
          split_enc.end_encoding

          qnorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          qnorm_enc.set_pipeline(rmsnorm_heads_pipeline)
          qnorm_enc.set_buffer(q_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          qnorm_enc.set_buffer(qnorm_buf, 1)
          qnorm_enc.set_value(head_dim.to_u32, 2)
          qnorm_enc.set_value(eps, 3)
          qnorm_enc.dispatch_threadgroups({n_head, 1, 1}, {32, 1, 1})
          qnorm_enc.end_encoding

          knorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          knorm_enc.set_pipeline(rmsnorm_heads_pipeline)
          knorm_enc.set_buffer(k_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          knorm_enc.set_buffer(knorm_buf, 1)
          knorm_enc.set_value(head_dim.to_u32, 2)
          knorm_enc.set_value(eps, 3)
          knorm_enc.dispatch_threadgroups({n_head_kv, 1, 1}, {32, 1, 1})
          knorm_enc.end_encoding

          qrope_enc = ML::Metal::ComputeEncoder.new(cmd)
          qrope_enc.set_pipeline(rope_partial_pipeline)
          qrope_enc.set_buffer(q_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          qrope_enc.set_value(head_dim.to_u32, 1)
          qrope_enc.set_value(rope_dim_count.to_u32, 2)
          qrope_enc.set_value(pos.to_u32, 3)
          qrope_enc.set_value(rope_freq_base, 4)
          qrope_enc.dispatch_threadgroups({n_head, 1, 1}, {32, 1, 1})
          qrope_enc.end_encoding

          krope_enc = ML::Metal::ComputeEncoder.new(cmd)
          krope_enc.set_pipeline(rope_partial_pipeline)
          krope_enc.set_buffer(k_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          krope_enc.set_value(head_dim.to_u32, 1)
          krope_enc.set_value(rope_dim_count.to_u32, 2)
          krope_enc.set_value(pos.to_u32, 3)
          krope_enc.set_value(rope_freq_base, 4)
          krope_enc.dispatch_threadgroups({n_head_kv, 1, 1}, {32, 1, 1})
          krope_enc.end_encoding

          kvwrite_enc = ML::Metal::ComputeEncoder.new(cmd)
          kvwrite_enc.set_pipeline(kv_write_pipeline)
          kvwrite_enc.set_buffer(k_buf, 0)
          kvwrite_enc.set_buffer(v_buf, 1)
          kvwrite_enc.set_buffer(k_cache_buf, 2, ML::Metal::BufferAccess::ReadWrite)
          kvwrite_enc.set_buffer(v_cache_buf, 3, ML::Metal::BufferAccess::ReadWrite)
          kvwrite_enc.set_value((pos * kv_dim).to_u32, 4)
          kvwrite_enc.set_value(kv_dim.to_u32, 5)
          kvwrite_enc.dispatch_1d(kv_dim, 256)
          kvwrite_enc.end_encoding

          attn_enc = ML::Metal::ComputeEncoder.new(cmd)
          attn_enc.set_pipeline(attn_pipeline)
          attn_enc.set_buffer(q_buf, 0)
          attn_enc.set_buffer(gate_buf, 1)
          attn_enc.set_buffer(k_cache_buf, 2)
          attn_enc.set_buffer(v_cache_buf, 3)
          attn_enc.set_buffer(attn_buf, 4, ML::Metal::BufferAccess::Write)
          attn_enc.set_value((pos + 1).to_u32, 5)
          attn_enc.set_value(n_head.to_u32, 6)
          attn_enc.set_value(n_head_kv.to_u32, 7)
          attn_enc.set_value(head_dim.to_u32, 8)
          attn_enc.set_value(heads_per_group.to_u32, 9)
          attn_enc.set_value(scale, 10)
          attn_enc.dispatch_threadgroups({n_head, 1, 1}, {32, 1, 1})
          attn_enc.end_encoding

          outproj_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(outproj_enc, out_pipe.not_nil!, attn_buf, attn_out_buf, out_w_buf, out_w_off, out_qw.in_dim, out_qw.out_dim)
          outproj_enc.end_encoding

          addnorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          addnorm_enc.set_pipeline(add_rmsnorm_pipeline)
          addnorm_enc.set_buffer(inp_buf, 0)
          addnorm_enc.set_buffer(attn_out_buf, 1)
          addnorm_enc.set_buffer(post_norm_buf, 2)
          addnorm_enc.set_buffer(residual_buf, 3, ML::Metal::BufferAccess::Write)
          addnorm_enc.set_buffer(normed_buf, 4, ML::Metal::BufferAccess::Write)
          addnorm_enc.set_value(hidden_dim.to_u32, 5)
          addnorm_enc.set_value(eps, 6)
          addnorm_enc.dispatch_threadgroups({1, 1, 1}, {256, 1, 1})
          addnorm_enc.end_encoding

          ffn_proj_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(ffn_proj_enc, ffn_gate_pipe.not_nil!, normed_buf, ffn_gate_buf, ffn_gate_w_buf, ffn_gate_w_off, ffn_gate_qw.in_dim, ffn_gate_qw.out_dim)
          encode_gemv(ffn_proj_enc, ffn_up_pipe.not_nil!, normed_buf, ffn_up_buf, ffn_up_w_buf, ffn_up_w_off, ffn_up_qw.in_dim, ffn_up_qw.out_dim)
          ffn_proj_enc.end_encoding

          swiglu_enc = ML::Metal::ComputeEncoder.new(cmd)
          swiglu_enc.set_pipeline(ffn_swiglu_pipeline)
          swiglu_enc.set_buffer(ffn_gate_buf, 0)
          swiglu_enc.set_buffer(ffn_up_buf, 1)
          swiglu_enc.set_buffer(ffn_comb_buf, 2, ML::Metal::BufferAccess::Write)
          swiglu_enc.set_value(ffn_dim.to_u32, 3)
          swiglu_enc.dispatch_1d(ffn_dim, 256)
          swiglu_enc.end_encoding

          ffn_down_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(ffn_down_enc, ffn_down_pipe.not_nil!, ffn_comb_buf, ffn_out_buf, ffn_down_w_buf, ffn_down_w_off, ffn_down_qw.in_dim, ffn_down_qw.out_dim)
          ffn_down_enc.end_encoding

          add_enc = ML::Metal::ComputeEncoder.new(cmd)
          add_enc.set_pipeline(add_vec_pipeline)
          add_enc.set_buffer(residual_buf, 0)
          add_enc.set_buffer(ffn_out_buf, 1)
          add_enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          add_enc.set_value(hidden_dim.to_u32, 3)
          add_enc.dispatch_1d(hidden_dim, 256)
          add_enc.end_encoding

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_f32(out_buf, hidden_dim)
          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_attn(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          result
        end

        # Final full-attention prefill specialization.
        #
        # The final decoder layer only needs K/V cache updates for every prompt
        # row; only the last row's hidden state is needed for next-token logits.
        # This route therefore projects K/V for the whole chunk, but computes
        # Q/attention/FFN output only for the final row.
        def self.full_attn_layer_chunk_project_last(inp : Array(Float32),
                                                    q_qw : QuantWeight,
                                                    k_qw : QuantWeight,
                                                    v_qw : QuantWeight,
                                                    attn_norm : Array(Float32),
                                                    q_norm : Array(Float32),
                                                    k_norm : Array(Float32),
                                                    out_qw : QuantWeight,
                                                    k_cache_buf : ML::MetalBuffer,
                                                    v_cache_buf : ML::MetalBuffer,
                                                    post_attention_norm : Array(Float32),
                                                    ffn_gate_qw : QuantWeight,
                                                    ffn_up_qw : QuantWeight,
                                                    ffn_down_qw : QuantWeight,
                                                    start_pos : Int32,
                                                    n_tokens : Int32,
                                                    n_head : Int32,
                                                    n_head_kv : Int32,
                                                    head_dim : Int32,
                                                    rope_dim_count : Int32,
                                                    heads_per_group : Int32,
                                                    rope_freq_base : Float32,
                                                    eps : Float32,
                                                    scale : Float32) : Array(Float32)?
          q_pipe = gemv_pipeline_for(q_qw)
          k_pipe = gemv_pipeline_for(k_qw)
          v_pipe = gemv_pipeline_for(v_qw)
          out_pipe = gemv_pipeline_for(out_qw)
          ffn_gate_pipe = gemv_pipeline_for(ffn_gate_qw)
          ffn_up_pipe = gemv_pipeline_for(ffn_up_qw)
          ffn_down_pipe = gemv_pipeline_for(ffn_down_qw)
          return nil if q_pipe.nil? || k_pipe.nil? || v_pipe.nil? || out_pipe.nil? ||
                        ffn_gate_pipe.nil? || ffn_up_pipe.nil? || ffn_down_pipe.nil?
          return nil unless n_tokens > 0

          ML::Metal::Device.init!

          hidden_dim = q_qw.in_dim
          q_dim = n_head * head_dim
          kv_dim = n_head_kv * head_dim
          ffn_dim = ffn_gate_qw.out_dim
          raise "final full-attn input size mismatch" unless inp.size == n_tokens * hidden_dim

          final_pos = start_pos + n_tokens - 1
          last_offset = (n_tokens - 1) * hidden_dim
          last_x = inp[last_offset, hidden_dim]
          last_byte_offset = last_offset.to_i64 * sizeof(Float32)

          inp_buf = Scratch.get(:full_last_inp, inp.size.to_i64 * sizeof(Float32))
          last_inp_buf = Scratch.get(:full_last_last_inp, hidden_dim.to_i64 * sizeof(Float32))
          norm_w_buf = Scratch.get(:full_last_norm_w, attn_norm.size.to_i64 * sizeof(Float32))
          cur_buf = Scratch.get(:full_last_cur, inp.size.to_i64 * sizeof(Float32))
          qfull_buf = Scratch.get(:full_last_qfull, q_qw.out_dim.to_i64 * sizeof(Float32))
          q_buf = Scratch.get(:full_last_q, q_dim.to_i64 * sizeof(Float32))
          gate_buf = Scratch.get(:full_last_gate, q_dim.to_i64 * sizeof(Float32))
          k_buf = Scratch.get(:full_last_k, (n_tokens * kv_dim).to_i64 * sizeof(Float32))
          v_buf = Scratch.get(:full_last_v, (n_tokens * kv_dim).to_i64 * sizeof(Float32))
          attn_buf = Scratch.get(:full_last_attn, q_dim.to_i64 * sizeof(Float32))
          attn_out_buf = Scratch.get(:full_last_attn_out, out_qw.out_dim.to_i64 * sizeof(Float32))
          qnorm_buf = Scratch.get(:full_last_qnorm, q_norm.size.to_i64 * sizeof(Float32))
          knorm_buf = Scratch.get(:full_last_knorm, k_norm.size.to_i64 * sizeof(Float32))
          post_norm_buf = Scratch.get(:full_last_postnorm_w, post_attention_norm.size.to_i64 * sizeof(Float32))
          residual_buf = Scratch.get(:full_last_residual, hidden_dim.to_i64 * sizeof(Float32))
          normed_buf = Scratch.get(:full_last_normed, hidden_dim.to_i64 * sizeof(Float32))
          ffn_gate_buf = Scratch.get(:full_last_ffn_gate, ffn_dim.to_i64 * sizeof(Float32))
          ffn_up_buf = Scratch.get(:full_last_ffn_up, ffn_dim.to_i64 * sizeof(Float32))
          ffn_comb_buf = Scratch.get(:full_last_ffn_comb, ffn_dim.to_i64 * sizeof(Float32))
          ffn_out_buf = Scratch.get(:full_last_ffn_out, ffn_down_qw.out_dim.to_i64 * sizeof(Float32))
          out_buf = Scratch.get(:full_last_out, hidden_dim.to_i64 * sizeof(Float32))

          inp_buf.write(inp)
          last_inp_buf.write(last_x)
          norm_w_buf.write(attn_norm)
          qnorm_buf.write(q_norm)
          knorm_buf.write(k_norm)
          post_norm_buf.write(post_attention_norm)

          q_w_buf, q_w_off = weight_slot(q_qw)
          k_w_buf, k_w_off = weight_slot(k_qw)
          v_w_buf, v_w_off = weight_slot(v_qw)
          out_w_buf, out_w_off = weight_slot(out_qw)
          ffn_gate_w_buf, ffn_gate_w_off = weight_slot(ffn_gate_qw)
          ffn_up_w_buf, ffn_up_w_off = weight_slot(ffn_up_qw)
          ffn_down_w_buf, ffn_down_w_off = weight_slot(ffn_down_qw)

          t0 = Time.instant if Profile.enabled?
          cmd = ML::Metal::CommandBuffer.new

          norm_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_rmsnorm_rows(norm_enc, inp_buf, norm_w_buf, cur_buf, hidden_dim, n_tokens, eps)
          norm_enc.end_encoding

          proj_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv_input_offset(proj_enc, q_pipe.not_nil!, cur_buf, last_byte_offset, qfull_buf, q_w_buf, q_w_off, q_qw.in_dim, q_qw.out_dim)
          encode_matmul(proj_enc, k_pipe.not_nil!, k_qw, cur_buf, k_buf, k_w_buf, k_w_off, k_qw.in_dim, k_qw.out_dim, n_tokens)
          encode_matmul(proj_enc, v_pipe.not_nil!, v_qw, cur_buf, v_buf, v_w_buf, v_w_off, v_qw.in_dim, v_qw.out_dim, n_tokens)
          proj_enc.end_encoding

          split_enc = ML::Metal::ComputeEncoder.new(cmd)
          split_enc.set_pipeline(split_qgate_pipeline)
          split_enc.set_buffer(qfull_buf, 0)
          split_enc.set_buffer(q_buf, 1, ML::Metal::BufferAccess::Write)
          split_enc.set_buffer(gate_buf, 2, ML::Metal::BufferAccess::Write)
          split_enc.set_value(n_head.to_u32, 3)
          split_enc.set_value(head_dim.to_u32, 4)
          split_enc.dispatch_1d(q_dim, 256)
          split_enc.end_encoding

          qnorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          qnorm_enc.set_pipeline(rmsnorm_heads_pipeline)
          qnorm_enc.set_buffer(q_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          qnorm_enc.set_buffer(qnorm_buf, 1)
          qnorm_enc.set_value(head_dim.to_u32, 2)
          qnorm_enc.set_value(eps, 3)
          qnorm_enc.dispatch_threadgroups({n_head, 1, 1}, {32, 1, 1})
          qnorm_enc.end_encoding

          knorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          knorm_enc.set_pipeline(rmsnorm_heads_rows_pipeline)
          knorm_enc.set_buffer(k_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          knorm_enc.set_buffer(knorm_buf, 1)
          knorm_enc.set_value(head_dim.to_u32, 2)
          knorm_enc.set_value(eps, 3)
          knorm_enc.set_value(n_head_kv.to_u32, 4)
          knorm_enc.set_value(n_tokens.to_u32, 5)
          knorm_enc.dispatch_threadgroups({n_head_kv, n_tokens, 1}, {32, 1, 1})
          knorm_enc.end_encoding

          qrope_enc = ML::Metal::ComputeEncoder.new(cmd)
          qrope_enc.set_pipeline(rope_partial_pipeline)
          qrope_enc.set_buffer(q_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          qrope_enc.set_value(head_dim.to_u32, 1)
          qrope_enc.set_value(rope_dim_count.to_u32, 2)
          qrope_enc.set_value(final_pos.to_u32, 3)
          qrope_enc.set_value(rope_freq_base, 4)
          qrope_enc.dispatch_threadgroups({n_head, 1, 1}, {32, 1, 1})
          qrope_enc.end_encoding

          krope_enc = ML::Metal::ComputeEncoder.new(cmd)
          krope_enc.set_pipeline(rope_partial_rows_pipeline)
          krope_enc.set_buffer(k_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          krope_enc.set_value(head_dim.to_u32, 1)
          krope_enc.set_value(rope_dim_count.to_u32, 2)
          krope_enc.set_value(start_pos.to_u32, 3)
          krope_enc.set_value(rope_freq_base, 4)
          krope_enc.set_value(n_head_kv.to_u32, 5)
          krope_enc.set_value(n_tokens.to_u32, 6)
          krope_enc.dispatch_threadgroups({n_head_kv, n_tokens, 1}, {32, 1, 1})
          krope_enc.end_encoding

          kvwrite_enc = ML::Metal::ComputeEncoder.new(cmd)
          kvwrite_enc.set_pipeline(kv_write_rows_pipeline)
          kvwrite_enc.set_buffer(k_buf, 0)
          kvwrite_enc.set_buffer(v_buf, 1)
          kvwrite_enc.set_buffer(k_cache_buf, 2, ML::Metal::BufferAccess::ReadWrite)
          kvwrite_enc.set_buffer(v_cache_buf, 3, ML::Metal::BufferAccess::ReadWrite)
          kvwrite_enc.set_value(start_pos.to_u32, 4)
          kvwrite_enc.set_value(kv_dim.to_u32, 5)
          kvwrite_enc.set_value(n_tokens.to_u32, 6)
          kvwrite_enc.dispatch_1d(n_tokens * kv_dim, 256)
          kvwrite_enc.end_encoding

          attn_enc = ML::Metal::ComputeEncoder.new(cmd)
          attn_enc.set_pipeline(attn_pipeline)
          attn_enc.set_buffer(q_buf, 0)
          attn_enc.set_buffer(gate_buf, 1)
          attn_enc.set_buffer(k_cache_buf, 2)
          attn_enc.set_buffer(v_cache_buf, 3)
          attn_enc.set_buffer(attn_buf, 4, ML::Metal::BufferAccess::Write)
          attn_enc.set_value((final_pos + 1).to_u32, 5)
          attn_enc.set_value(n_head.to_u32, 6)
          attn_enc.set_value(n_head_kv.to_u32, 7)
          attn_enc.set_value(head_dim.to_u32, 8)
          attn_enc.set_value(heads_per_group.to_u32, 9)
          attn_enc.set_value(scale, 10)
          attn_enc.dispatch_threadgroups({n_head, 1, 1}, {32, 1, 1})
          attn_enc.end_encoding

          outproj_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(outproj_enc, out_pipe.not_nil!, attn_buf, attn_out_buf, out_w_buf, out_w_off, out_qw.in_dim, out_qw.out_dim)
          outproj_enc.end_encoding

          addnorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          addnorm_enc.set_pipeline(add_rmsnorm_pipeline)
          addnorm_enc.set_buffer(last_inp_buf, 0)
          addnorm_enc.set_buffer(attn_out_buf, 1)
          addnorm_enc.set_buffer(post_norm_buf, 2)
          addnorm_enc.set_buffer(residual_buf, 3, ML::Metal::BufferAccess::Write)
          addnorm_enc.set_buffer(normed_buf, 4, ML::Metal::BufferAccess::Write)
          addnorm_enc.set_value(hidden_dim.to_u32, 5)
          addnorm_enc.set_value(eps, 6)
          addnorm_enc.dispatch_threadgroups({1, 1, 1}, {256, 1, 1})
          addnorm_enc.end_encoding

          ffn_proj_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(ffn_proj_enc, ffn_gate_pipe.not_nil!, normed_buf, ffn_gate_buf, ffn_gate_w_buf, ffn_gate_w_off, ffn_gate_qw.in_dim, ffn_gate_qw.out_dim)
          encode_gemv(ffn_proj_enc, ffn_up_pipe.not_nil!, normed_buf, ffn_up_buf, ffn_up_w_buf, ffn_up_w_off, ffn_up_qw.in_dim, ffn_up_qw.out_dim)
          ffn_proj_enc.end_encoding

          swiglu_enc = ML::Metal::ComputeEncoder.new(cmd)
          ffn_act_buf = swiglu_inplace_enabled? ? ffn_up_buf : ffn_comb_buf
          swiglu_enc.set_pipeline(ffn_swiglu_pipeline)
          swiglu_enc.set_buffer(ffn_gate_buf, 0)
          swiglu_enc.set_buffer(ffn_up_buf, 1)
          swiglu_enc.set_buffer(ffn_act_buf, 2, ML::Metal::BufferAccess::Write)
          swiglu_enc.set_value(ffn_dim.to_u32, 3)
          swiglu_enc.dispatch_1d(ffn_dim, 256)
          swiglu_enc.end_encoding

          ffn_down_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(ffn_down_enc, ffn_down_pipe.not_nil!, ffn_act_buf, ffn_out_buf, ffn_down_w_buf, ffn_down_w_off, ffn_down_qw.in_dim, ffn_down_qw.out_dim)
          ffn_down_enc.end_encoding

          add_enc = ML::Metal::ComputeEncoder.new(cmd)
          add_enc.set_pipeline(add_vec_pipeline)
          add_enc.set_buffer(residual_buf, 0)
          add_enc.set_buffer(ffn_out_buf, 1)
          add_enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          add_enc.set_value(hidden_dim.to_u32, 3)
          add_enc.dispatch_1d(hidden_dim, 256)
          add_enc.end_encoding

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_f32(out_buf, hidden_dim)
          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_attn(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          result
        end

        def self.full_attn_layer_chunk_project(inp : Array(Float32),
                                               q_qw : QuantWeight,
                                               k_qw : QuantWeight,
                                               v_qw : QuantWeight,
                                               attn_norm : Array(Float32),
                                               q_norm : Array(Float32),
                                               k_norm : Array(Float32),
                                               out_qw : QuantWeight,
                                               k_cache_buf : ML::MetalBuffer,
                                               v_cache_buf : ML::MetalBuffer,
                                               post_attention_norm : Array(Float32),
                                               ffn_gate_qw : QuantWeight,
                                               ffn_up_qw : QuantWeight,
                                               ffn_down_qw : QuantWeight,
                                               start_pos : Int32,
                                               n_tokens : Int32,
                                               n_head : Int32,
                                               n_head_kv : Int32,
                                               head_dim : Int32,
                                               rope_dim_count : Int32,
                                               heads_per_group : Int32,
                                               rope_freq_base : Float32,
                                               eps : Float32,
                                               scale : Float32) : Array(Float32)?
          q_pipe = gemv_pipeline_for(q_qw)
          k_pipe = gemv_pipeline_for(k_qw)
          v_pipe = gemv_pipeline_for(v_qw)
          out_pipe = gemv_pipeline_for(out_qw)
          ffn_gate_pipe = gemv_pipeline_for(ffn_gate_qw)
          ffn_up_pipe = gemv_pipeline_for(ffn_up_qw)
          ffn_down_pipe = gemv_pipeline_for(ffn_down_qw)
          return nil if q_pipe.nil? || k_pipe.nil? || v_pipe.nil? || out_pipe.nil? ||
                        ffn_gate_pipe.nil? || ffn_up_pipe.nil? || ffn_down_pipe.nil?
          return nil unless n_tokens > 0

          ML::Metal::Device.init!

          hidden_dim = q_qw.in_dim
          q_dim = n_head * head_dim
          kv_dim = n_head_kv * head_dim
          ffn_dim = ffn_gate_qw.out_dim
          raise "full_attn_layer_chunk input size mismatch" unless inp.size == n_tokens * hidden_dim

          inp_buf = Scratch.get(:full_chunk_inp, inp.size.to_i64 * sizeof(Float32))
          norm_w_buf = Scratch.get(:full_chunk_norm_w, attn_norm.size.to_i64 * sizeof(Float32))
          cur_buf = Scratch.get(:full_chunk_cur, inp.size.to_i64 * sizeof(Float32))
          qfull_buf = Scratch.get(:full_chunk_qfull, (n_tokens * q_qw.out_dim).to_i64 * sizeof(Float32))
          q_buf = Scratch.get(:full_chunk_q, (n_tokens * q_dim).to_i64 * sizeof(Float32))
          gate_buf = Scratch.get(:full_chunk_gate, (n_tokens * q_dim).to_i64 * sizeof(Float32))
          k_buf = Scratch.get(:full_chunk_k, (n_tokens * kv_dim).to_i64 * sizeof(Float32))
          v_buf = Scratch.get(:full_chunk_v, (n_tokens * kv_dim).to_i64 * sizeof(Float32))
          attn_buf = Scratch.get(:full_chunk_attn, (n_tokens * q_dim).to_i64 * sizeof(Float32))
          attn_out_buf = Scratch.get(:full_chunk_attn_out, (n_tokens * out_qw.out_dim).to_i64 * sizeof(Float32))
          qnorm_buf = Scratch.get(:full_chunk_qnorm, q_norm.size.to_i64 * sizeof(Float32))
          knorm_buf = Scratch.get(:full_chunk_knorm, k_norm.size.to_i64 * sizeof(Float32))
          post_norm_buf = Scratch.get(:full_chunk_postnorm_w, post_attention_norm.size.to_i64 * sizeof(Float32))
          residual_buf = Scratch.get(:full_chunk_residual, inp.size.to_i64 * sizeof(Float32))
          normed_buf = Scratch.get(:full_chunk_normed, inp.size.to_i64 * sizeof(Float32))
          ffn_gate_buf = Scratch.get(:full_chunk_ffn_gate, (n_tokens * ffn_dim).to_i64 * sizeof(Float32))
          ffn_up_buf = Scratch.get(:full_chunk_ffn_up, (n_tokens * ffn_dim).to_i64 * sizeof(Float32))
          ffn_comb_buf = Scratch.get(:full_chunk_ffn_comb, (n_tokens * ffn_dim).to_i64 * sizeof(Float32))
          ffn_out_buf = Scratch.get(:full_chunk_ffn_out, (n_tokens * ffn_down_qw.out_dim).to_i64 * sizeof(Float32))
          out_buf = Scratch.get(:full_chunk_out, inp.size.to_i64 * sizeof(Float32))

          inp_buf.write(inp)
          norm_w_buf.write(attn_norm)
          qnorm_buf.write(q_norm)
          knorm_buf.write(k_norm)
          post_norm_buf.write(post_attention_norm)

          q_w_buf, q_w_off = weight_slot(q_qw)
          k_w_buf, k_w_off = weight_slot(k_qw)
          v_w_buf, v_w_off = weight_slot(v_qw)
          out_w_buf, out_w_off = weight_slot(out_qw)
          ffn_gate_w_buf, ffn_gate_w_off = weight_slot(ffn_gate_qw)
          ffn_up_w_buf, ffn_up_w_off = weight_slot(ffn_up_qw)
          ffn_down_w_buf, ffn_down_w_off = weight_slot(ffn_down_qw)

          t0 = Time.instant if Profile.enabled?
          cmd = ML::Metal::CommandBuffer.new

          norm_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_rmsnorm_rows(norm_enc, inp_buf, norm_w_buf, cur_buf, hidden_dim, n_tokens, eps)
          norm_enc.end_encoding

          proj_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_matmul(proj_enc, q_pipe.not_nil!, q_qw, cur_buf, qfull_buf, q_w_buf, q_w_off, q_qw.in_dim, q_qw.out_dim, n_tokens)
          encode_matmul(proj_enc, k_pipe.not_nil!, k_qw, cur_buf, k_buf, k_w_buf, k_w_off, k_qw.in_dim, k_qw.out_dim, n_tokens)
          encode_matmul(proj_enc, v_pipe.not_nil!, v_qw, cur_buf, v_buf, v_w_buf, v_w_off, v_qw.in_dim, v_qw.out_dim, n_tokens)
          proj_enc.end_encoding

          split_enc = ML::Metal::ComputeEncoder.new(cmd)
          split_enc.set_pipeline(split_qgate_rows_pipeline)
          split_enc.set_buffer(qfull_buf, 0)
          split_enc.set_buffer(q_buf, 1, ML::Metal::BufferAccess::Write)
          split_enc.set_buffer(gate_buf, 2, ML::Metal::BufferAccess::Write)
          split_enc.set_value(n_head.to_u32, 3)
          split_enc.set_value(head_dim.to_u32, 4)
          split_enc.set_value(n_tokens.to_u32, 5)
          split_enc.dispatch_1d(n_tokens * q_dim, 256)
          split_enc.end_encoding

          qnorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          qnorm_enc.set_pipeline(rmsnorm_heads_rows_pipeline)
          qnorm_enc.set_buffer(q_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          qnorm_enc.set_buffer(qnorm_buf, 1)
          qnorm_enc.set_value(head_dim.to_u32, 2)
          qnorm_enc.set_value(eps, 3)
          qnorm_enc.set_value(n_head.to_u32, 4)
          qnorm_enc.set_value(n_tokens.to_u32, 5)
          qnorm_enc.dispatch_threadgroups({n_head, n_tokens, 1}, {32, 1, 1})
          qnorm_enc.end_encoding

          knorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          knorm_enc.set_pipeline(rmsnorm_heads_rows_pipeline)
          knorm_enc.set_buffer(k_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          knorm_enc.set_buffer(knorm_buf, 1)
          knorm_enc.set_value(head_dim.to_u32, 2)
          knorm_enc.set_value(eps, 3)
          knorm_enc.set_value(n_head_kv.to_u32, 4)
          knorm_enc.set_value(n_tokens.to_u32, 5)
          knorm_enc.dispatch_threadgroups({n_head_kv, n_tokens, 1}, {32, 1, 1})
          knorm_enc.end_encoding

          qrope_enc = ML::Metal::ComputeEncoder.new(cmd)
          qrope_enc.set_pipeline(rope_partial_rows_pipeline)
          qrope_enc.set_buffer(q_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          qrope_enc.set_value(head_dim.to_u32, 1)
          qrope_enc.set_value(rope_dim_count.to_u32, 2)
          qrope_enc.set_value(start_pos.to_u32, 3)
          qrope_enc.set_value(rope_freq_base, 4)
          qrope_enc.set_value(n_head.to_u32, 5)
          qrope_enc.set_value(n_tokens.to_u32, 6)
          qrope_enc.dispatch_threadgroups({n_head, n_tokens, 1}, {32, 1, 1})
          qrope_enc.end_encoding

          krope_enc = ML::Metal::ComputeEncoder.new(cmd)
          krope_enc.set_pipeline(rope_partial_rows_pipeline)
          krope_enc.set_buffer(k_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          krope_enc.set_value(head_dim.to_u32, 1)
          krope_enc.set_value(rope_dim_count.to_u32, 2)
          krope_enc.set_value(start_pos.to_u32, 3)
          krope_enc.set_value(rope_freq_base, 4)
          krope_enc.set_value(n_head_kv.to_u32, 5)
          krope_enc.set_value(n_tokens.to_u32, 6)
          krope_enc.dispatch_threadgroups({n_head_kv, n_tokens, 1}, {32, 1, 1})
          krope_enc.end_encoding

          kvwrite_enc = ML::Metal::ComputeEncoder.new(cmd)
          kvwrite_enc.set_pipeline(kv_write_rows_pipeline)
          kvwrite_enc.set_buffer(k_buf, 0)
          kvwrite_enc.set_buffer(v_buf, 1)
          kvwrite_enc.set_buffer(k_cache_buf, 2, ML::Metal::BufferAccess::ReadWrite)
          kvwrite_enc.set_buffer(v_cache_buf, 3, ML::Metal::BufferAccess::ReadWrite)
          kvwrite_enc.set_value(start_pos.to_u32, 4)
          kvwrite_enc.set_value(kv_dim.to_u32, 5)
          kvwrite_enc.set_value(n_tokens.to_u32, 6)
          kvwrite_enc.dispatch_1d(n_tokens * kv_dim, 256)
          kvwrite_enc.end_encoding

          attn_enc = ML::Metal::ComputeEncoder.new(cmd)
          attn_enc.set_pipeline(attn_rows_pipeline)
          attn_enc.set_buffer(q_buf, 0)
          attn_enc.set_buffer(gate_buf, 1)
          attn_enc.set_buffer(k_cache_buf, 2)
          attn_enc.set_buffer(v_cache_buf, 3)
          attn_enc.set_buffer(attn_buf, 4, ML::Metal::BufferAccess::Write)
          attn_enc.set_value(start_pos.to_u32, 5)
          attn_enc.set_value(n_tokens.to_u32, 6)
          attn_enc.set_value(n_head.to_u32, 7)
          attn_enc.set_value(n_head_kv.to_u32, 8)
          attn_enc.set_value(head_dim.to_u32, 9)
          attn_enc.set_value(heads_per_group.to_u32, 10)
          attn_enc.set_value(scale, 11)
          attn_enc.dispatch_threadgroups({n_head, n_tokens, 1}, {32, 1, 1})
          attn_enc.end_encoding

          outproj_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_matmul(outproj_enc, out_pipe.not_nil!, out_qw, attn_buf, attn_out_buf, out_w_buf, out_w_off, out_qw.in_dim, out_qw.out_dim, n_tokens)
          outproj_enc.end_encoding

          addnorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_add_rmsnorm_rows(addnorm_enc, inp_buf, attn_out_buf, post_norm_buf, residual_buf, normed_buf, hidden_dim, n_tokens, eps)
          addnorm_enc.end_encoding

          ffn_proj_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_matmul(ffn_proj_enc, ffn_gate_pipe.not_nil!, ffn_gate_qw, normed_buf, ffn_gate_buf, ffn_gate_w_buf, ffn_gate_w_off, ffn_gate_qw.in_dim, ffn_gate_qw.out_dim, n_tokens)
          encode_matmul(ffn_proj_enc, ffn_up_pipe.not_nil!, ffn_up_qw, normed_buf, ffn_up_buf, ffn_up_w_buf, ffn_up_w_off, ffn_up_qw.in_dim, ffn_up_qw.out_dim, n_tokens)
          ffn_proj_enc.end_encoding

          swiglu_enc = ML::Metal::ComputeEncoder.new(cmd)
          ffn_act_buf = swiglu_inplace_enabled? ? ffn_up_buf : ffn_comb_buf
          swiglu_enc.set_pipeline(ffn_swiglu_pipeline)
          swiglu_enc.set_buffer(ffn_gate_buf, 0)
          swiglu_enc.set_buffer(ffn_up_buf, 1)
          swiglu_enc.set_buffer(ffn_act_buf, 2, ML::Metal::BufferAccess::Write)
          swiglu_enc.set_value((n_tokens * ffn_dim).to_u32, 3)
          swiglu_enc.dispatch_1d(n_tokens * ffn_dim, 256)
          swiglu_enc.end_encoding

          ffn_down_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_matmul(ffn_down_enc, ffn_down_pipe.not_nil!, ffn_down_qw, ffn_act_buf, ffn_out_buf, ffn_down_w_buf, ffn_down_w_off, ffn_down_qw.in_dim, ffn_down_qw.out_dim, n_tokens)
          ffn_down_enc.end_encoding

          add_enc = ML::Metal::ComputeEncoder.new(cmd)
          add_enc.set_pipeline(add_vec_pipeline)
          add_enc.set_buffer(residual_buf, 0)
          add_enc.set_buffer(ffn_out_buf, 1)
          add_enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          add_enc.set_value((n_tokens * hidden_dim).to_u32, 3)
          add_enc.dispatch_1d(n_tokens * hidden_dim, 256)
          add_enc.end_encoding

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_f32(out_buf, n_tokens * hidden_dim)
          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_attn(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          result
        end

        # Exact prefill boundary fusion for the Qwen35 cadence:
        # one full-attention chunk followed by the next consecutive recurrent
        # run in a single command buffer. This removes the CPU read/write and
        # synchronization boundary between F and RRR groups without changing
        # math or state update order.
        def self.full_attn_then_recurrent_chunk_project_many(inp : Array(Float32),
                                                             q_qw : QuantWeight,
                                                             k_qw : QuantWeight,
                                                             v_qw : QuantWeight,
                                                             attn_norm : Array(Float32),
                                                             q_norm : Array(Float32),
                                                             k_norm : Array(Float32),
                                                             out_qw : QuantWeight,
                                                             k_cache_buf : ML::MetalBuffer,
                                                             v_cache_buf : ML::MetalBuffer,
                                                             post_attention_norm : Array(Float32),
                                                             ffn_gate_qw : QuantWeight,
                                                             ffn_up_qw : QuantWeight,
                                                             ffn_down_qw : QuantWeight,
                                                             start_pos : Int32,
                                                             n_tokens : Int32,
                                                             n_head : Int32,
                                                             n_head_kv : Int32,
                                                             head_dim : Int32,
                                                             rope_dim_count : Int32,
                                                             heads_per_group : Int32,
                                                             rope_freq_base : Float32,
                                                             eps : Float32,
                                                             scale : Float32,
                                                             conv_state_bufs : Array(ML::MetalBuffer),
                                                             ssm_state_bufs : Array(ML::MetalBuffer),
                                                             rec_layers : Array(Qwen35RecurrentWeights),
                                                             h_k : Int32,
                                                             h_v : Int32,
                                                             s : Int32,
                                                             conv_k : Int32,
                                                             profile_label : String = "full_rec_chunk_many") : Array(Float32)?
          q_pipe = gemv_pipeline_for(q_qw)
          k_pipe = gemv_pipeline_for(k_qw)
          v_pipe = gemv_pipeline_for(v_qw)
          out_pipe = gemv_pipeline_for(out_qw)
          full_ffn_gate_pipe = gemv_pipeline_for(ffn_gate_qw)
          full_ffn_up_pipe = gemv_pipeline_for(ffn_up_qw)
          full_ffn_down_pipe = gemv_pipeline_for(ffn_down_qw)
          return nil if q_pipe.nil? || k_pipe.nil? || v_pipe.nil? || out_pipe.nil? ||
                        full_ffn_gate_pipe.nil? || full_ffn_up_pipe.nil? || full_ffn_down_pipe.nil?
          return nil unless n_tokens > 0
          return nil if rec_layers.empty?
          return nil unless conv_state_bufs.size == rec_layers.size && ssm_state_bufs.size == rec_layers.size

          rec_layers.each do |lw|
            qkv_pipe = gemv_pipeline_for(lw.attn_qkv_qw)
            gate_pipe = gemv_pipeline_for(lw.attn_gate_qw)
            alpha_pipe = gemv_pipeline_for(lw.ssm_alpha_qw)
            beta_pipe = gemv_pipeline_for(lw.ssm_beta_qw)
            rec_out_pipe = gemv_pipeline_for(lw.ssm_out_qw)
            rec_ffn_gate_pipe = gemv_pipeline_for(lw.ffn_gate_qw)
            rec_ffn_up_pipe = gemv_pipeline_for(lw.ffn_up_qw)
            rec_ffn_down_pipe = gemv_pipeline_for(lw.ffn_down_qw)
            return nil if qkv_pipe.nil? || gate_pipe.nil? || alpha_pipe.nil? || beta_pipe.nil? ||
                          rec_out_pipe.nil? || rec_ffn_gate_pipe.nil? || rec_ffn_up_pipe.nil? ||
                          rec_ffn_down_pipe.nil?
          end

          ML::Metal::Device.init!

          hidden_dim = q_qw.in_dim
          q_dim = n_head * head_dim
          kv_dim = n_head_kv * head_dim
          full_ffn_dim = ffn_gate_qw.out_dim
          rec_qkv_dim = 2 * h_k * s + h_v * s
          d_inner = h_v * s
          rec_ffn_dim = rec_layers.first.ffn_gate_qw.out_dim
          rec_scale = (1.0 / Math.sqrt(s.to_f64)).to_f32
          raise "full+recurrent chunk input size mismatch" unless inp.size == n_tokens * hidden_dim

          full_tag = "frec_full_#{q_qw.raw.to_unsafe.address}"
          inp_buf = Scratch.get(:frec_inp, inp.size.to_i64 * sizeof(Float32))
          full_norm_w_buf = Scratch.get("#{full_tag}_norm_w", attn_norm.size.to_i64 * sizeof(Float32))
          full_cur_buf = Scratch.get(:frec_full_cur, inp.size.to_i64 * sizeof(Float32))
          full_qfull_buf = Scratch.get(:frec_full_qfull, (n_tokens * q_qw.out_dim).to_i64 * sizeof(Float32))
          full_q_buf = Scratch.get(:frec_full_q, (n_tokens * q_dim).to_i64 * sizeof(Float32))
          full_gate_buf = Scratch.get(:frec_full_gate, (n_tokens * q_dim).to_i64 * sizeof(Float32))
          full_k_buf = Scratch.get(:frec_full_k, (n_tokens * kv_dim).to_i64 * sizeof(Float32))
          full_v_buf = Scratch.get(:frec_full_v, (n_tokens * kv_dim).to_i64 * sizeof(Float32))
          full_attn_buf = Scratch.get(:frec_full_attn, (n_tokens * q_dim).to_i64 * sizeof(Float32))
          full_attn_out_buf = Scratch.get(:frec_full_attn_out, (n_tokens * out_qw.out_dim).to_i64 * sizeof(Float32))
          qnorm_buf = Scratch.get("#{full_tag}_qnorm", q_norm.size.to_i64 * sizeof(Float32))
          knorm_buf = Scratch.get("#{full_tag}_knorm", k_norm.size.to_i64 * sizeof(Float32))
          full_post_norm_buf = Scratch.get("#{full_tag}_postnorm", post_attention_norm.size.to_i64 * sizeof(Float32))
          full_residual_buf = Scratch.get(:frec_full_residual, inp.size.to_i64 * sizeof(Float32))
          full_normed_buf = Scratch.get(:frec_full_normed, inp.size.to_i64 * sizeof(Float32))
          full_ffn_gate_buf = Scratch.get(:frec_full_ffn_gate, (n_tokens * full_ffn_dim).to_i64 * sizeof(Float32))
          full_ffn_up_buf = Scratch.get(:frec_full_ffn_up, (n_tokens * full_ffn_dim).to_i64 * sizeof(Float32))
          full_ffn_comb_buf = Scratch.get(:frec_full_ffn_comb, (n_tokens * full_ffn_dim).to_i64 * sizeof(Float32))
          full_ffn_out_buf = Scratch.get(:frec_full_ffn_out, (n_tokens * ffn_down_qw.out_dim).to_i64 * sizeof(Float32))
          full_out_buf = Scratch.get(:frec_full_out, inp.size.to_i64 * sizeof(Float32))

          rec_dst_buf = Scratch.get(:frec_rec_hidden_b, inp.size.to_i64 * sizeof(Float32))
          rec_cur_buf = Scratch.get(:frec_rec_cur, inp.size.to_i64 * sizeof(Float32))
          rec_qkv_buf = Scratch.get(:frec_rec_qkv, (n_tokens * rec_qkv_dim).to_i64 * sizeof(Float32))
          rec_qkv_h16_buf = Scratch.get(:frec_rec_qkv_h16, (n_tokens * rec_qkv_dim).to_i64 * 2_i64)
          rec_z_buf = Scratch.get(:frec_rec_z, (n_tokens * d_inner).to_i64 * sizeof(Float32))
          rec_alpha_buf = Scratch.get(:frec_rec_alpha, (n_tokens * h_v).to_i64 * sizeof(Float32))
          rec_beta_buf = Scratch.get(:frec_rec_beta, (n_tokens * h_v).to_i64 * sizeof(Float32))
          rec_g_buf = Scratch.get(:frec_rec_g, (n_tokens * h_v).to_i64 * sizeof(Float32))
          rec_q_buf = Scratch.get(:frec_rec_q, (n_tokens * h_k * s).to_i64 * sizeof(Float32))
          rec_k_buf = Scratch.get(:frec_rec_k, (n_tokens * h_k * s).to_i64 * sizeof(Float32))
          rec_v_buf = Scratch.get(:frec_rec_v, (n_tokens * d_inner).to_i64 * sizeof(Float32))
          rec_attn_mid_buf = Scratch.get(:frec_rec_mid, (n_tokens * d_inner).to_i64 * sizeof(Float32))
          rec_attn_out_buf = Scratch.get(:frec_rec_attn_out, (n_tokens * hidden_dim).to_i64 * sizeof(Float32))
          rec_residual_buf = Scratch.get(:frec_rec_residual, inp.size.to_i64 * sizeof(Float32))
          rec_normed_buf = Scratch.get(:frec_rec_normed, inp.size.to_i64 * sizeof(Float32))
          rec_ffn_gate_buf = Scratch.get(:frec_rec_ffn_gate, (n_tokens * rec_ffn_dim).to_i64 * sizeof(Float32))
          rec_ffn_up_buf = Scratch.get(:frec_rec_ffn_up, (n_tokens * rec_ffn_dim).to_i64 * sizeof(Float32))
          rec_ffn_comb_buf = Scratch.get(:frec_rec_ffn_comb, (n_tokens * rec_ffn_dim).to_i64 * sizeof(Float32))
          rec_ffn_out_buf = Scratch.get(:frec_rec_ffn_out, (n_tokens * hidden_dim).to_i64 * sizeof(Float32))

          inp_buf.write(inp)
          ConstCache.write_once("#{full_tag}_norm_w", full_norm_w_buf, attn_norm)
          ConstCache.write_once("#{full_tag}_qnorm", qnorm_buf, q_norm)
          ConstCache.write_once("#{full_tag}_knorm", knorm_buf, k_norm)
          ConstCache.write_once("#{full_tag}_postnorm", full_post_norm_buf, post_attention_norm)

          q_w_buf, q_w_off = weight_slot(q_qw)
          k_w_buf, k_w_off = weight_slot(k_qw)
          v_w_buf, v_w_off = weight_slot(v_qw)
          out_w_buf, out_w_off = weight_slot(out_qw)
          full_ffn_gate_w_buf, full_ffn_gate_w_off = weight_slot(ffn_gate_qw)
          full_ffn_up_w_buf, full_ffn_up_w_off = weight_slot(ffn_up_qw)
          full_ffn_down_w_buf, full_ffn_down_w_off = weight_slot(ffn_down_qw)

          t0 = Time.instant if Profile.enabled?
          cmd = ML::Metal::CommandBuffer.new

          norm_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_rmsnorm_rows(norm_enc, inp_buf, full_norm_w_buf, full_cur_buf, hidden_dim, n_tokens, eps)
          norm_enc.end_encoding

          Profile.trace("prefill.full.qkv") do
            proj_enc = ML::Metal::ComputeEncoder.new(cmd)
            encode_matmul(proj_enc, q_pipe.not_nil!, q_qw, full_cur_buf, full_qfull_buf, q_w_buf, q_w_off, q_qw.in_dim, q_qw.out_dim, n_tokens)
            encode_matmul(proj_enc, k_pipe.not_nil!, k_qw, full_cur_buf, full_k_buf, k_w_buf, k_w_off, k_qw.in_dim, k_qw.out_dim, n_tokens)
            encode_matmul(proj_enc, v_pipe.not_nil!, v_qw, full_cur_buf, full_v_buf, v_w_buf, v_w_off, v_qw.in_dim, v_qw.out_dim, n_tokens)
            proj_enc.end_encoding
          end

          split_enc = ML::Metal::ComputeEncoder.new(cmd)
          split_enc.set_pipeline(split_qgate_rows_pipeline)
          split_enc.set_buffer(full_qfull_buf, 0)
          split_enc.set_buffer(full_q_buf, 1, ML::Metal::BufferAccess::Write)
          split_enc.set_buffer(full_gate_buf, 2, ML::Metal::BufferAccess::Write)
          split_enc.set_value(n_head.to_u32, 3)
          split_enc.set_value(head_dim.to_u32, 4)
          split_enc.set_value(n_tokens.to_u32, 5)
          split_enc.dispatch_1d(n_tokens * q_dim, 256)
          split_enc.end_encoding

          qnorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          qnorm_enc.set_pipeline(rmsnorm_heads_rows_pipeline)
          qnorm_enc.set_buffer(full_q_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          qnorm_enc.set_buffer(qnorm_buf, 1)
          qnorm_enc.set_value(head_dim.to_u32, 2)
          qnorm_enc.set_value(eps, 3)
          qnorm_enc.set_value(n_head.to_u32, 4)
          qnorm_enc.set_value(n_tokens.to_u32, 5)
          qnorm_enc.dispatch_threadgroups({n_head, n_tokens, 1}, {32, 1, 1})
          qnorm_enc.end_encoding

          knorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          knorm_enc.set_pipeline(rmsnorm_heads_rows_pipeline)
          knorm_enc.set_buffer(full_k_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          knorm_enc.set_buffer(knorm_buf, 1)
          knorm_enc.set_value(head_dim.to_u32, 2)
          knorm_enc.set_value(eps, 3)
          knorm_enc.set_value(n_head_kv.to_u32, 4)
          knorm_enc.set_value(n_tokens.to_u32, 5)
          knorm_enc.dispatch_threadgroups({n_head_kv, n_tokens, 1}, {32, 1, 1})
          knorm_enc.end_encoding

          qrope_enc = ML::Metal::ComputeEncoder.new(cmd)
          qrope_enc.set_pipeline(rope_partial_rows_pipeline)
          qrope_enc.set_buffer(full_q_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          qrope_enc.set_value(head_dim.to_u32, 1)
          qrope_enc.set_value(rope_dim_count.to_u32, 2)
          qrope_enc.set_value(start_pos.to_u32, 3)
          qrope_enc.set_value(rope_freq_base, 4)
          qrope_enc.set_value(n_head.to_u32, 5)
          qrope_enc.set_value(n_tokens.to_u32, 6)
          qrope_enc.dispatch_threadgroups({n_head, n_tokens, 1}, {32, 1, 1})
          qrope_enc.end_encoding

          krope_enc = ML::Metal::ComputeEncoder.new(cmd)
          krope_enc.set_pipeline(rope_partial_rows_pipeline)
          krope_enc.set_buffer(full_k_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          krope_enc.set_value(head_dim.to_u32, 1)
          krope_enc.set_value(rope_dim_count.to_u32, 2)
          krope_enc.set_value(start_pos.to_u32, 3)
          krope_enc.set_value(rope_freq_base, 4)
          krope_enc.set_value(n_head_kv.to_u32, 5)
          krope_enc.set_value(n_tokens.to_u32, 6)
          krope_enc.dispatch_threadgroups({n_head_kv, n_tokens, 1}, {32, 1, 1})
          krope_enc.end_encoding

          kvwrite_enc = ML::Metal::ComputeEncoder.new(cmd)
          kvwrite_enc.set_pipeline(kv_write_rows_pipeline)
          kvwrite_enc.set_buffer(full_k_buf, 0)
          kvwrite_enc.set_buffer(full_v_buf, 1)
          kvwrite_enc.set_buffer(k_cache_buf, 2, ML::Metal::BufferAccess::ReadWrite)
          kvwrite_enc.set_buffer(v_cache_buf, 3, ML::Metal::BufferAccess::ReadWrite)
          kvwrite_enc.set_value(start_pos.to_u32, 4)
          kvwrite_enc.set_value(kv_dim.to_u32, 5)
          kvwrite_enc.set_value(n_tokens.to_u32, 6)
          kvwrite_enc.dispatch_1d(n_tokens * kv_dim, 256)
          kvwrite_enc.end_encoding

          attn_enc = ML::Metal::ComputeEncoder.new(cmd)
          attn_enc.set_pipeline(attn_rows_pipeline)
          attn_enc.set_buffer(full_q_buf, 0)
          attn_enc.set_buffer(full_gate_buf, 1)
          attn_enc.set_buffer(k_cache_buf, 2)
          attn_enc.set_buffer(v_cache_buf, 3)
          attn_enc.set_buffer(full_attn_buf, 4, ML::Metal::BufferAccess::Write)
          attn_enc.set_value(start_pos.to_u32, 5)
          attn_enc.set_value(n_tokens.to_u32, 6)
          attn_enc.set_value(n_head.to_u32, 7)
          attn_enc.set_value(n_head_kv.to_u32, 8)
          attn_enc.set_value(head_dim.to_u32, 9)
          attn_enc.set_value(heads_per_group.to_u32, 10)
          attn_enc.set_value(scale, 11)
          attn_enc.dispatch_threadgroups({n_head, n_tokens, 1}, {32, 1, 1})
          attn_enc.end_encoding

          Profile.trace("prefill.full.o_proj") do
            outproj_enc = ML::Metal::ComputeEncoder.new(cmd)
            encode_matmul(outproj_enc, out_pipe.not_nil!, out_qw, full_attn_buf, full_attn_out_buf, out_w_buf, out_w_off, out_qw.in_dim, out_qw.out_dim, n_tokens)
            outproj_enc.end_encoding
          end

          addnorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_add_rmsnorm_rows(addnorm_enc, inp_buf, full_attn_out_buf, full_post_norm_buf, full_residual_buf, full_normed_buf, hidden_dim, n_tokens, eps)
          addnorm_enc.end_encoding

          Profile.trace("prefill.full.ffn_upgate") do
            full_ffn_proj_enc = ML::Metal::ComputeEncoder.new(cmd)
            pair_q4 = q4_pair_h16_gemm_candidate?(ffn_gate_qw, ffn_up_qw, n_tokens)
            if pair_q4
              Profile.bump_matmul_shape("q4_h16_gemm #{ffn_gate_qw.type.name} #{ffn_gate_qw.in_dim}x#{ffn_gate_qw.out_dim} b#{n_tokens}", ffn_gate_qw.raw.size.to_i64)
              Profile.bump_matmul_shape("q4_h16_gemm #{ffn_up_qw.type.name} #{ffn_up_qw.in_dim}x#{ffn_up_qw.out_dim} b#{n_tokens}", ffn_up_qw.raw.size.to_i64)
              encode_q4k_gemm_h16_pair(full_ffn_proj_enc, full_normed_buf, full_ffn_gate_buf, full_ffn_up_buf, full_ffn_gate_w_buf, full_ffn_gate_w_off, full_ffn_up_w_buf, full_ffn_up_w_off, ffn_gate_qw.in_dim, ffn_gate_qw.out_dim, n_tokens)
            else
              encode_matmul(full_ffn_proj_enc, full_ffn_gate_pipe.not_nil!, ffn_gate_qw, full_normed_buf, full_ffn_gate_buf, full_ffn_gate_w_buf, full_ffn_gate_w_off, ffn_gate_qw.in_dim, ffn_gate_qw.out_dim, n_tokens)
              encode_matmul(full_ffn_proj_enc, full_ffn_up_pipe.not_nil!, ffn_up_qw, full_normed_buf, full_ffn_up_buf, full_ffn_up_w_buf, full_ffn_up_w_off, ffn_up_qw.in_dim, ffn_up_qw.out_dim, n_tokens)
            end
            full_ffn_proj_enc.end_encoding
          end

          full_swiglu_enc = ML::Metal::ComputeEncoder.new(cmd)
          full_ffn_act_buf = swiglu_inplace_enabled? ? full_ffn_up_buf : full_ffn_comb_buf
          full_swiglu_enc.set_pipeline(ffn_swiglu_pipeline)
          full_swiglu_enc.set_buffer(full_ffn_gate_buf, 0)
          full_swiglu_enc.set_buffer(full_ffn_up_buf, 1)
          full_swiglu_enc.set_buffer(full_ffn_act_buf, 2, ML::Metal::BufferAccess::Write)
          full_swiglu_enc.set_value((n_tokens * full_ffn_dim).to_u32, 3)
          full_swiglu_enc.dispatch_1d(n_tokens * full_ffn_dim, 256)
          full_swiglu_enc.end_encoding

          fused_down_add = false
          if prefill_ffn_down_add_fused_enabled?
            Profile.trace("prefill.full.ffn_down_add") do
              full_ffn_down_enc = ML::Metal::ComputeEncoder.new(cmd)
              fused_down_add = encode_matmul_add(full_ffn_down_enc, full_ffn_down_pipe.not_nil!, ffn_down_qw, full_ffn_act_buf, full_residual_buf, full_out_buf, full_ffn_down_w_buf, full_ffn_down_w_off, ffn_down_qw.in_dim, ffn_down_qw.out_dim, n_tokens)
              full_ffn_down_enc.end_encoding
            end
          end

          unless fused_down_add
            Profile.trace("prefill.full.ffn_down") do
              full_ffn_down_enc = ML::Metal::ComputeEncoder.new(cmd)
              encode_matmul(full_ffn_down_enc, full_ffn_down_pipe.not_nil!, ffn_down_qw, full_ffn_act_buf, full_ffn_out_buf, full_ffn_down_w_buf, full_ffn_down_w_off, ffn_down_qw.in_dim, ffn_down_qw.out_dim, n_tokens)
              full_ffn_down_enc.end_encoding
            end

            full_add_enc = ML::Metal::ComputeEncoder.new(cmd)
            full_add_enc.set_pipeline(add_vec_pipeline)
            full_add_enc.set_buffer(full_residual_buf, 0)
            full_add_enc.set_buffer(full_ffn_out_buf, 1)
            full_add_enc.set_buffer(full_out_buf, 2, ML::Metal::BufferAccess::Write)
            full_add_enc.set_value((n_tokens * hidden_dim).to_u32, 3)
            full_add_enc.dispatch_1d(n_tokens * hidden_dim, 256)
            full_add_enc.end_encoding
          end

          src_buf = full_out_buf
          dst_buf = rec_dst_buf

          rec_layers.each_with_index do |lw, local_i|
            tag = "frec_rec_#{local_i}_#{lw.attn_qkv_qw.raw.to_unsafe.address}"
            norm_w_buf = Scratch.get("#{tag}_norm_w", lw.attn_norm.size.to_i64 * sizeof(Float32))
            conv_w_buf = Scratch.get("#{tag}_conv_w", lw.ssm_conv1d.size.to_i64 * sizeof(Float32))
            dt_bias_buf = Scratch.get("#{tag}_dt_bias", lw.ssm_dt_bias.size.to_i64 * sizeof(Float32))
            ssm_a_buf = Scratch.get("#{tag}_ssm_a", lw.ssm_a.size.to_i64 * sizeof(Float32))
            ssm_norm_buf = Scratch.get("#{tag}_ssm_norm", lw.ssm_norm.size.to_i64 * sizeof(Float32))
            post_w_buf = Scratch.get("#{tag}_post_w", lw.post_attention_norm.size.to_i64 * sizeof(Float32))
            ConstCache.write_once("#{tag}_norm_w", norm_w_buf, lw.attn_norm)
            ConstCache.write_once("#{tag}_conv_w", conv_w_buf, lw.ssm_conv1d)
            ConstCache.write_once("#{tag}_dt_bias", dt_bias_buf, lw.ssm_dt_bias)
            ConstCache.write_once("#{tag}_ssm_a", ssm_a_buf, lw.ssm_a)
            ConstCache.write_once("#{tag}_ssm_norm", ssm_norm_buf, lw.ssm_norm)
            ConstCache.write_once("#{tag}_post_w", post_w_buf, lw.post_attention_norm)

            qkv_w_buf, qkv_w_off = weight_slot(lw.attn_qkv_qw)
            gate_w_buf, gate_w_off = weight_slot(lw.attn_gate_qw)
            alpha_w_buf, alpha_w_off = weight_slot(lw.ssm_alpha_qw)
            beta_w_buf, beta_w_off = weight_slot(lw.ssm_beta_qw)
            rec_out_w_buf, rec_out_w_off = weight_slot(lw.ssm_out_qw)
            rec_ffn_gate_w_buf, rec_ffn_gate_w_off = weight_slot(lw.ffn_gate_qw)
            rec_ffn_up_w_buf, rec_ffn_up_w_off = weight_slot(lw.ffn_up_qw)
            rec_ffn_down_w_buf, rec_ffn_down_w_off = weight_slot(lw.ffn_down_qw)

            rec_norm_enc = ML::Metal::ComputeEncoder.new(cmd)
            encode_rmsnorm_rows(rec_norm_enc, src_buf, norm_w_buf, rec_cur_buf, hidden_dim, n_tokens, eps)
            rec_norm_enc.end_encoding

            Profile.trace("prefill.rec.proj") do
              rec_proj_enc = ML::Metal::ComputeEncoder.new(cmd)
              qkv_h16 = q5_qkv_h16_conv_enabled? && q56_batch_gemm_enabled? && lw.attn_qkv_qw.type.q5_k? && n_tokens > GEMM_BATCH_THRESHOLD
              shared_h16 = rec_proj_shared_h16_enabled? && qkv_h16 && q4_h16_gemm_enabled? &&
                           lw.attn_gate_qw.type.q4_k? && n_tokens > GEMM_BATCH_THRESHOLD
              if shared_h16
                rec_proj_x16_buf = Scratch.get(:full_rec_chunk_many_rec_proj_x16, (n_tokens * lw.attn_qkv_qw.in_dim).to_i64 * 2_i64)
                Profile.bump_conversion("f32_to_f16 rec_proj_shared_input #{lw.attn_qkv_qw.in_dim} b#{n_tokens}", (n_tokens * lw.attn_qkv_qw.in_dim).to_i64 * 6_i64)
                rec_proj_enc.set_pipeline(f32_to_f16_pipeline)
                rec_proj_enc.set_buffer(rec_cur_buf, 0)
                rec_proj_enc.set_buffer(rec_proj_x16_buf, 1, ML::Metal::BufferAccess::Write)
                rec_proj_enc.set_value((n_tokens * lw.attn_qkv_qw.in_dim).to_u32, 2)
                rec_proj_enc.dispatch_1d(n_tokens * lw.attn_qkv_qw.in_dim, 256)

                Profile.bump_matmul_shape("q5_h16_gemm #{lw.attn_qkv_qw.type.name} #{lw.attn_qkv_qw.in_dim}x#{lw.attn_qkv_qw.out_dim} b#{n_tokens}", lw.attn_qkv_qw.raw.size.to_i64)
                encode_q56k_gemm_h16_from_h16(rec_proj_enc, mm5_pipeline, rec_proj_x16_buf, rec_qkv_h16_buf, qkv_w_buf, qkv_w_off, lw.attn_qkv_qw.in_dim, lw.attn_qkv_qw.out_dim, n_tokens)
                Profile.bump_matmul_shape("q4_h16_gemm #{lw.attn_gate_qw.type.name} #{lw.attn_gate_qw.in_dim}x#{lw.attn_gate_qw.out_dim} b#{n_tokens}", lw.attn_gate_qw.raw.size.to_i64)
                encode_q4k_gemm_h16_from_h16(rec_proj_enc, rec_proj_x16_buf, rec_z_buf, gate_w_buf, gate_w_off, lw.attn_gate_qw.in_dim, lw.attn_gate_qw.out_dim, n_tokens)
              elsif qkv_h16
                Profile.bump_matmul_shape("q5_h16_gemm #{lw.attn_qkv_qw.type.name} #{lw.attn_qkv_qw.in_dim}x#{lw.attn_qkv_qw.out_dim} b#{n_tokens}", lw.attn_qkv_qw.raw.size.to_i64)
                encode_q56k_gemm_h16(rec_proj_enc, mm5_pipeline, rec_cur_buf, rec_qkv_h16_buf, qkv_w_buf, qkv_w_off, lw.attn_qkv_qw.in_dim, lw.attn_qkv_qw.out_dim, n_tokens)
                encode_matmul(rec_proj_enc, gemv_pipeline_for(lw.attn_gate_qw).not_nil!, lw.attn_gate_qw, rec_cur_buf, rec_z_buf, gate_w_buf, gate_w_off, lw.attn_gate_qw.in_dim, lw.attn_gate_qw.out_dim, n_tokens)
              else
                encode_matmul(rec_proj_enc, gemv_pipeline_for(lw.attn_qkv_qw).not_nil!, lw.attn_qkv_qw, rec_cur_buf, rec_qkv_buf, qkv_w_buf, qkv_w_off, lw.attn_qkv_qw.in_dim, lw.attn_qkv_qw.out_dim, n_tokens)
                encode_matmul(rec_proj_enc, gemv_pipeline_for(lw.attn_gate_qw).not_nil!, lw.attn_gate_qw, rec_cur_buf, rec_z_buf, gate_w_buf, gate_w_off, lw.attn_gate_qw.in_dim, lw.attn_gate_qw.out_dim, n_tokens)
              end
              encode_matmul(rec_proj_enc, gemv_pipeline_for(lw.ssm_alpha_qw).not_nil!, lw.ssm_alpha_qw, rec_cur_buf, rec_alpha_buf, alpha_w_buf, alpha_w_off, lw.ssm_alpha_qw.in_dim, lw.ssm_alpha_qw.out_dim, n_tokens)
              encode_matmul(rec_proj_enc, gemv_pipeline_for(lw.ssm_beta_qw).not_nil!, lw.ssm_beta_qw, rec_cur_buf, rec_beta_buf, beta_w_buf, beta_w_off, lw.ssm_beta_qw.in_dim, lw.ssm_beta_qw.out_dim, n_tokens)
              rec_proj_enc.end_encoding
            end

            conv_enc = ML::Metal::ComputeEncoder.new(cmd)
            qkv_h16 = q5_qkv_h16_conv_enabled? && q56_batch_gemm_enabled? && lw.attn_qkv_qw.type.q5_k? && n_tokens > GEMM_BATCH_THRESHOLD
            conv_enc.set_pipeline(qkv_h16 ? recurrent_conv_shift_chunk_h16_pipeline : recurrent_conv_shift_chunk_pipeline)
            conv_enc.set_buffer(conv_state_bufs[local_i], 0, ML::Metal::BufferAccess::ReadWrite)
            conv_enc.set_buffer(qkv_h16 ? rec_qkv_h16_buf : rec_qkv_buf, 1)
            conv_enc.set_buffer(conv_w_buf, 2)
            conv_enc.set_buffer(rec_q_buf, 3, ML::Metal::BufferAccess::Write)
            conv_enc.set_buffer(rec_k_buf, 4, ML::Metal::BufferAccess::Write)
            conv_enc.set_buffer(rec_v_buf, 5, ML::Metal::BufferAccess::Write)
            conv_enc.set_value(h_k.to_u32, 6)
            conv_enc.set_value(h_v.to_u32, 7)
            conv_enc.set_value(s.to_u32, 8)
            conv_enc.set_value(conv_k.to_u32, 9)
            conv_enc.set_value(n_tokens.to_u32, 10)
            conv_enc.dispatch_1d(rec_qkv_dim, 256)
            conv_enc.end_encoding

            rec_qnorm_enc = ML::Metal::ComputeEncoder.new(cmd)
            rec_qnorm_enc.set_pipeline(l2_heads_chunk_pipeline)
            rec_qnorm_enc.set_buffer(rec_q_buf, 0, ML::Metal::BufferAccess::ReadWrite)
            rec_qnorm_enc.set_value(h_k.to_u32, 1)
            rec_qnorm_enc.set_value(s.to_u32, 2)
            rec_qnorm_enc.set_value(eps, 3)
            rec_qnorm_enc.dispatch_threadgroups({h_k, n_tokens, 1}, {32, 1, 1})
            rec_qnorm_enc.end_encoding

            rec_knorm_enc = ML::Metal::ComputeEncoder.new(cmd)
            rec_knorm_enc.set_pipeline(l2_heads_chunk_pipeline)
            rec_knorm_enc.set_buffer(rec_k_buf, 0, ML::Metal::BufferAccess::ReadWrite)
            rec_knorm_enc.set_value(h_k.to_u32, 1)
            rec_knorm_enc.set_value(s.to_u32, 2)
            rec_knorm_enc.set_value(eps, 3)
            rec_knorm_enc.dispatch_threadgroups({h_k, n_tokens, 1}, {32, 1, 1})
            rec_knorm_enc.end_encoding

            ab_enc = ML::Metal::ComputeEncoder.new(cmd)
            ab_enc.set_pipeline(recurrent_ab_chunk_pipeline)
            ab_enc.set_buffer(rec_alpha_buf, 0)
            ab_enc.set_buffer(rec_beta_buf, 1, ML::Metal::BufferAccess::ReadWrite)
            ab_enc.set_buffer(dt_bias_buf, 2)
            ab_enc.set_buffer(ssm_a_buf, 3)
            ab_enc.set_buffer(rec_g_buf, 4, ML::Metal::BufferAccess::Write)
            ab_enc.set_value(h_v.to_u32, 5)
            ab_enc.set_value(n_tokens.to_u32, 6)
            ab_enc.dispatch_1d(n_tokens * h_v, 64)
            ab_enc.end_encoding

            dn_enc = ML::Metal::ComputeEncoder.new(cmd)
            use_dn_rowwise = dn_chunk_rowwise_enabled?(s)
            dn_enc.set_pipeline(use_dn_rowwise ? dn128_chunk_rowwise_pipeline : dn128_chunk_fused_pipeline)
            dn_enc.set_buffer(ssm_state_bufs[local_i], 0, ML::Metal::BufferAccess::ReadWrite)
            dn_enc.set_buffer(rec_q_buf, 1)
            dn_enc.set_buffer(rec_k_buf, 2)
            dn_enc.set_buffer(rec_v_buf, 3)
            dn_enc.set_buffer(rec_g_buf, 4)
            dn_enc.set_buffer(rec_beta_buf, 5)
            dn_enc.set_buffer(rec_attn_mid_buf, 6, ML::Metal::BufferAccess::Write)
            dn_enc.set_value(h_k.to_u32, 7)
            dn_enc.set_value(h_v.to_u32, 8)
            dn_enc.set_value(s.to_u32, 9)
            dn_enc.set_value(rec_scale, 10)
            dn_enc.set_value(n_tokens.to_u32, 11)
            if use_dn_rowwise
              dn_enc.dispatch_threadgroups({(s + 3) // 4, h_v, 1}, {32, 4, 1})
            else
              dn_enc.dispatch_threadgroups({h_v, 1, 1}, {128, 1, 1})
            end
            dn_enc.end_encoding

            post_enc = ML::Metal::ComputeEncoder.new(cmd)
            post_enc.set_pipeline(dn_post_chunk_pipeline)
            post_enc.set_buffer(rec_attn_mid_buf, 0, ML::Metal::BufferAccess::ReadWrite)
            post_enc.set_buffer(rec_z_buf, 1)
            post_enc.set_buffer(ssm_norm_buf, 2)
            post_enc.set_value(h_v.to_u32, 3)
            post_enc.set_value(s.to_u32, 4)
            post_enc.set_value(eps, 5)
            post_enc.set_value(n_tokens.to_u32, 6)
            post_enc.dispatch_threadgroups({h_v, n_tokens, 1}, {32, 1, 1})
            post_enc.end_encoding

            Profile.trace("prefill.rec.o_proj") do
              rec_out_enc = ML::Metal::ComputeEncoder.new(cmd)
              encode_matmul(rec_out_enc, gemv_pipeline_for(lw.ssm_out_qw).not_nil!, lw.ssm_out_qw, rec_attn_mid_buf, rec_attn_out_buf, rec_out_w_buf, rec_out_w_off, lw.ssm_out_qw.in_dim, lw.ssm_out_qw.out_dim, n_tokens)
              rec_out_enc.end_encoding
            end

            rec_addnorm_enc = ML::Metal::ComputeEncoder.new(cmd)
            encode_add_rmsnorm_rows(rec_addnorm_enc, src_buf, rec_attn_out_buf, post_w_buf, rec_residual_buf, rec_normed_buf, hidden_dim, n_tokens, eps)
            rec_addnorm_enc.end_encoding

            Profile.trace("prefill.rec.ffn_upgate") do
              rec_ffn_proj_enc = ML::Metal::ComputeEncoder.new(cmd)
              pair_q4 = q4_pair_h16_gemm_candidate?(lw.ffn_gate_qw, lw.ffn_up_qw, n_tokens)
              if pair_q4
                Profile.bump_matmul_shape("q4_h16_gemm #{lw.ffn_gate_qw.type.name} #{lw.ffn_gate_qw.in_dim}x#{lw.ffn_gate_qw.out_dim} b#{n_tokens}", lw.ffn_gate_qw.raw.size.to_i64)
                Profile.bump_matmul_shape("q4_h16_gemm #{lw.ffn_up_qw.type.name} #{lw.ffn_up_qw.in_dim}x#{lw.ffn_up_qw.out_dim} b#{n_tokens}", lw.ffn_up_qw.raw.size.to_i64)
                encode_q4k_gemm_h16_pair(rec_ffn_proj_enc, rec_normed_buf, rec_ffn_gate_buf, rec_ffn_up_buf, rec_ffn_gate_w_buf, rec_ffn_gate_w_off, rec_ffn_up_w_buf, rec_ffn_up_w_off, lw.ffn_gate_qw.in_dim, lw.ffn_gate_qw.out_dim, n_tokens)
              else
                encode_matmul(rec_ffn_proj_enc, gemv_pipeline_for(lw.ffn_gate_qw).not_nil!, lw.ffn_gate_qw, rec_normed_buf, rec_ffn_gate_buf, rec_ffn_gate_w_buf, rec_ffn_gate_w_off, lw.ffn_gate_qw.in_dim, lw.ffn_gate_qw.out_dim, n_tokens)
                encode_matmul(rec_ffn_proj_enc, gemv_pipeline_for(lw.ffn_up_qw).not_nil!, lw.ffn_up_qw, rec_normed_buf, rec_ffn_up_buf, rec_ffn_up_w_buf, rec_ffn_up_w_off, lw.ffn_up_qw.in_dim, lw.ffn_up_qw.out_dim, n_tokens)
              end
              rec_ffn_proj_enc.end_encoding
            end

            rec_swiglu_enc = ML::Metal::ComputeEncoder.new(cmd)
            rec_ffn_act_buf = swiglu_inplace_enabled? ? rec_ffn_up_buf : rec_ffn_comb_buf
            rec_swiglu_enc.set_pipeline(ffn_swiglu_pipeline)
            rec_swiglu_enc.set_buffer(rec_ffn_gate_buf, 0)
            rec_swiglu_enc.set_buffer(rec_ffn_up_buf, 1)
            rec_swiglu_enc.set_buffer(rec_ffn_act_buf, 2, ML::Metal::BufferAccess::Write)
            rec_swiglu_enc.set_value((n_tokens * rec_ffn_dim).to_u32, 3)
            rec_swiglu_enc.dispatch_1d(n_tokens * rec_ffn_dim, 256)
            rec_swiglu_enc.end_encoding

            fused_down_add = false
            if prefill_ffn_down_add_fused_enabled?
              Profile.trace("prefill.rec.ffn_down_add") do
                rec_ffn_down_enc = ML::Metal::ComputeEncoder.new(cmd)
                fused_down_add = encode_matmul_add(rec_ffn_down_enc, gemv_pipeline_for(lw.ffn_down_qw).not_nil!, lw.ffn_down_qw, rec_ffn_act_buf, rec_residual_buf, dst_buf, rec_ffn_down_w_buf, rec_ffn_down_w_off, lw.ffn_down_qw.in_dim, lw.ffn_down_qw.out_dim, n_tokens)
                rec_ffn_down_enc.end_encoding
              end
            end

            unless fused_down_add
              Profile.trace("prefill.rec.ffn_down") do
                rec_ffn_down_enc = ML::Metal::ComputeEncoder.new(cmd)
                encode_matmul(rec_ffn_down_enc, gemv_pipeline_for(lw.ffn_down_qw).not_nil!, lw.ffn_down_qw, rec_ffn_act_buf, rec_ffn_out_buf, rec_ffn_down_w_buf, rec_ffn_down_w_off, lw.ffn_down_qw.in_dim, lw.ffn_down_qw.out_dim, n_tokens)
                rec_ffn_down_enc.end_encoding
              end

              rec_add_enc = ML::Metal::ComputeEncoder.new(cmd)
              rec_add_enc.set_pipeline(add_vec_pipeline)
              rec_add_enc.set_buffer(rec_residual_buf, 0)
              rec_add_enc.set_buffer(rec_ffn_out_buf, 1)
              rec_add_enc.set_buffer(dst_buf, 2, ML::Metal::BufferAccess::Write)
              rec_add_enc.set_value((n_tokens * hidden_dim).to_u32, 3)
              rec_add_enc.dispatch_1d(n_tokens * hidden_dim, 256)
              rec_add_enc.end_encoding
            end

            src_buf, dst_buf = dst_buf, src_buf
          end

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_f32(src_buf, n_tokens * hidden_dim)
          if Profile.enabled?
            t_read = Time.instant
            encode_ns = (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64
            wait_ns = (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64
            read_ns = (t_read - t_wait.not_nil!).total_nanoseconds.to_i64
            Profile.bump_dn(encode_ns, wait_ns, read_ns)
            Profile.bump_group(profile_label, encode_ns, wait_ns, read_ns)
          end
          result
        end

        # Final output path:
        #   output RMSNorm -> lm_head projection
        # in one command buffer, with only logits read back.
        def self.rmsnorm_project(x : Array(Float32),
                                 norm_weight : Array(Float32),
                                 out_qw : QuantWeight,
                                 eps : Float32) : Array(Float32)?
          out_pipe = gemv_pipeline_for(out_qw)
          return nil if out_pipe.nil?

          ML::Metal::Device.init!

          hidden_dim = x.size
          x_buf = Scratch.get(:head_x, hidden_dim.to_i64 * sizeof(Float32))
          norm_w_buf = Scratch.get(:head_norm_w, norm_weight.size.to_i64 * sizeof(Float32))
          normed_buf = Scratch.get(:head_normed, hidden_dim.to_i64 * sizeof(Float32))
          out_buf = Scratch.get(:head_out, out_qw.out_dim.to_i64 * sizeof(Float32))
          x_buf.write(x)
          norm_w_buf.write(norm_weight)

          out_w_buf, out_w_off = if slot = mmap_slot_for(out_qw.raw)
                                   slot
                                 else
                                   {out_qw.fallback_metal_buffer, 0_i64}
                                 end

          t0 = Time.instant if Profile.enabled?
          cmd = ML::Metal::CommandBuffer.new

          norm_enc = ML::Metal::ComputeEncoder.new(cmd)
          norm_enc.set_pipeline(rmsnorm_vec_pipeline)
          norm_enc.set_buffer(x_buf, 0)
          norm_enc.set_buffer(norm_w_buf, 1)
          norm_enc.set_buffer(normed_buf, 2, ML::Metal::BufferAccess::Write)
          norm_enc.set_value(hidden_dim.to_u32, 3)
          norm_enc.set_value(eps, 4)
          norm_enc.dispatch_threadgroups({1, 1, 1}, {256, 1, 1})
          norm_enc.end_encoding

          out_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(out_enc, out_pipe.not_nil!, normed_buf, out_buf, out_w_buf, out_w_off, out_qw.in_dim, out_qw.out_dim)
          out_enc.end_encoding

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_f32(out_buf, out_qw.out_dim)
          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_gemv(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          result
        end

        def self.rmsnorm_project_top1(x : Array(Float32),
                                      norm_weight : Array(Float32),
                                      out_qw : QuantWeight,
                                      eps : Float32) : Array(Float32)?
          return nil unless can_use_head_top1_fused?(out_qw)

          ML::Metal::Device.init!

          hidden_dim = x.size
          tile_count = (out_qw.out_dim + HEAD_TOP1_ROWS_PER_TG - 1) // HEAD_TOP1_ROWS_PER_TG
          x_buf = Scratch.get(:head_top1_x, hidden_dim.to_i64 * sizeof(Float32))
          norm_w_buf = Scratch.get(:head_top1_norm_w, norm_weight.size.to_i64 * sizeof(Float32))
          normed_buf = Scratch.get(:head_top1_normed, hidden_dim.to_i64 * sizeof(Float32))
          tile_values_buf = Scratch.get(:head_top1_tile_values, tile_count.to_i64 * sizeof(Float32))
          tile_ids_buf = Scratch.get(:head_top1_tile_ids, tile_count.to_i64 * sizeof(UInt32))
          top1_id_buf = Scratch.get(:head_top1_id, sizeof(UInt32).to_i64)
          top1_value_buf = Scratch.get(:head_top1_value, sizeof(Float32).to_i64)
          x_buf.write(x)
          norm_w_buf.write(norm_weight)

          out_w_buf, out_w_off = weight_slot(out_qw)

          t0 = Time.instant if Profile.enabled?
          cmd = ML::Metal::CommandBuffer.new

          norm_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_rmsnorm_vec(norm_enc, x_buf, norm_w_buf, normed_buf, hidden_dim, eps)
          norm_enc.end_encoding

          head_top1_enc = ML::Metal::ComputeEncoder.new(cmd)
          head_top1_enc.set_pipeline(out_qw.type.q8_0? ? mv8_top1_tiles_pipeline : mv6_top1_tiles_pipeline)
          head_top1_enc.set_buffer(out_w_buf, 0, ML::Metal::BufferAccess::Read, offset: out_w_off)
          head_top1_enc.set_buffer(normed_buf, 1)
          head_top1_enc.set_buffer(tile_values_buf, 2, ML::Metal::BufferAccess::Write)
          head_top1_enc.set_buffer(tile_ids_buf, 3, ML::Metal::BufferAccess::Write)
          head_top1_enc.set_value(out_qw.in_dim.to_u32, 4)
          head_top1_enc.set_value(out_qw.out_dim.to_u32, 5)
          head_top1_enc.dispatch_threadgroups({tile_count, 1, 1}, {out_qw.type.q8_0? ? MV_Q8_NSG * 32 : 64, 1, 1})
          head_top1_enc.end_encoding

          reduce_top1_enc = ML::Metal::ComputeEncoder.new(cmd)
          reduce_top1_enc.set_pipeline(top1_reduce_tiles_pipeline)
          reduce_top1_enc.set_buffer(tile_values_buf, 0)
          reduce_top1_enc.set_buffer(tile_ids_buf, 1)
          reduce_top1_enc.set_buffer(top1_id_buf, 2, ML::Metal::BufferAccess::Write)
          reduce_top1_enc.set_buffer(top1_value_buf, 3, ML::Metal::BufferAccess::Write)
          reduce_top1_enc.set_value(tile_count.to_u32, 4)
          reduce_top1_enc.dispatch_threadgroups({1, 1, 1}, {256, 1, 1})
          reduce_top1_enc.end_encoding

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_top1(top1_id_buf, top1_value_buf)
          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_gemv(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          result
        end

        def self.rmsnorm_project_top1_rows(x : Array(Float32),
                                           rows : Int32,
                                           norm_weight : Array(Float32),
                                           out_qw : QuantWeight,
                                           eps : Float32) : Array({Int32, Float32})?
          return nil unless head_top1_fused_enabled?
          return nil unless out_qw.type.q6_k?
          return nil unless out_qw.in_dim % QK_K == 0
          return nil unless rows > 0
          hidden_dim = out_qw.in_dim
          return nil unless x.size == rows * hidden_dim

          ML::Metal::Device.init!

          tile_count = (out_qw.out_dim + HEAD_TOP1_ROWS_PER_TG - 1) // HEAD_TOP1_ROWS_PER_TG
          x_buf = Scratch.get(:head_top1_rows_x, x.size.to_i64 * sizeof(Float32))
          norm_w_buf = Scratch.get(:head_top1_rows_norm_w, norm_weight.size.to_i64 * sizeof(Float32))
          normed_buf = Scratch.get(:head_top1_rows_normed, x.size.to_i64 * sizeof(Float32))
          tile_values_buf = Scratch.get(:head_top1_rows_tile_values, (rows * tile_count).to_i64 * sizeof(Float32))
          tile_ids_buf = Scratch.get(:head_top1_rows_tile_ids, (rows * tile_count).to_i64 * sizeof(UInt32))
          top1_id_buf = Scratch.get(:head_top1_rows_id, rows.to_i64 * sizeof(UInt32))
          top1_value_buf = Scratch.get(:head_top1_rows_value, rows.to_i64 * sizeof(Float32))
          x_buf.write(x)
          norm_w_buf.write(norm_weight)

          out_w_buf, out_w_off = weight_slot(out_qw)

          t0 = Time.instant if Profile.enabled?
          cmd = ML::Metal::CommandBuffer.new

          norm_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_rmsnorm_rows(norm_enc, x_buf, norm_w_buf, normed_buf, hidden_dim, rows, eps)
          norm_enc.end_encoding

          head_top1_enc = ML::Metal::ComputeEncoder.new(cmd)
          head_top1_enc.set_pipeline(mv6_top1_tiles_batch_pipeline)
          head_top1_enc.set_buffer(out_w_buf, 0, ML::Metal::BufferAccess::Read, offset: out_w_off)
          head_top1_enc.set_buffer(normed_buf, 1)
          head_top1_enc.set_buffer(tile_values_buf, 2, ML::Metal::BufferAccess::Write)
          head_top1_enc.set_buffer(tile_ids_buf, 3, ML::Metal::BufferAccess::Write)
          head_top1_enc.set_value(out_qw.in_dim.to_u32, 4)
          head_top1_enc.set_value(out_qw.out_dim.to_u32, 5)
          head_top1_enc.set_value(tile_count.to_u32, 6)
          head_top1_enc.dispatch_threadgroups({tile_count, rows, 1}, {64, 1, 1})
          head_top1_enc.end_encoding

          reduce_top1_enc = ML::Metal::ComputeEncoder.new(cmd)
          reduce_top1_enc.set_pipeline(top1_reduce_tiles_batch_pipeline)
          reduce_top1_enc.set_buffer(tile_values_buf, 0)
          reduce_top1_enc.set_buffer(tile_ids_buf, 1)
          reduce_top1_enc.set_buffer(top1_id_buf, 2, ML::Metal::BufferAccess::Write)
          reduce_top1_enc.set_buffer(top1_value_buf, 3, ML::Metal::BufferAccess::Write)
          reduce_top1_enc.set_value(tile_count.to_u32, 4)
          reduce_top1_enc.dispatch_threadgroups({rows, 1, 1}, {256, 1, 1})
          reduce_top1_enc.end_encoding

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_top1_rows(top1_id_buf, top1_value_buf, rows)
          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_gemv(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          result
        end

        def self.rmsnorm_project_full_top1_rows_guarded(x : Array(Float32),
                                                        rows : Int32,
                                                        norm_weight : Array(Float32),
                                                        out_qw : QuantWeight,
                                                        eps : Float32) : Array({Int32, Float32})?
          return nil unless out_qw.type.q6_k?
          return nil unless out_qw.in_dim % QK_K == 0
          return nil unless rows > GEMM_BATCH_THRESHOLD
          hidden_dim = out_qw.in_dim
          return nil unless x.size == rows * hidden_dim

          margin = (ENV["QWEN35_HEAD_FULL_ROWS_MARGIN"]? || "0.25").to_f32
          return nil if margin < 0.0_f32

          ML::Metal::Device.init!

          x_buf = Scratch.get(:head_full_rows_guard_x, x.size.to_i64 * sizeof(Float32))
          norm_w_buf = Scratch.get(:head_full_rows_guard_norm_w, norm_weight.size.to_i64 * sizeof(Float32))
          normed_buf = Scratch.get(:head_full_rows_guard_normed, x.size.to_i64 * sizeof(Float32))
          normed16_buf = Scratch.get(:head_full_rows_guard_normed16, x.size.to_i64 * 2_i64)
          bias_buf = Scratch.get("head_full_rows_guard_bias_#{out_qw.out_dim}", out_qw.out_dim.to_i64 * sizeof(Float32))
          logits16_buf = Scratch.get(:head_full_rows_guard_logits16, (rows * out_qw.out_dim).to_i64 * 2_i64)
          top1_id_buf = Scratch.get(:head_full_rows_guard_id, rows.to_i64 * sizeof(UInt32))
          top1_value_buf = Scratch.get(:head_full_rows_guard_value, rows.to_i64 * sizeof(Float32))
          second_id_buf = Scratch.get(:head_full_rows_guard_second_id, rows.to_i64 * sizeof(UInt32))
          second_value_buf = Scratch.get(:head_full_rows_guard_second_value, rows.to_i64 * sizeof(Float32))
          x_buf.write(x)
          norm_w_buf.write(norm_weight)
          ConstCache.write_zero_f32_once("head_full_rows_guard_bias_#{out_qw.out_dim}", bias_buf, out_qw.out_dim)

          out_w_buf, out_w_off = weight_slot(out_qw)

          t0 = Time.instant if Profile.enabled?
          cmd = ML::Metal::CommandBuffer.new

          norm_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_rmsnorm_rows(norm_enc, x_buf, norm_w_buf, normed_buf, hidden_dim, rows, eps)
          norm_enc.end_encoding

          out_enc = ML::Metal::ComputeEncoder.new(cmd)
          Profile.bump_matmul_shape("q6_gemm_top2_guard #{out_qw.type.name} #{out_qw.in_dim}x#{out_qw.out_dim} b#{rows}", out_qw.raw.size.to_i64)
          Profile.bump_conversion("f32_to_f16 q56_gemm_input_guard #{out_qw.in_dim} b#{rows}", (rows * out_qw.in_dim).to_i64 * 6_i64)
          out_enc.set_pipeline(f32_to_f16_pipeline)
          out_enc.set_buffer(normed_buf, 0)
          out_enc.set_buffer(normed16_buf, 1, ML::Metal::BufferAccess::Write)
          out_enc.set_value((rows * out_qw.in_dim).to_u32, 2)
          out_enc.dispatch_1d(rows * out_qw.in_dim, 256)

          out_enc.set_pipeline(mm6_pipeline)
          out_enc.set_buffer(out_w_buf, 0, ML::Metal::BufferAccess::Read, offset: out_w_off)
          out_enc.set_buffer(normed16_buf, 1)
          out_enc.set_buffer(bias_buf, 2)
          out_enc.set_buffer(logits16_buf, 3, ML::Metal::BufferAccess::Write)
          out_enc.set_value(out_qw.in_dim.to_u32, 4)
          out_enc.set_value(out_qw.out_dim.to_u32, 5)
          out_enc.set_value(rows.to_u32, 6)
          out_enc.set_value(0_u32, 7)
          out_enc.set_threadgroup_memory(MM_SHMEM, 0)
          out_enc.dispatch_threadgroups({
            (rows + MM_NR1 - 1) // MM_NR1,
            (out_qw.out_dim + MM_NR0 - 1) // MM_NR0,
            1,
          }, {MM_TG, 1, 1})
          out_enc.end_encoding

          reduce_enc = ML::Metal::ComputeEncoder.new(cmd)
          reduce_enc.set_pipeline(top2_reduce_f16_rows_pipeline)
          reduce_enc.set_buffer(logits16_buf, 0)
          reduce_enc.set_buffer(top1_id_buf, 1, ML::Metal::BufferAccess::Write)
          reduce_enc.set_buffer(top1_value_buf, 2, ML::Metal::BufferAccess::Write)
          reduce_enc.set_buffer(second_id_buf, 3, ML::Metal::BufferAccess::Write)
          reduce_enc.set_buffer(second_value_buf, 4, ML::Metal::BufferAccess::Write)
          reduce_enc.set_value(out_qw.out_dim.to_u32, 5)
          reduce_enc.dispatch_threadgroups({rows, 1, 1}, {256, 1, 1})
          reduce_enc.end_encoding

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?

          top2s = read_shared_top2_rows(top1_id_buf, top1_value_buf, second_id_buf, second_value_buf, rows)
          result = Array({Int32, Float32}).new(rows) { |i| {top2s[i][0], top2s[i][1]} }

          fallback_rows = [] of Int32
          top2s.each_with_index do |(_, top_value, _, second_value), i|
            fallback_rows << i.to_i32 if top_value - second_value < margin
          end

          if !fallback_rows.empty?
            fallback_x = Array(Float32).new(fallback_rows.size * hidden_dim, 0.0_f32)
            fallback_rows.each_with_index do |row, compact_row|
              src_offset = row * hidden_dim
              dst_offset = compact_row * hidden_dim
              hidden_dim.times do |j|
                fallback_x[dst_offset + j] = x[src_offset + j]
              end
            end
            if exact = rmsnorm_project_top1_rows(fallback_x, fallback_rows.size.to_i32, norm_weight, out_qw, eps)
              fallback_rows.each_with_index do |row, compact_row|
                result[row] = exact[compact_row]
              end
            else
              return nil
            end
          end

          if ENV["QWEN35_HEAD_FULL_ROWS_GUARD_TRACE"]? == "1"
            STDERR.puts "head_full_rows_guard rows=#{rows} fallback=#{fallback_rows.size} margin=#{margin}"
          end

          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_gemv(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end

          result
        end

        def self.rmsnorm_project_full_top1_rows(x : Array(Float32),
                                                rows : Int32,
                                                norm_weight : Array(Float32),
                                                out_qw : QuantWeight,
                                                eps : Float32) : Array({Int32, Float32})?
          return nil unless out_qw.type.q6_k?
          return nil unless out_qw.in_dim % QK_K == 0
          return nil unless rows > GEMM_BATCH_THRESHOLD
          hidden_dim = out_qw.in_dim
          return nil unless x.size == rows * hidden_dim

          ML::Metal::Device.init!

          x_buf = Scratch.get(:head_full_rows_x, x.size.to_i64 * sizeof(Float32))
          norm_w_buf = Scratch.get(:head_full_rows_norm_w, norm_weight.size.to_i64 * sizeof(Float32))
          normed_buf = Scratch.get(:head_full_rows_normed, x.size.to_i64 * sizeof(Float32))
          normed16_buf = Scratch.get(:head_full_rows_normed16, x.size.to_i64 * 2_i64)
          bias_buf = Scratch.get("head_full_rows_bias_#{out_qw.out_dim}", out_qw.out_dim.to_i64 * sizeof(Float32))
          logits16_buf = Scratch.get(:head_full_rows_logits16, (rows * out_qw.out_dim).to_i64 * 2_i64)
          top1_id_buf = Scratch.get(:head_full_rows_id, rows.to_i64 * sizeof(UInt32))
          top1_value_buf = Scratch.get(:head_full_rows_value, rows.to_i64 * sizeof(Float32))
          x_buf.write(x)
          norm_w_buf.write(norm_weight)
          ConstCache.write_zero_f32_once("head_full_rows_bias_#{out_qw.out_dim}", bias_buf, out_qw.out_dim)

          out_w_buf, out_w_off = weight_slot(out_qw)

          t0 = Time.instant if Profile.enabled?
          cmd = ML::Metal::CommandBuffer.new

          norm_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_rmsnorm_rows(norm_enc, x_buf, norm_w_buf, normed_buf, hidden_dim, rows, eps)
          norm_enc.end_encoding

          out_enc = ML::Metal::ComputeEncoder.new(cmd)
          Profile.bump_matmul_shape("q6_gemm_top1 #{out_qw.type.name} #{out_qw.in_dim}x#{out_qw.out_dim} b#{rows}", out_qw.raw.size.to_i64)
          Profile.bump_conversion("f32_to_f16 q56_gemm_input #{out_qw.in_dim} b#{rows}", (rows * out_qw.in_dim).to_i64 * 6_i64)
          out_enc.set_pipeline(f32_to_f16_pipeline)
          out_enc.set_buffer(normed_buf, 0)
          out_enc.set_buffer(normed16_buf, 1, ML::Metal::BufferAccess::Write)
          out_enc.set_value((rows * out_qw.in_dim).to_u32, 2)
          out_enc.dispatch_1d(rows * out_qw.in_dim, 256)

          out_enc.set_pipeline(mm6_pipeline)
          out_enc.set_buffer(out_w_buf, 0, ML::Metal::BufferAccess::Read, offset: out_w_off)
          out_enc.set_buffer(normed16_buf, 1)
          out_enc.set_buffer(bias_buf, 2)
          out_enc.set_buffer(logits16_buf, 3, ML::Metal::BufferAccess::Write)
          out_enc.set_value(out_qw.in_dim.to_u32, 4)
          out_enc.set_value(out_qw.out_dim.to_u32, 5)
          out_enc.set_value(rows.to_u32, 6)
          out_enc.set_value(0_u32, 7)
          out_enc.set_threadgroup_memory(MM_SHMEM, 0)
          out_enc.dispatch_threadgroups({
            (rows + MM_NR1 - 1) // MM_NR1,
            (out_qw.out_dim + MM_NR0 - 1) // MM_NR0,
            1,
          }, {MM_TG, 1, 1})
          out_enc.end_encoding

          reduce_enc = ML::Metal::ComputeEncoder.new(cmd)
          reduce_enc.set_pipeline(top1_reduce_f16_rows_pipeline)
          reduce_enc.set_buffer(logits16_buf, 0)
          reduce_enc.set_buffer(top1_id_buf, 1, ML::Metal::BufferAccess::Write)
          reduce_enc.set_buffer(top1_value_buf, 2, ML::Metal::BufferAccess::Write)
          reduce_enc.set_value(out_qw.out_dim.to_u32, 3)
          reduce_enc.dispatch_threadgroups({rows, 1, 1}, {256, 1, 1})
          reduce_enc.end_encoding

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_top1_rows(top1_id_buf, top1_value_buf, rows)

          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_gemv(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end

          result
        end

        # Whole-token decode wave:
        #   embedding upload -> all 32 layers on GPU with ping-pong hidden buffers
        #   -> output RMSNorm + lm_head -> logits readback
        #
        # This is the first route that removes the per-layer CPU round-trip.
        def self.forward_decode_wave(emb : Array(Float32),
                                     layers : Array(Qwen35LayerWeights),
                                     k_cache_bufs : Array(ML::MetalBuffer?),
                                     v_cache_bufs : Array(ML::MetalBuffer?),
                                     conv_state_bufs : Array(ML::MetalBuffer?),
                                     ssm_state_bufs : Array(ML::MetalBuffer?),
                                     output_norm : Array(Float32),
                                     output_qw : QuantWeight,
                                     hp : Qwen35Hparams,
                                     pos : Int32,
                                     top1 : Bool = false,
                                     emit_head : Bool = true) : Array(Float32)?
          out_pipe = gemv_pipeline_for(output_qw)
          return nil if emit_head && out_pipe.nil?

          ML::Metal::Device.init!

          hidden_dim = emb.size
          q_dim = hp.n_head * hp.head_dim
          kv_dim = hp.n_head_kv * hp.head_dim
          rec_qkv_dim = 2 * hp.ssm_group_count * hp.ssm_state_size + hp.ssm_time_step_rank * hp.ssm_state_size
          d_inner = hp.ssm_inner_size
          rec_ffn_dim = layers.each.find(&.is_a?(Qwen35RecurrentWeights)).try(&.as(Qwen35RecurrentWeights).ffn_gate_qw.out_dim) || 0
          full_ffn_dim = layers.each.find(&.is_a?(Qwen35FullAttnWeights)).try(&.as(Qwen35FullAttnWeights).ffn_gate_qw.out_dim) || 0
          ffn_dim = rec_ffn_dim > 0 ? rec_ffn_dim : full_ffn_dim

          src_buf = Scratch.get(:wave_hidden_a, hidden_dim.to_i64 * sizeof(Float32))
          dst_buf = Scratch.get(:wave_hidden_b, hidden_dim.to_i64 * sizeof(Float32))
          pre_norm_buf = Scratch.get(:wave_pre_norm, hidden_dim.to_i64 * sizeof(Float32))
          residual_buf = Scratch.get(:wave_residual, hidden_dim.to_i64 * sizeof(Float32))
          output_norm_buf = emit_head ? Scratch.get(:wave_output_norm, output_norm.size.to_i64 * sizeof(Float32)) : nil
          logits_buf = emit_head ? Scratch.get(:wave_logits, output_qw.out_dim.to_i64 * sizeof(Float32)) : nil
          tile_count = emit_head ? ((output_qw.out_dim + HEAD_TOP1_ROWS_PER_TG - 1) // HEAD_TOP1_ROWS_PER_TG) : 0
          top1_tile_values_buf = emit_head ? Scratch.get(:wave_top1_tile_values, tile_count.to_i64 * sizeof(Float32)) : nil
          top1_tile_ids_buf = emit_head ? Scratch.get(:wave_top1_tile_ids, tile_count.to_i64 * sizeof(UInt32)) : nil
          top1_id_buf = emit_head ? Scratch.get(:wave_top1_id, sizeof(UInt32).to_i64) : nil
          top1_value_buf = emit_head ? Scratch.get(:wave_top1_value, sizeof(Float32).to_i64) : nil

          # Full-attention scratch.
          qfull_buf = Scratch.get(:wave_qfull, (2 * q_dim).to_i64 * sizeof(Float32))
          q_buf = Scratch.get(:wave_q, q_dim.to_i64 * sizeof(Float32))
          gate_buf = Scratch.get(:wave_gate, q_dim.to_i64 * sizeof(Float32))
          k_buf = Scratch.get(:wave_k, kv_dim.to_i64 * sizeof(Float32))
          v_buf = Scratch.get(:wave_v, kv_dim.to_i64 * sizeof(Float32))
          attn_buf = Scratch.get(:wave_attn, q_dim.to_i64 * sizeof(Float32))
          attn_out_buf = Scratch.get(:wave_attn_out, hidden_dim.to_i64 * sizeof(Float32))

          # Recurrent scratch.
          rec_qkv_buf = Scratch.get(:wave_rec_qkv, rec_qkv_dim.to_i64 * sizeof(Float32))
          z_buf = Scratch.get(:wave_rec_z, d_inner.to_i64 * sizeof(Float32))
          alpha_buf = Scratch.get(:wave_rec_alpha, hp.ssm_time_step_rank.to_i64 * sizeof(Float32))
          beta_buf = Scratch.get(:wave_rec_beta, hp.ssm_time_step_rank.to_i64 * sizeof(Float32))
          g_buf = Scratch.get(:wave_rec_g, hp.ssm_time_step_rank.to_i64 * sizeof(Float32))
          rec_q_buf = Scratch.get(:wave_rec_q, (hp.ssm_group_count * hp.ssm_state_size).to_i64 * sizeof(Float32))
          rec_k_buf = Scratch.get(:wave_rec_k, (hp.ssm_group_count * hp.ssm_state_size).to_i64 * sizeof(Float32))
          rec_v_buf = Scratch.get(:wave_rec_v, d_inner.to_i64 * sizeof(Float32))
          rec_mid_buf = Scratch.get(:wave_rec_mid, d_inner.to_i64 * sizeof(Float32))
          rec_attn_out_buf = Scratch.get(:wave_rec_attn_out, hidden_dim.to_i64 * sizeof(Float32))

          # FFN scratch shared by both layer kinds.
          ffn_gate_buf = Scratch.get(:wave_ffn_gate, ffn_dim.to_i64 * sizeof(Float32))
          ffn_up_buf = Scratch.get(:wave_ffn_up, ffn_dim.to_i64 * sizeof(Float32))
          ffn_comb_buf = Scratch.get(:wave_ffn_comb, ffn_dim.to_i64 * sizeof(Float32))
          ffn_out_buf = Scratch.get(:wave_ffn_out, hidden_dim.to_i64 * sizeof(Float32))

          src_buf.write(emb)
          if emit_head
            ConstCache.write_once("wave_output_norm", output_norm_buf.not_nil!, output_norm)
          end

          layer_norm_bufs = Array(ML::MetalBuffer?).new(layers.size, nil)
          post_norm_bufs = Array(ML::MetalBuffer?).new(layers.size, nil)
          qnorm_bufs = Array(ML::MetalBuffer?).new(layers.size, nil)
          knorm_bufs = Array(ML::MetalBuffer?).new(layers.size, nil)
          conv_w_bufs = Array(ML::MetalBuffer?).new(layers.size, nil)
          dt_bias_bufs = Array(ML::MetalBuffer?).new(layers.size, nil)
          ssm_a_bufs = Array(ML::MetalBuffer?).new(layers.size, nil)
          ssm_norm_bufs = Array(ML::MetalBuffer?).new(layers.size, nil)

          layers.each_with_index do |lw, il|
            case lw
            in Qwen35FullAttnWeights
              layer_norm_buf = Scratch.get("wave_layer_norm_#{il}", hidden_dim.to_i64 * sizeof(Float32))
              ConstCache.write_once("wave_layer_norm_#{il}", layer_norm_buf, lw.attn_norm)
              layer_norm_bufs[il] = layer_norm_buf

              post_norm_buf = Scratch.get("wave_post_norm_#{il}", hidden_dim.to_i64 * sizeof(Float32))
              ConstCache.write_once("wave_post_norm_#{il}", post_norm_buf, lw.post_attention_norm)
              post_norm_bufs[il] = post_norm_buf

              qnorm_buf = Scratch.get("wave_qnorm_#{il}", hp.head_dim.to_i64 * sizeof(Float32))
              ConstCache.write_once("wave_qnorm_#{il}", qnorm_buf, lw.attn_q_norm)
              qnorm_bufs[il] = qnorm_buf

              knorm_buf = Scratch.get("wave_knorm_#{il}", hp.head_dim.to_i64 * sizeof(Float32))
              ConstCache.write_once("wave_knorm_#{il}", knorm_buf, lw.attn_k_norm)
              knorm_bufs[il] = knorm_buf
            in Qwen35RecurrentWeights
              layer_norm_buf = Scratch.get("wave_layer_norm_#{il}", hidden_dim.to_i64 * sizeof(Float32))
              ConstCache.write_once("wave_layer_norm_#{il}", layer_norm_buf, lw.attn_norm)
              layer_norm_bufs[il] = layer_norm_buf

              post_norm_buf = Scratch.get("wave_post_norm_#{il}", hidden_dim.to_i64 * sizeof(Float32))
              ConstCache.write_once("wave_post_norm_#{il}", post_norm_buf, lw.post_attention_norm)
              post_norm_bufs[il] = post_norm_buf

              conv_w_buf = Scratch.get("wave_rec_conv_w_#{il}", (hp.ssm_conv_kernel * rec_qkv_dim).to_i64 * sizeof(Float32))
              ConstCache.write_once("wave_rec_conv_w_#{il}", conv_w_buf, lw.ssm_conv1d)
              conv_w_bufs[il] = conv_w_buf

              dt_bias_buf = Scratch.get("wave_rec_dt_bias_#{il}", hp.ssm_time_step_rank.to_i64 * sizeof(Float32))
              ConstCache.write_once("wave_rec_dt_bias_#{il}", dt_bias_buf, lw.ssm_dt_bias)
              dt_bias_bufs[il] = dt_bias_buf

              ssm_a_buf = Scratch.get("wave_rec_ssm_a_#{il}", hp.ssm_time_step_rank.to_i64 * sizeof(Float32))
              ConstCache.write_once("wave_rec_ssm_a_#{il}", ssm_a_buf, lw.ssm_a)
              ssm_a_bufs[il] = ssm_a_buf

              ssm_norm_buf = Scratch.get("wave_rec_ssm_norm_#{il}", hp.ssm_state_size.to_i64 * sizeof(Float32))
              ConstCache.write_once("wave_rec_ssm_norm_#{il}", ssm_norm_buf, lw.ssm_norm)
              ssm_norm_bufs[il] = ssm_norm_buf
            end
          end

          out_w_buf, out_w_off = emit_head ? weight_slot(output_qw) : {nil, 0_i64}
          attn_scale = (1.0 / Math.sqrt(hp.head_dim.to_f64)).to_f32
          rec_scale = (1.0 / Math.sqrt(hp.ssm_state_size.to_f64)).to_f32
          use_dn_post_fused = dn_post_fused_enabled?
          wave_dn_pipeline = use_dn_post_fused ? dn128_fused_post_pipeline : active_dn_pipeline
          wave_dn_threadgroup_size = use_dn_post_fused ? 128 : dn_threadgroup_size
          use_conv_shift_fused = recurrent_conv_shift_fused_enabled?
          chunk_layers = wave_chunk_layers
          pending_cmds = [] of ML::Metal::CommandBuffer

          t0 = Time.instant if Profile.enabled?
          cmd = ML::Metal::CommandBuffer.new(fast: wave_fast_command_buffer_enabled?)

          layers.each_with_index do |lw, il|
            case lw
            in Qwen35FullAttnWeights
              Profile.trace("full.layer") do
              layer_norm_buf = layer_norm_bufs[il].not_nil!
              post_norm_buf = post_norm_bufs[il].not_nil!
              qnorm_buf = qnorm_bufs[il].not_nil!
              knorm_buf = knorm_bufs[il].not_nil!
              q_w_buf, q_w_off = weight_slot(lw.attn_q_qw)
              k_w_buf, k_w_off = weight_slot(lw.attn_k_qw)
              v_w_buf, v_w_off = weight_slot(lw.attn_v_qw)
              attn_out_w_buf, attn_out_w_off = weight_slot(lw.attn_output_qw)
              ffn_gate_w_buf, ffn_gate_w_off = weight_slot(lw.ffn_gate_qw)
              ffn_up_w_buf, ffn_up_w_off = weight_slot(lw.ffn_up_qw)
              ffn_down_w_buf, ffn_down_w_off = weight_slot(lw.ffn_down_qw)
              k_cache_buf = k_cache_bufs[il].not_nil!
              v_cache_buf = v_cache_bufs[il].not_nil!

              Profile.trace("full.norm") do
                norm_enc = ML::Metal::ComputeEncoder.new(cmd)
                encode_rmsnorm_vec(norm_enc, src_buf, layer_norm_buf, pre_norm_buf, hidden_dim, hp.rms_eps)
                norm_enc.end_encoding
              end

              Profile.trace("full.qkv") do
                proj_enc = ML::Metal::ComputeEncoder.new(cmd)
                encode_gemv(proj_enc, gemv_pipeline_for(lw.attn_q_qw).not_nil!, pre_norm_buf, qfull_buf, q_w_buf, q_w_off, lw.attn_q_qw.in_dim, lw.attn_q_qw.out_dim)
                if q8_kv_dual_gemv_candidate?(lw.attn_k_qw, lw.attn_v_qw)
                  encode_gemv_q8_dual(proj_enc, pre_norm_buf, k_buf, v_buf,
                    k_w_buf, k_w_off, v_w_buf, v_w_off, lw.attn_k_qw.in_dim, lw.attn_k_qw.out_dim)
                else
                  encode_gemv(proj_enc, gemv_pipeline_for(lw.attn_k_qw).not_nil!, pre_norm_buf, k_buf, k_w_buf, k_w_off, lw.attn_k_qw.in_dim, lw.attn_k_qw.out_dim)
                  encode_gemv(proj_enc, gemv_pipeline_for(lw.attn_v_qw).not_nil!, pre_norm_buf, v_buf, v_w_buf, v_w_off, lw.attn_v_qw.in_dim, lw.attn_v_qw.out_dim)
                end
                proj_enc.end_encoding
              end

              Profile.trace("full.qgate") do
                split_enc = ML::Metal::ComputeEncoder.new(cmd)
                split_enc.set_pipeline(split_qgate_pipeline)
                split_enc.set_buffer(qfull_buf, 0)
                split_enc.set_buffer(q_buf, 1, ML::Metal::BufferAccess::Write)
                split_enc.set_buffer(gate_buf, 2, ML::Metal::BufferAccess::Write)
                split_enc.set_value(hp.n_head.to_u32, 3)
                split_enc.set_value(hp.head_dim.to_u32, 4)
                split_enc.dispatch_1d(q_dim, 256)
                split_enc.end_encoding
              end

              Profile.trace("full.qknorm") do
                qnorm_enc = ML::Metal::ComputeEncoder.new(cmd)
                qnorm_enc.set_pipeline(rmsnorm_heads_pipeline)
                qnorm_enc.set_buffer(q_buf, 0, ML::Metal::BufferAccess::ReadWrite)
                qnorm_enc.set_buffer(qnorm_buf, 1)
                qnorm_enc.set_value(hp.head_dim.to_u32, 2)
                qnorm_enc.set_value(hp.rms_eps, 3)
                qnorm_enc.dispatch_threadgroups({hp.n_head, 1, 1}, {32, 1, 1})
                qnorm_enc.end_encoding

                knorm_enc = ML::Metal::ComputeEncoder.new(cmd)
                knorm_enc.set_pipeline(rmsnorm_heads_pipeline)
                knorm_enc.set_buffer(k_buf, 0, ML::Metal::BufferAccess::ReadWrite)
                knorm_enc.set_buffer(knorm_buf, 1)
                knorm_enc.set_value(hp.head_dim.to_u32, 2)
                knorm_enc.set_value(hp.rms_eps, 3)
                knorm_enc.dispatch_threadgroups({hp.n_head_kv, 1, 1}, {32, 1, 1})
                knorm_enc.end_encoding
              end

              Profile.trace("full.rope") do
                qrope_enc = ML::Metal::ComputeEncoder.new(cmd)
                qrope_enc.set_pipeline(rope_partial_pipeline)
                qrope_enc.set_buffer(q_buf, 0, ML::Metal::BufferAccess::ReadWrite)
                qrope_enc.set_value(hp.head_dim.to_u32, 1)
                qrope_enc.set_value(hp.rope_dim_count.to_u32, 2)
                qrope_enc.set_value(pos.to_u32, 3)
                qrope_enc.set_value(hp.rope_freq_base, 4)
                qrope_enc.dispatch_threadgroups({hp.n_head, 1, 1}, {32, 1, 1})
                qrope_enc.end_encoding

                krope_enc = ML::Metal::ComputeEncoder.new(cmd)
                krope_enc.set_pipeline(rope_partial_pipeline)
                krope_enc.set_buffer(k_buf, 0, ML::Metal::BufferAccess::ReadWrite)
                krope_enc.set_value(hp.head_dim.to_u32, 1)
                krope_enc.set_value(hp.rope_dim_count.to_u32, 2)
                krope_enc.set_value(pos.to_u32, 3)
                krope_enc.set_value(hp.rope_freq_base, 4)
                krope_enc.dispatch_threadgroups({hp.n_head_kv, 1, 1}, {32, 1, 1})
                krope_enc.end_encoding
              end

              Profile.trace("full.attn") do
                kvwrite_enc = ML::Metal::ComputeEncoder.new(cmd)
                kvwrite_enc.set_pipeline(kv_write_pipeline)
                kvwrite_enc.set_buffer(k_buf, 0)
                kvwrite_enc.set_buffer(v_buf, 1)
                kvwrite_enc.set_buffer(k_cache_buf, 2, ML::Metal::BufferAccess::ReadWrite)
                kvwrite_enc.set_buffer(v_cache_buf, 3, ML::Metal::BufferAccess::ReadWrite)
                kvwrite_enc.set_value((pos * kv_dim).to_u32, 4)
                kvwrite_enc.set_value(kv_dim.to_u32, 5)
                kvwrite_enc.dispatch_1d(kv_dim, 256)
                kvwrite_enc.end_encoding

                attn_enc = ML::Metal::ComputeEncoder.new(cmd)
                attn_enc.set_pipeline(attn_pipeline)
                attn_enc.set_buffer(q_buf, 0)
                attn_enc.set_buffer(gate_buf, 1)
                attn_enc.set_buffer(k_cache_buf, 2)
                attn_enc.set_buffer(v_cache_buf, 3)
                attn_enc.set_buffer(attn_buf, 4, ML::Metal::BufferAccess::Write)
                attn_enc.set_value((pos + 1).to_u32, 5)
                attn_enc.set_value(hp.n_head.to_u32, 6)
                attn_enc.set_value(hp.n_head_kv.to_u32, 7)
                attn_enc.set_value(hp.head_dim.to_u32, 8)
                attn_enc.set_value((hp.n_head // hp.n_head_kv).to_u32, 9)
                attn_enc.set_value(attn_scale, 10)
                attn_enc.dispatch_threadgroups({hp.n_head, 1, 1}, {32, 1, 1})
                attn_enc.end_encoding
              end

              Profile.trace("full.o_proj") do
                attn_out_enc = ML::Metal::ComputeEncoder.new(cmd)
                encode_gemv(attn_out_enc, gemv_pipeline_for(lw.attn_output_qw).not_nil!, attn_buf, attn_out_buf, attn_out_w_buf, attn_out_w_off, lw.attn_output_qw.in_dim, lw.attn_output_qw.out_dim)
                attn_out_enc.end_encoding
              end

              Profile.trace("full.addnorm") do
                addnorm_enc = ML::Metal::ComputeEncoder.new(cmd)
                encode_add_rmsnorm(addnorm_enc, src_buf, attn_out_buf, post_norm_buf, residual_buf, pre_norm_buf, hidden_dim, hp.rms_eps)
                addnorm_enc.end_encoding
              end

              Profile.trace("full.ffn_upgate") do
                ffn_proj_enc = ML::Metal::ComputeEncoder.new(cmd)
                if q8_dual_gemv_candidate?(lw.ffn_gate_qw, lw.ffn_up_qw)
                  encode_gemv_q8_dual(ffn_proj_enc, pre_norm_buf, ffn_gate_buf, ffn_up_buf,
                    ffn_gate_w_buf, ffn_gate_w_off, ffn_up_w_buf, ffn_up_w_off,
                    lw.ffn_gate_qw.in_dim, lw.ffn_gate_qw.out_dim)
                else
                  encode_gemv(ffn_proj_enc, gemv_pipeline_for(lw.ffn_gate_qw).not_nil!, pre_norm_buf, ffn_gate_buf, ffn_gate_w_buf, ffn_gate_w_off, lw.ffn_gate_qw.in_dim, lw.ffn_gate_qw.out_dim)
                  encode_gemv(ffn_proj_enc, gemv_pipeline_for(lw.ffn_up_qw).not_nil!, pre_norm_buf, ffn_up_buf, ffn_up_w_buf, ffn_up_w_off, lw.ffn_up_qw.in_dim, lw.ffn_up_qw.out_dim)
                end
                ffn_proj_enc.end_encoding
              end

              Profile.trace("full.ffn_act") do
                swiglu_enc = ML::Metal::ComputeEncoder.new(cmd)
                ffn_act_buf = decode_swiglu_inplace_enabled? ? ffn_up_buf : ffn_comb_buf
                swiglu_enc.set_pipeline(ffn_swiglu_pipeline)
                swiglu_enc.set_buffer(ffn_gate_buf, 0)
                swiglu_enc.set_buffer(ffn_up_buf, 1)
                swiglu_enc.set_buffer(ffn_act_buf, 2, ML::Metal::BufferAccess::Write)
                swiglu_enc.set_value(lw.ffn_gate_qw.out_dim.to_u32, 3)
                swiglu_enc.dispatch_1d(lw.ffn_gate_qw.out_dim, 256)
                swiglu_enc.end_encoding
              end

              if ffn_down_add_fused_enabled? && (add_pipe = gemv_add_pipeline_for(lw.ffn_down_qw))
                Profile.trace("full.ffn_down_add") do
                  ffn_down_enc = ML::Metal::ComputeEncoder.new(cmd)
                  ffn_act_buf = decode_swiglu_inplace_enabled? ? ffn_up_buf : ffn_comb_buf
                  encode_gemv_add(ffn_down_enc, add_pipe, ffn_act_buf, residual_buf, dst_buf,
                    ffn_down_w_buf, ffn_down_w_off, lw.ffn_down_qw.in_dim, lw.ffn_down_qw.out_dim)
                  ffn_down_enc.end_encoding
                end
              else
                Profile.trace("full.ffn_down") do
                  ffn_down_enc = ML::Metal::ComputeEncoder.new(cmd)
                  ffn_act_buf = decode_swiglu_inplace_enabled? ? ffn_up_buf : ffn_comb_buf
                  encode_gemv(ffn_down_enc, gemv_pipeline_for(lw.ffn_down_qw).not_nil!, ffn_act_buf, ffn_out_buf, ffn_down_w_buf, ffn_down_w_off, lw.ffn_down_qw.in_dim, lw.ffn_down_qw.out_dim)
                  ffn_down_enc.end_encoding
                end

                Profile.trace("full.add") do
                  add_enc = ML::Metal::ComputeEncoder.new(cmd)
                  add_enc.set_pipeline(add_vec_pipeline)
                  add_enc.set_buffer(residual_buf, 0)
                  add_enc.set_buffer(ffn_out_buf, 1)
                  add_enc.set_buffer(dst_buf, 2, ML::Metal::BufferAccess::Write)
                  add_enc.set_value(hidden_dim.to_u32, 3)
                  add_enc.dispatch_1d(hidden_dim, 256)
                  add_enc.end_encoding
                end
              end
              end
            in Qwen35RecurrentWeights
              Profile.trace("rec.layer") do
              layer_norm_buf = layer_norm_bufs[il].not_nil!
              post_norm_buf = post_norm_bufs[il].not_nil!
              conv_w_buf = conv_w_bufs[il].not_nil!
              dt_bias_buf = dt_bias_bufs[il].not_nil!
              ssm_a_buf = ssm_a_bufs[il].not_nil!
              ssm_norm_buf = ssm_norm_bufs[il].not_nil!
              qkv_w_buf, qkv_w_off = weight_slot(lw.attn_qkv_qw)
              gate_w_buf, gate_w_off = weight_slot(lw.attn_gate_qw)
              alpha_w_buf, alpha_w_off = weight_slot(lw.ssm_alpha_qw)
              beta_w_buf, beta_w_off = weight_slot(lw.ssm_beta_qw)
              ssm_out_w_buf, ssm_out_w_off = weight_slot(lw.ssm_out_qw)
              ffn_gate_w_buf, ffn_gate_w_off = weight_slot(lw.ffn_gate_qw)
              ffn_up_w_buf, ffn_up_w_off = weight_slot(lw.ffn_up_qw)
              ffn_down_w_buf, ffn_down_w_off = weight_slot(lw.ffn_down_qw)
              conv_state_buf = conv_state_bufs[il].not_nil!
              ssm_state_buf = ssm_state_bufs[il].not_nil!

                Profile.trace("rec.norm") do
                  norm_enc = ML::Metal::ComputeEncoder.new(cmd)
                  encode_rmsnorm_vec(norm_enc, src_buf, layer_norm_buf, pre_norm_buf, hidden_dim, hp.rms_eps)
                  norm_enc.end_encoding
                end

                Profile.trace("rec.proj") do
                  proj_enc = ML::Metal::ComputeEncoder.new(cmd)
                  encode_gemv(proj_enc, gemv_pipeline_for(lw.attn_qkv_qw).not_nil!, pre_norm_buf, rec_qkv_buf, qkv_w_buf, qkv_w_off, lw.attn_qkv_qw.in_dim, lw.attn_qkv_qw.out_dim)
                  encode_gemv(proj_enc, gemv_pipeline_for(lw.attn_gate_qw).not_nil!, pre_norm_buf, z_buf, gate_w_buf, gate_w_off, lw.attn_gate_qw.in_dim, lw.attn_gate_qw.out_dim)
                  if q8_alpha_beta_dual_gemv_candidate?(lw.ssm_alpha_qw, lw.ssm_beta_qw)
                    encode_gemv_q8_dual(proj_enc, pre_norm_buf, alpha_buf, beta_buf,
                      alpha_w_buf, alpha_w_off, beta_w_buf, beta_w_off,
                      lw.ssm_alpha_qw.in_dim, lw.ssm_alpha_qw.out_dim)
                  else
                    encode_gemv(proj_enc, gemv_pipeline_for(lw.ssm_alpha_qw).not_nil!, pre_norm_buf, alpha_buf, alpha_w_buf, alpha_w_off, lw.ssm_alpha_qw.in_dim, lw.ssm_alpha_qw.out_dim)
                    encode_gemv(proj_enc, gemv_pipeline_for(lw.ssm_beta_qw).not_nil!, pre_norm_buf, beta_buf, beta_w_buf, beta_w_off, lw.ssm_beta_qw.in_dim, lw.ssm_beta_qw.out_dim)
                  end
                  proj_enc.end_encoding
                end

                Profile.trace("rec.convshift") do
                  if use_conv_shift_fused
                    conv_enc = ML::Metal::ComputeEncoder.new(cmd)
                    conv_enc.set_pipeline(recurrent_conv_shift_pipeline)
                    conv_enc.set_buffer(conv_state_buf, 0, ML::Metal::BufferAccess::ReadWrite)
                    conv_enc.set_buffer(rec_qkv_buf, 1)
                    conv_enc.set_buffer(conv_w_buf, 2)
                    conv_enc.set_buffer(rec_q_buf, 3, ML::Metal::BufferAccess::Write)
                    conv_enc.set_buffer(rec_k_buf, 4, ML::Metal::BufferAccess::Write)
                    conv_enc.set_buffer(rec_v_buf, 5, ML::Metal::BufferAccess::Write)
                    conv_enc.set_value(hp.ssm_group_count.to_u32, 6)
                    conv_enc.set_value(hp.ssm_time_step_rank.to_u32, 7)
                    conv_enc.set_value(hp.ssm_state_size.to_u32, 8)
                    conv_enc.set_value(hp.ssm_conv_kernel.to_u32, 9)
                    conv_enc.dispatch_1d(rec_qkv_dim, 256)
                    conv_enc.end_encoding
                  else
                    conv_enc = ML::Metal::ComputeEncoder.new(cmd)
                    conv_enc.set_pipeline(recurrent_conv_pipeline)
                    conv_enc.set_buffer(conv_state_buf, 0)
                    conv_enc.set_buffer(rec_qkv_buf, 1)
                    conv_enc.set_buffer(conv_w_buf, 2)
                    conv_enc.set_buffer(rec_q_buf, 3, ML::Metal::BufferAccess::Write)
                    conv_enc.set_buffer(rec_k_buf, 4, ML::Metal::BufferAccess::Write)
                    conv_enc.set_buffer(rec_v_buf, 5, ML::Metal::BufferAccess::Write)
                    conv_enc.set_value(hp.ssm_group_count.to_u32, 6)
                    conv_enc.set_value(hp.ssm_time_step_rank.to_u32, 7)
                    conv_enc.set_value(hp.ssm_state_size.to_u32, 8)
                    conv_enc.set_value(hp.ssm_conv_kernel.to_u32, 9)
                    conv_enc.dispatch_1d(rec_qkv_dim, 256)
                    conv_enc.end_encoding

                    shift_enc = ML::Metal::ComputeEncoder.new(cmd)
                    shift_enc.set_pipeline(recurrent_shift_pipeline)
                    shift_enc.set_buffer(conv_state_buf, 0, ML::Metal::BufferAccess::ReadWrite)
                    shift_enc.set_buffer(rec_qkv_buf, 1)
                    shift_enc.set_value(rec_qkv_dim.to_u32, 2)
                    shift_enc.set_value(hp.ssm_conv_kernel.to_u32, 3)
                    shift_enc.dispatch_1d(rec_qkv_dim, 256)
                    shift_enc.end_encoding
                  end
                end

                Profile.trace("rec.qknorm") do
                  qnorm_enc = ML::Metal::ComputeEncoder.new(cmd)
                  qnorm_enc.set_pipeline(l2_heads_pipeline)
                  qnorm_enc.set_buffer(rec_q_buf, 0, ML::Metal::BufferAccess::ReadWrite)
                  qnorm_enc.set_value(hp.ssm_state_size.to_u32, 1)
                  qnorm_enc.set_value(hp.rms_eps, 2)
                  qnorm_enc.dispatch_threadgroups({hp.ssm_group_count, 1, 1}, {32, 1, 1})
                  qnorm_enc.end_encoding

                  knorm_enc = ML::Metal::ComputeEncoder.new(cmd)
                  knorm_enc.set_pipeline(l2_heads_pipeline)
                  knorm_enc.set_buffer(rec_k_buf, 0, ML::Metal::BufferAccess::ReadWrite)
                  knorm_enc.set_value(hp.ssm_state_size.to_u32, 1)
                  knorm_enc.set_value(hp.rms_eps, 2)
                  knorm_enc.dispatch_threadgroups({hp.ssm_group_count, 1, 1}, {32, 1, 1})
                  knorm_enc.end_encoding
                end

                Profile.trace("rec.ab") do
                  ab_enc = ML::Metal::ComputeEncoder.new(cmd)
                  ab_enc.set_pipeline(recurrent_ab_pipeline)
                  ab_enc.set_buffer(alpha_buf, 0)
                  ab_enc.set_buffer(beta_buf, 1, ML::Metal::BufferAccess::ReadWrite)
                  ab_enc.set_buffer(dt_bias_buf, 2)
                  ab_enc.set_buffer(ssm_a_buf, 3)
                  ab_enc.set_buffer(g_buf, 4, ML::Metal::BufferAccess::Write)
                  ab_enc.set_value(hp.ssm_time_step_rank.to_u32, 5)
                  ab_enc.dispatch_1d(hp.ssm_time_step_rank, 32)
                  ab_enc.end_encoding
                end

                Profile.trace("rec.dn") do
                  dn_enc = ML::Metal::ComputeEncoder.new(cmd)
                  dn_enc.set_pipeline(wave_dn_pipeline)
                  dn_enc.set_buffer(ssm_state_buf, 0, ML::Metal::BufferAccess::ReadWrite)
                  dn_enc.set_buffer(rec_q_buf, 1)
                  dn_enc.set_buffer(rec_k_buf, 2)
                  dn_enc.set_buffer(rec_v_buf, 3)
                  dn_enc.set_buffer(g_buf, 4)
                  dn_enc.set_buffer(beta_buf, 5)
                  dn_enc.set_buffer(rec_mid_buf, 6, ML::Metal::BufferAccess::Write)
                  dn_enc.set_value(hp.ssm_group_count.to_u32, 7)
                  dn_enc.set_value(hp.ssm_time_step_rank.to_u32, 8)
                  dn_enc.set_value(hp.ssm_state_size.to_u32, 9)
                  dn_enc.set_value(rec_scale, 10)
                  if use_dn_post_fused
                    dn_enc.set_buffer(z_buf, 11)
                    dn_enc.set_buffer(ssm_norm_buf, 12)
                    dn_enc.set_value(hp.rms_eps, 13)
                  end
                  dn_enc.dispatch_threadgroups({hp.ssm_time_step_rank, 1, 1}, {wave_dn_threadgroup_size, 1, 1})
                  dn_enc.end_encoding

                  unless use_dn_post_fused
                    post_enc = ML::Metal::ComputeEncoder.new(cmd)
                    post_enc.set_pipeline(dn_post_pipeline)
                    post_enc.set_buffer(rec_mid_buf, 0, ML::Metal::BufferAccess::ReadWrite)
                    post_enc.set_buffer(z_buf, 1)
                    post_enc.set_buffer(ssm_norm_buf, 2)
                    post_enc.set_value(hp.ssm_time_step_rank.to_u32, 3)
                    post_enc.set_value(hp.ssm_state_size.to_u32, 4)
                    post_enc.set_value(hp.rms_eps, 5)
                    post_enc.dispatch_threadgroups({hp.ssm_time_step_rank, 1, 1}, {32, 1, 1})
                    post_enc.end_encoding
                  end
                end

                Profile.trace("rec.o_proj") do
                  rec_out_enc = ML::Metal::ComputeEncoder.new(cmd)
                  encode_gemv(rec_out_enc, gemv_pipeline_for(lw.ssm_out_qw).not_nil!, rec_mid_buf, rec_attn_out_buf, ssm_out_w_buf, ssm_out_w_off, lw.ssm_out_qw.in_dim, lw.ssm_out_qw.out_dim)
                  rec_out_enc.end_encoding
                end

                Profile.trace("rec.addnorm") do
                  addnorm_enc = ML::Metal::ComputeEncoder.new(cmd)
                  encode_add_rmsnorm(addnorm_enc, src_buf, rec_attn_out_buf, post_norm_buf, residual_buf, pre_norm_buf, hidden_dim, hp.rms_eps)
                  addnorm_enc.end_encoding
                end

                Profile.trace("rec.ffn_upgate") do
                  ffn_proj_enc = ML::Metal::ComputeEncoder.new(cmd)
                  if q8_dual_gemv_candidate?(lw.ffn_gate_qw, lw.ffn_up_qw)
                    encode_gemv_q8_dual(ffn_proj_enc, pre_norm_buf, ffn_gate_buf, ffn_up_buf,
                      ffn_gate_w_buf, ffn_gate_w_off, ffn_up_w_buf, ffn_up_w_off,
                      lw.ffn_gate_qw.in_dim, lw.ffn_gate_qw.out_dim)
                  else
                    encode_gemv(ffn_proj_enc, gemv_pipeline_for(lw.ffn_gate_qw).not_nil!, pre_norm_buf, ffn_gate_buf, ffn_gate_w_buf, ffn_gate_w_off, lw.ffn_gate_qw.in_dim, lw.ffn_gate_qw.out_dim)
                    encode_gemv(ffn_proj_enc, gemv_pipeline_for(lw.ffn_up_qw).not_nil!, pre_norm_buf, ffn_up_buf, ffn_up_w_buf, ffn_up_w_off, lw.ffn_up_qw.in_dim, lw.ffn_up_qw.out_dim)
                  end
                  ffn_proj_enc.end_encoding
                end

                Profile.trace("rec.ffn_act") do
                  swiglu_enc = ML::Metal::ComputeEncoder.new(cmd)
                  ffn_act_buf = decode_swiglu_inplace_enabled? ? ffn_up_buf : ffn_comb_buf
                  swiglu_enc.set_pipeline(ffn_swiglu_pipeline)
                  swiglu_enc.set_buffer(ffn_gate_buf, 0)
                  swiglu_enc.set_buffer(ffn_up_buf, 1)
                  swiglu_enc.set_buffer(ffn_act_buf, 2, ML::Metal::BufferAccess::Write)
                  swiglu_enc.set_value(lw.ffn_gate_qw.out_dim.to_u32, 3)
                  swiglu_enc.dispatch_1d(lw.ffn_gate_qw.out_dim, 256)
                  swiglu_enc.end_encoding
                end

                if ffn_down_add_fused_enabled? && (add_pipe = gemv_add_pipeline_for(lw.ffn_down_qw))
                  Profile.trace("rec.ffn_down_add") do
                    ffn_down_enc = ML::Metal::ComputeEncoder.new(cmd)
                    ffn_act_buf = decode_swiglu_inplace_enabled? ? ffn_up_buf : ffn_comb_buf
                    encode_gemv_add(ffn_down_enc, add_pipe, ffn_act_buf, residual_buf, dst_buf,
                      ffn_down_w_buf, ffn_down_w_off, lw.ffn_down_qw.in_dim, lw.ffn_down_qw.out_dim)
                    ffn_down_enc.end_encoding
                  end
                else
                  Profile.trace("rec.ffn_down") do
                    ffn_down_enc = ML::Metal::ComputeEncoder.new(cmd)
                    ffn_act_buf = decode_swiglu_inplace_enabled? ? ffn_up_buf : ffn_comb_buf
                    encode_gemv(ffn_down_enc, gemv_pipeline_for(lw.ffn_down_qw).not_nil!, ffn_act_buf, ffn_out_buf, ffn_down_w_buf, ffn_down_w_off, lw.ffn_down_qw.in_dim, lw.ffn_down_qw.out_dim)
                    ffn_down_enc.end_encoding
                  end

                  Profile.trace("rec.add") do
                    add_enc = ML::Metal::ComputeEncoder.new(cmd)
                    add_enc.set_pipeline(add_vec_pipeline)
                    add_enc.set_buffer(residual_buf, 0)
                    add_enc.set_buffer(ffn_out_buf, 1)
                    add_enc.set_buffer(dst_buf, 2, ML::Metal::BufferAccess::Write)
                    add_enc.set_value(hidden_dim.to_u32, 3)
                    add_enc.dispatch_1d(hidden_dim, 256)
                    add_enc.end_encoding
                  end
                end
              end
            end

            src_buf, dst_buf = dst_buf, src_buf
            if chunk_layers > 0 && il + 1 < layers.size && (il + 1) % chunk_layers == 0
              cmd.commit
              pending_cmds << cmd
              cmd = ML::Metal::CommandBuffer.new(fast: wave_fast_command_buffer_enabled?)
            end
          end

          use_head_top1 = false
          if emit_head
            Profile.trace("head") do
              Profile.trace("head.norm") do
                head_norm_enc = ML::Metal::ComputeEncoder.new(cmd)
                encode_rmsnorm_vec(head_norm_enc, src_buf, output_norm_buf.not_nil!, pre_norm_buf, hidden_dim, hp.rms_eps)
                head_norm_enc.end_encoding
              end

              use_head_top1 = top1 && can_use_head_top1_fused?(output_qw)
              if use_head_top1
                Profile.trace("head.top1") do
                  head_top1_enc = ML::Metal::ComputeEncoder.new(cmd)
                  head_top1_enc.set_pipeline(output_qw.type.q8_0? ? mv8_top1_tiles_pipeline : mv6_top1_tiles_pipeline)
                  head_top1_enc.set_buffer(out_w_buf.not_nil!, 0, ML::Metal::BufferAccess::Read, offset: out_w_off)
                  head_top1_enc.set_buffer(pre_norm_buf, 1)
                  head_top1_enc.set_buffer(top1_tile_values_buf.not_nil!, 2, ML::Metal::BufferAccess::Write)
                  head_top1_enc.set_buffer(top1_tile_ids_buf.not_nil!, 3, ML::Metal::BufferAccess::Write)
                  head_top1_enc.set_value(output_qw.in_dim.to_u32, 4)
                  head_top1_enc.set_value(output_qw.out_dim.to_u32, 5)
                  head_top1_enc.dispatch_threadgroups({tile_count, 1, 1}, {output_qw.type.q8_0? ? MV_Q8_NSG * 32 : 64, 1, 1})
                  head_top1_enc.end_encoding

                  reduce_top1_enc = ML::Metal::ComputeEncoder.new(cmd)
                  reduce_top1_enc.set_pipeline(top1_reduce_tiles_pipeline)
                  reduce_top1_enc.set_buffer(top1_tile_values_buf.not_nil!, 0)
                  reduce_top1_enc.set_buffer(top1_tile_ids_buf.not_nil!, 1)
                  reduce_top1_enc.set_buffer(top1_id_buf.not_nil!, 2, ML::Metal::BufferAccess::Write)
                  reduce_top1_enc.set_buffer(top1_value_buf.not_nil!, 3, ML::Metal::BufferAccess::Write)
                  reduce_top1_enc.set_value(tile_count.to_u32, 4)
                  reduce_top1_enc.dispatch_threadgroups({1, 1, 1}, {256, 1, 1})
                  reduce_top1_enc.end_encoding
                end
              else
                Profile.trace("head.full") do
                  head_out_enc = ML::Metal::ComputeEncoder.new(cmd)
                  encode_gemv(head_out_enc, out_pipe.not_nil!, pre_norm_buf, logits_buf.not_nil!, out_w_buf.not_nil!, out_w_off, output_qw.in_dim, output_qw.out_dim)
                  head_out_enc.end_encoding
                end
              end
            end
          end

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          pending_cmds.each(&.wait)
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = if emit_head
                     use_head_top1 ? read_shared_top1(top1_id_buf.not_nil!, top1_value_buf.not_nil!) : read_shared_f32(logits_buf.not_nil!, output_qw.out_dim)
                   else
                     [] of Float32
                   end
          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_wave(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          result
        end

        # Shared GEMV machinery: takes pre-allocated weight buffer and a
        # byte offset into it (for zero-copy whole-mmap dispatch), uploads
        # x, dispatches the kernel, returns the downloaded output.
        private def self.matmul_gemv_buf(pipeline : ML::Metal::ComputePipeline,
                                         x : Array(Float32),
                                         w_buf : ML::MetalBuffer,
                                         w_offset : Int64,
                                         in_dim : Int32,
                                         out_dim : Int32,
                                         batch : Int32) : Array(Float32)
          t0 = Time.instant if Profile.enabled?
          x_buf   = Scratch.get(:mv_x,   x.size.to_i64 * sizeof(Float32))
          x_buf.write(x)

          out_buf = Scratch.get(:mv_out, (batch * out_dim).to_i64 * sizeof(Float32))

          cmd = ML::Metal::CommandBuffer.new
          enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(enc, pipeline, x_buf, out_buf, w_buf, w_offset, in_dim, out_dim, batch)
          enc.end_encoding
          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_f32(out_buf, batch * out_dim)
          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_gemv(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          result
        end

        # Q4_K-specific GEMM path (prefill batch > threshold). Takes
        # pre-allocated weight buffer and byte offset.
        private def self.matmul_q4k_gemm_buf(x : Array(Float32),
                                             w_buf : ML::MetalBuffer,
                                             w_offset : Int64,
                                             in_dim : Int32,
                                             out_dim : Int32,
                                             batch : Int32) : Array(Float32)
          x_buf   = Scratch.get(:mm_x,   x.size.to_i64 * sizeof(Float32))
          x_buf.write(x)

          out_buf = Scratch.get(:mm_out, (batch * out_dim).to_i64 * sizeof(Float32))

          cmd = ML::Metal::CommandBuffer.new
          enc = ML::Metal::ComputeEncoder.new(cmd)
          enc.set_pipeline(mm_pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
          enc.set_buffer(x_buf, 1)
          enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32,  3)
          enc.set_value(out_dim.to_u32, 4)
          enc.set_value(batch.to_u32,   5)
          enc.set_threadgroup_memory(MM_SHMEM, 0)
          grid = {
            (batch   + MM_NR1 - 1) // MM_NR1,
            (out_dim + MM_NR0 - 1) // MM_NR0,
            1,
          }
          enc.dispatch_threadgroups(grid, {MM_TG, 1, 1})
          enc.end_encoding
          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_f32(out_buf, batch * out_dim)
          if Profile.enabled?
            Profile.bump_gemm(
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
            )
          end
          result
        end

        # Q5_K/Q6_K simdgroup-matrix GEMM path. The prefill f32-output
        # kernels round through half internally to preserve the previous
        # F16-output numeric contract without a separate conversion dispatch.
        private def self.matmul_q56k_gemm_buf(pipeline : ML::Metal::ComputePipeline,
                                              x : Array(Float32),
                                              w_buf : ML::MetalBuffer,
                                              w_offset : Int64,
                                              in_dim : Int32,
                                              out_dim : Int32,
                                              batch : Int32) : Array(Float32)
          x_buf = Scratch.get(:mm56_x, x.size.to_i64 * 2_i64)
          write_shared_f16(x_buf, x)

          out_buf = Scratch.get(:mm56_out, (batch * out_dim).to_i64 * sizeof(Float32))

          cmd = ML::Metal::CommandBuffer.new
          enc = ML::Metal::ComputeEncoder.new(cmd)
          enc.set_pipeline(pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
          enc.set_buffer(x_buf, 1)
          enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32,  3)
          enc.set_value(out_dim.to_u32, 4)
          enc.set_value(batch.to_u32,   5)
          enc.set_threadgroup_memory(MM_SHMEM, 0)
          grid = {
            (batch   + MM_NR1 - 1) // MM_NR1,
            (out_dim + MM_NR0 - 1) // MM_NR0,
            1,
          }
          enc.dispatch_threadgroups(grid, {MM_TG, 1, 1})
          enc.end_encoding
          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          if Profile.enabled?
            Profile.bump_gemm(
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
            )
          end
          read_shared_f32(out_buf, batch * out_dim)
        end

        # Upload w_raw into a fresh MetalBuffer (test + one-shot paths).
        private def self.upload_weights(w_raw : Bytes) : ML::MetalBuffer
          buf = ML::MetalBuffer.new(w_raw.size.to_i64)
          buf.write_bytes(w_raw.to_unsafe, w_raw.size)
          buf
        end

        def self.matmul_q5k(x : Array(Float32),
                            w_raw : Bytes,
                            in_dim : Int32,
                            out_dim : Int32,
                            batch : Int32) : Array(Float32)
          raise "in_dim must be multiple of #{QK_K}: got #{in_dim}" unless in_dim % QK_K == 0
          raise "x size mismatch: expected #{batch * in_dim}, got #{x.size}" unless x.size == batch * in_dim
          expected_w = (in_dim // QK_K) * Q5K_BLOCK_BYTES * out_dim
          raise "w_raw size mismatch: expected #{expected_w}, got #{w_raw.size}" unless w_raw.size == expected_w
          ML::Metal::Device.init!
          buf, off = weight_slot(w_raw)
          if q56_batch_gemm_enabled? && batch > GEMM_BATCH_THRESHOLD
            matmul_q56k_gemm_buf(mm5_f32out_pipeline, x, buf, off, in_dim, out_dim, batch)
          else
            matmul_gemv_buf(mv5_pipeline, x, buf, off, in_dim, out_dim, batch)
          end
        end

        def self.matmul_q6k(x : Array(Float32),
                            w_raw : Bytes,
                            in_dim : Int32,
                            out_dim : Int32,
                            batch : Int32) : Array(Float32)
          raise "in_dim must be multiple of #{QK_K}: got #{in_dim}" unless in_dim % QK_K == 0
          raise "x size mismatch: expected #{batch * in_dim}, got #{x.size}" unless x.size == batch * in_dim
          expected_w = (in_dim // QK_K) * Q6K_BLOCK_BYTES * out_dim
          raise "w_raw size mismatch: expected #{expected_w}, got #{w_raw.size}" unless w_raw.size == expected_w
          ML::Metal::Device.init!
          buf, off = weight_slot(w_raw)
          if q56_batch_gemm_enabled? && batch > GEMM_BATCH_THRESHOLD
            matmul_q56k_gemm_buf(mm6_f32out_pipeline, x, buf, off, in_dim, out_dim, batch)
          else
            matmul_gemv_buf(mv6_pipeline, x, buf, off, in_dim, out_dim, batch)
          end
        end

        def self.matmul_q8_0(x : Array(Float32),
                             w_raw : Bytes,
                             in_dim : Int32,
                             out_dim : Int32,
                             batch : Int32) : Array(Float32)
          raise "in_dim must be multiple of #{Q8_0_QK}: got #{in_dim}" unless in_dim % Q8_0_QK == 0
          raise "x size mismatch: expected #{batch * in_dim}, got #{x.size}" unless x.size == batch * in_dim
          expected_w = (in_dim // Q8_0_QK) * Q8_0_BLOCK_BYTES * out_dim
          raise "w_raw size mismatch: expected #{expected_w}, got #{w_raw.size}" unless w_raw.size == expected_w
          ML::Metal::Device.init!
          buf, off = weight_slot(w_raw)
          matmul_gemv_buf(mv8_pipeline, x, buf, off, in_dim, out_dim, batch)
        end

        # Full-upload Q4_K matmul. Output row-major [batch, out_dim].
        # result[b, o] = Σ_k x[b, k] * W_dequant[o, k]
        #
        # w_raw: quantized weights [out_dim rows × in_dim cols],
        #        each row packed as (in_dim/256) Q4_K blocks of 144 bytes.
        def self.matmul_q4k(x : Array(Float32),
                            w_raw : Bytes,
                            in_dim : Int32,
                            out_dim : Int32,
                            batch : Int32) : Array(Float32)
          raise "in_dim must be multiple of #{QK_K}: got #{in_dim}" unless in_dim % QK_K == 0
          raise "x size mismatch: expected #{batch * in_dim}, got #{x.size}" unless x.size == batch * in_dim
          expected_w = (in_dim // QK_K) * Q4K_BLOCK_BYTES * out_dim
          raise "w_raw size mismatch: expected #{expected_w}, got #{w_raw.size}" unless w_raw.size == expected_w

          ML::Metal::Device.init!
          w_buf, w_off = weight_slot(w_raw)

          if batch > GEMM_BATCH_THRESHOLD
            matmul_q4k_gemm_buf(x, w_buf, w_off, in_dim, out_dim, batch)
          else
            matmul_gemv_buf(mv_pipeline, x, w_buf, w_off, in_dim, out_dim, batch)
          end
        end

        # Resolve `raw` into a (buffer, byte-offset) pair, preferring the
        # whole-mmap NoCopy buffer. Falls back to a one-shot upload for
        # bytes outside the registered mmap region.
        private def self.weight_slot(w_raw : Bytes) : {ML::MetalBuffer, Int64}
          if slot = mmap_slot_for(w_raw)
            slot
          else
            {upload_weights(w_raw), 0_i64}
          end
        end

        # Batched single-input GEMV: upload x once, encode a GEMV per
        # qw on the same compute encoder, commit+wait once, read all
        # outputs. Returns `nil` if any qw is a type we don't GPU-route
        # (caller falls back to per-qw `matmul`).
        #
        # All qws must share `in_dim == x.size`. Output shape is the list
        # of [out_dim] arrays in the same order as `qws`.
        def self.matmul_many(qws : Array(QuantWeight), x : Array(Float32)) : Array(Array(Float32))?
          return [] of Array(Float32) if qws.empty?
          ML::Metal::Device.init!

          # Resolve pipeline + weight buf for each qw upfront.
          # Bail out if any qw isn't Metal-routable.
          resolved = Array({ML::Metal::ComputePipeline, ML::MetalBuffer, Int64, Int32, Int32}).new(qws.size)
          qws.each do |qw|
            pipeline = case qw.type
                       when .q4_k? then mv_pipeline
                       when .q5_k? then mv5_pipeline
                       when .q6_k? then mv6_pipeline
                       when .q8_0? then mv8_pipeline
                       else
                         return nil
                       end
            buf, off = if slot = mmap_slot_for(qw.raw)
                         slot
                       else
                         {qw.fallback_metal_buffer, 0_i64}
                       end
            resolved << {pipeline, buf, off, qw.in_dim, qw.out_dim}
          end

          t0 = Time.instant if Profile.enabled?
          x_buf = Scratch.get(:mv_many_x, x.size.to_i64 * sizeof(Float32))
          x_buf.write(x)

          # Per-slot tags so concurrently-alive outputs don't alias within
          # one encoder. Up to MANY_SLOT_TAGS.size simultaneous outputs.
          out_bufs = Array(ML::MetalBuffer).new(qws.size) do |i|
            Scratch.get(MANY_SLOT_TAGS[i], qws[i].out_dim.to_i64 * sizeof(Float32))
          end

          cmd = ML::Metal::CommandBuffer.new
          enc = ML::Metal::ComputeEncoder.new(cmd)
          resolved.each_with_index do |(pipeline, w_buf, w_off, in_dim, out_dim), i|
            enc.set_pipeline(pipeline)
            enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_off)
            enc.set_buffer(x_buf, 1)
            enc.set_buffer(out_bufs[i], 2, ML::Metal::BufferAccess::Write)
            enc.set_value(in_dim.to_u32,  3)
            enc.set_value(out_dim.to_u32, 4)
            enc.set_value(1_u32,          5)
            rows_per_tg = gemv_rows_per_tg_for(pipeline)
            grid = {(out_dim + rows_per_tg - 1) // rows_per_tg, 1, 1}
            enc.dispatch_threadgroups(grid, {64, 1, 1})
          end
          enc.end_encoding
          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          results = Array(Array(Float32)).new(qws.size) { |i| read_shared_f32(out_bufs[i], qws[i].out_dim) }
          if Profile.enabled?
            t_read = Time.instant
            # One sync for N dispatches — count as ONE gemv call so
            # `total metal syncs` reflects actual barriers, not work.
            Profile.bump_gemv(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          results
        end

        # Persistent-buffer path: dispatch by QuantWeight type, using the
        # whole-mmap buffer when available (zero-copy) or falling back to
        # a per-weight upload held by the QuantWeight itself. Returns nil
        # when the type is not GPU-supported (caller falls back to CPU).
        def self.matmul(qw : QuantWeight, x : Array(Float32), batch : Int32) : Array(Float32)?
          ML::Metal::Device.init!

          buf, off = if slot = mmap_slot_for(qw.raw)
                       slot
                     else
                       {qw.fallback_metal_buffer, 0_i64}
                     end

          case qw.type
          when .q4_k?
            if batch > GEMM_BATCH_THRESHOLD
              matmul_q4k_gemm_buf(x, buf, off, qw.in_dim, qw.out_dim, batch)
            else
              matmul_gemv_buf(mv_pipeline, x, buf, off, qw.in_dim, qw.out_dim, batch)
            end
          when .q5_k?
            if q56_batch_gemm_enabled? && batch > GEMM_BATCH_THRESHOLD
              matmul_q56k_gemm_buf(mm5_pipeline, x, buf, off, qw.in_dim, qw.out_dim, batch)
            else
              matmul_gemv_buf(mv5_pipeline, x, buf, off, qw.in_dim, qw.out_dim, batch)
            end
          when .q6_k?
            if q56_batch_gemm_enabled? && batch > GEMM_BATCH_THRESHOLD
              matmul_q56k_gemm_buf(mm6_pipeline, x, buf, off, qw.in_dim, qw.out_dim, batch)
            else
              matmul_gemv_buf(mv6_pipeline, x, buf, off, qw.in_dim, qw.out_dim, batch)
            end
          when .q8_0?
            matmul_gemv_buf(mv8_pipeline, x, buf, off, qw.in_dim, qw.out_dim, batch)
          else
            nil
          end
        end
      {% end %}
    end
  end
end
