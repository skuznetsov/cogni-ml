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
      Q4K_BLOCK_BYTES = 144
      Q5K_BLOCK_BYTES = 176
      Q6K_BLOCK_BYTES = 210
      QK_K            =  256

      # GEMV (decode) tiling — same dispatch for Q4_K/Q5_K/Q6_K.
      MV_NSG = 2   # simdgroups per threadgroup (must match kernel)
      MV_NR0 = 2   # output rows per simdgroup   (must match kernel)

      # GEMM (prefill) tiling — Q4_K only for now.
      MM_NR0    =    64
      MM_NR1    =    32
      MM_TG     =   128   # threads per threadgroup (4 simdgroups × 32)
      MM_SHMEM  = 12288   # bytes: 2 × (MM_SA_SIZE + MM_SB_SIZE) = 2 × 6144

      # Above this batch, use GEMM. At or below, GEMV is faster.
      GEMM_BATCH_THRESHOLD = 8

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
        DELTA_NET_SOURCE = {{ read_file("#{__DIR__}/kernels/delta_net.metal") }}

        @@mv_pipeline   : ML::Metal::ComputePipeline?
        @@mm_pipeline   : ML::Metal::ComputePipeline?
        @@mv5_pipeline  : ML::Metal::ComputePipeline?
        @@mv6_pipeline  : ML::Metal::ComputePipeline?
        @@dn_pipeline   : ML::Metal::ComputePipeline?

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

        # Lazy compile and cache pipelines on first use.
        private def self.mv_pipeline : ML::Metal::ComputePipeline
          @@mv_pipeline ||= ML::Metal::PipelineCache.get("simd_mv_q4k_f32") {
            ML::Metal::ComputePipeline.new("simd_mv_q4k_f32", GEMM_Q4K_SOURCE)
          }
        end

        private def self.mm_pipeline : ML::Metal::ComputePipeline
          @@mm_pipeline ||= ML::Metal::PipelineCache.get("simd_mm_q4k_f32") {
            ML::Metal::ComputePipeline.new("simd_mm_q4k_f32", GEMM_Q4K_SOURCE)
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

        private def self.dn_pipeline : ML::Metal::ComputePipeline
          @@dn_pipeline ||= ML::Metal::PipelineCache.get("delta_net_step") {
            ML::Metal::ComputePipeline.new("delta_net_step", DELTA_NET_SOURCE)
          }
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

          q_buf  = ML::MetalBuffer.new(q_conv.size.to_i64 * sizeof(Float32))
          k_buf  = ML::MetalBuffer.new(k_conv.size.to_i64 * sizeof(Float32))
          v_buf  = ML::MetalBuffer.new(v_conv.size.to_i64 * sizeof(Float32))
          g_buf  = ML::MetalBuffer.new(ghead.size.to_i64 * sizeof(Float32))
          b_buf  = ML::MetalBuffer.new(beta.size.to_i64  * sizeof(Float32))
          out_buf = ML::MetalBuffer.new((h_v * s).to_i64 * sizeof(Float32))

          q_buf.write(q_conv)
          k_buf.write(k_conv)
          v_buf.write(v_conv)
          g_buf.write(ghead)
          b_buf.write(beta)

          cmd = ML::Metal::CommandBuffer.new
          enc = ML::Metal::ComputeEncoder.new(cmd)
          enc.set_pipeline(dn_pipeline)
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
          enc.dispatch_threadgroups({h_v, 1, 1}, {32, 1, 1})
          enc.end_encoding
          cmd.commit_and_wait

          out_buf.read(h_v * s)
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
          x_buf = ML::MetalBuffer.new(x.size.to_i64 * sizeof(Float32))
          x_buf.write(x)

          out_buf = ML::MetalBuffer.new((batch * out_dim).to_i64 * sizeof(Float32))

          cmd = ML::Metal::CommandBuffer.new
          enc = ML::Metal::ComputeEncoder.new(cmd)
          enc.set_pipeline(pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
          enc.set_buffer(x_buf, 1)
          enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32,  3)
          enc.set_value(out_dim.to_u32, 4)
          enc.set_value(batch.to_u32,   5)
          rows_per_tg = MV_NSG * MV_NR0
          grid = {(out_dim + rows_per_tg - 1) // rows_per_tg, batch, 1}
          enc.dispatch_threadgroups(grid, {32 * MV_NSG, 1, 1})
          enc.end_encoding
          cmd.commit_and_wait

          out_buf.read(batch * out_dim)
        end

        # Q4_K-specific GEMM path (prefill batch > threshold). Takes
        # pre-allocated weight buffer and byte offset.
        private def self.matmul_q4k_gemm_buf(x : Array(Float32),
                                             w_buf : ML::MetalBuffer,
                                             w_offset : Int64,
                                             in_dim : Int32,
                                             out_dim : Int32,
                                             batch : Int32) : Array(Float32)
          x_buf = ML::MetalBuffer.new(x.size.to_i64 * sizeof(Float32))
          x_buf.write(x)

          out_buf = ML::MetalBuffer.new((batch * out_dim).to_i64 * sizeof(Float32))

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
          cmd.commit_and_wait

          out_buf.read(batch * out_dim)
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
          matmul_gemv_buf(mv5_pipeline, x, buf, off, in_dim, out_dim, batch)
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
          matmul_gemv_buf(mv6_pipeline, x, buf, off, in_dim, out_dim, batch)
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
            matmul_gemv_buf(mv5_pipeline, x, buf, off, qw.in_dim, qw.out_dim, batch)
          when .q6_k?
            matmul_gemv_buf(mv6_pipeline, x, buf, off, qw.in_dim, qw.out_dim, batch)
          else
            nil
          end
        end
      {% end %}
    end
  end
end
