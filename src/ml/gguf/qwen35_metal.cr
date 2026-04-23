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

        @@mv_pipeline   : ML::Metal::ComputePipeline?
        @@mm_pipeline   : ML::Metal::ComputePipeline?
        @@mv5_pipeline  : ML::Metal::ComputePipeline?
        @@mv6_pipeline  : ML::Metal::ComputePipeline?

        def self.available? : Bool
          ML::Metal::Device.init!
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

        # GEMV wrapper used by Q5_K and Q6_K (GEMM path not yet implemented).
        private def self.matmul_gemv(pipeline : ML::Metal::ComputePipeline,
                                     x : Array(Float32),
                                     w_raw : Bytes,
                                     in_dim : Int32,
                                     out_dim : Int32,
                                     batch : Int32) : Array(Float32)
          ML::Metal::Device.init!

          x_buf = ML::MetalBuffer.new(x.size.to_i64 * sizeof(Float32))
          x_buf.write(x)

          w_buf = ML::MetalBuffer.new(w_raw.size.to_i64)
          w_buf.write_bytes(w_raw.to_unsafe, w_raw.size)

          out_buf = ML::MetalBuffer.new((batch * out_dim).to_i64 * sizeof(Float32))

          cmd = ML::Metal::CommandBuffer.new
          enc = ML::Metal::ComputeEncoder.new(cmd)
          enc.set_pipeline(pipeline)
          enc.set_buffer(w_buf, 0)
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

        def self.matmul_q5k(x : Array(Float32),
                            w_raw : Bytes,
                            in_dim : Int32,
                            out_dim : Int32,
                            batch : Int32) : Array(Float32)
          raise "in_dim must be multiple of #{QK_K}: got #{in_dim}" unless in_dim % QK_K == 0
          raise "x size mismatch: expected #{batch * in_dim}, got #{x.size}" unless x.size == batch * in_dim
          expected_w = (in_dim // QK_K) * Q5K_BLOCK_BYTES * out_dim
          raise "w_raw size mismatch: expected #{expected_w}, got #{w_raw.size}" unless w_raw.size == expected_w
          matmul_gemv(mv5_pipeline, x, w_raw, in_dim, out_dim, batch)
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
          matmul_gemv(mv6_pipeline, x, w_raw, in_dim, out_dim, batch)
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

          x_buf = ML::MetalBuffer.new(x.size.to_i64 * sizeof(Float32))
          x_buf.write(x)

          w_buf = ML::MetalBuffer.new(w_raw.size.to_i64)
          w_buf.write_bytes(w_raw.to_unsafe, w_raw.size)

          out_buf = ML::MetalBuffer.new((batch * out_dim).to_i64 * sizeof(Float32))

          use_gemm = batch > GEMM_BATCH_THRESHOLD
          cmd = ML::Metal::CommandBuffer.new
          enc = ML::Metal::ComputeEncoder.new(cmd)

          if use_gemm
            enc.set_pipeline(mm_pipeline)
            enc.set_buffer(w_buf, 0)
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
          else
            enc.set_pipeline(mv_pipeline)
            enc.set_buffer(w_buf, 0)
            enc.set_buffer(x_buf, 1)
            enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
            enc.set_value(in_dim.to_u32,  3)
            enc.set_value(out_dim.to_u32, 4)
            enc.set_value(batch.to_u32,   5)
            rows_per_tg = MV_NSG * MV_NR0
            grid = {(out_dim + rows_per_tg - 1) // rows_per_tg, batch, 1}
            enc.dispatch_threadgroups(grid, {32 * MV_NSG, 1, 1})
          end

          enc.end_encoding
          cmd.commit_and_wait

          out_buf.read(batch * out_dim)
        end
      {% end %}
    end
  end
end
