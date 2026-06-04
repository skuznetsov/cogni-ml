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
      Q4K_BLOCK_BYTES    = 144
      Q5K_BLOCK_BYTES    = 176
      Q6K_BLOCK_BYTES    = 210
      Q8_0_BLOCK_BYTES   =  34
      IQ4_NL_BLOCK_BYTES =  18
      QK_K               = 256
      Q8_0_QK            =  32
      IQ4_NL_QK          =  32

      # GEMV (decode) tiling — must match the quant-specific kernels.
      MV_Q4_NSG             =  2
      MV_Q4_NR0             =  2
      MV_Q5_NSG             =  2
      MV_Q5_NR0             =  1
      MV_Q6_NSG             =  2
      MV_Q6_NR0             =  1
      MV_Q8_NSG             =  4
      MV_Q8_NR0             =  1
      MV_IQ4_NL_NSG         =  2
      MV_IQ4_NL_NR0         =  2
      MV_F32_NSG            =  4
      MV_F32_NR0            =  1
      HEAD_TOP1_ROWS_PER_TG = 12

      # GEMM (prefill) tiling — Q4_K only for now.
      MM_NR0      =    64
      MM_NR1      =    32
      MM_TG       =   128 # threads per threadgroup (4 simdgroups × 32)
      MM_SHMEM    = 12288 # bytes: 2 × (MM_SA_SIZE + MM_SB_SIZE) = 2 × 6144
      MM48_NR1    =    48
      MM48_TG     =   192 # threads per threadgroup (6 simdgroups × 32)
      MM48_SHMEM  = 14336 # bytes: 2 × (4096 + 3072), larger than 64×48 f32 edge scratch
      MM64_NR1    =    64
      MM64_TG     =   256 # threads per threadgroup (8 simdgroups × 32)
      MM64_SHMEM  = 16384 # bytes: 2 × (MM64_SA_SIZE + MM64_SB_SIZE)
      MM80_NR1    =    80
      MM80_TG     =   320 # threads per threadgroup (10 simdgroups × 32)
      MM80_SHMEM  = 20480 # bytes: max double-buffered tile and 64×80 f32 edge scratch
      MM96_NR1    =    96
      MM96_TG     =   384 # threads per threadgroup (12 simdgroups × 32)
      MM96_SHMEM  = 24576 # bytes: max double-buffered tile and 64×96 f32 edge scratch
      MM112_NR1   =   112
      MM112_TG    =   448 # threads per threadgroup (14 simdgroups × 32)
      MM112_SHMEM = 28672 # bytes: max double-buffered tile and 64×112 f32 edge scratch
      Q4_TENSOR_NR1   =   128
      Q4_TENSOR_TG    =   128 # 4 simdgroups × 32, matches simd_mm_q4k_tensor_f32out
      Q4_TENSOR_SHMEM =  4096 # one 64×32 H16 dequantized A tile

      # Above this batch, use GEMM. At or below, GEMV is faster.
      # The default is deliberately conservative; small speculative verifier
      # chunks can lower it via env for bounded A/B without changing the normal
      # prefill/decode policy.
      GEMM_BATCH_THRESHOLD = ENV["QWEN35_GEMM_BATCH_THRESHOLD"]?.try(&.to_i?) || 8

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

        def self.bf16_gemv(w_raw : Bytes,
                           in_dim : Int32,
                           out_dim : Int32,
                           x : Array(Float32)) : Array(Float32)
          raise "Metal disabled (cpu_only)"
        end

        def self.bf16_q_gate_gemv(w_raw : Bytes,
                                  in_dim : Int32,
                                  q_dim : Int32,
                                  head_dim : Int32,
                                  x : Array(Float32)) : Array(Float32)
          raise "Metal disabled (cpu_only)"
        end

        def self.clear_bf16_weight_cache : Nil
        end

        def self.project_top1_no_norm(out_qw : QuantWeight,
                                      x : Array(Float32)) : Array(Float32)?
          nil
        end

        def self.project_top2_no_norm(out_qw : QuantWeight,
                                      x : Array(Float32)) : Array(Float32)?
          nil
        end

        def self.ffn_project_residual_top2(x : Array(Float32),
                                           residual : Array(Float32),
                                           gate_qw : QuantWeight,
                                           up_qw : QuantWeight,
                                           down_qw : QuantWeight,
                                           norm_weight : Array(Float32),
                                           head_qw : QuantWeight,
                                           eps : Float32) : NamedTuple(hidden: Array(Float32), top2: Array(Float32))?
          nil
        end

        def self.rmsnorm_project_top1_allowed_ids(x : Array(Float32),
                                                  norm_weight : Array(Float32),
                                                  out_qw : QuantWeight,
                                                  eps : Float32,
                                                  allowed_ids : Array(Int32)) : Array(Float32)?
          nil
        end

        def self.mtp_one_token_hidden_from_fc_in(fc_in : Array(Float32),
                                                 fc_raw : Bytes,
                                                 v_raw : Bytes,
                                                 q_raw : Bytes,
                                                 o_raw : Bytes,
                                                 ffn_gate_raw : Bytes,
                                                 ffn_up_raw : Bytes,
                                                 ffn_down_raw : Bytes,
                                                 input_norm : Array(Float32),
                                                 post_norm : Array(Float32),
                                                 final_norm : Array(Float32),
                                                 hidden_dim : Int32,
                                                 q_dim : Int32,
                                                 kv_dim : Int32,
                                                 ffn_dim : Int32,
                                                 n_head : Int32,
                                                 n_head_kv : Int32,
                                                 head_dim : Int32,
                                                 eps : Float32) : Array(Float32)?
          nil
        end
      {% else %}
        GEMM_Q4K_SOURCE  = {{ read_file("#{__DIR__}/kernels/gemm_q4k.metal") }}
        GEMM_Q56K_SOURCE = {{ read_file("#{__DIR__}/kernels/gemm_q56k.metal") }}
        GEMM_MM_SOURCE   = {{ read_file("#{__DIR__}/kernels/gemm_mm.metal") }}
        DELTA_NET_SOURCE = {{ read_file("#{__DIR__}/kernels/delta_net.metal") }}
        FFN_UPDOWN_Q8_SOURCE = {{ read_file("#{__DIR__}/kernels/ffn_updown_q8.metal") }}
        ATTN_DECODE_SOURCE = {{ read_file("#{__DIR__}/kernels/attn_decode_qwen35.metal") }}
        FFN_SOURCE = {{ read_file("#{__DIR__}/kernels/ffn_qwen35.metal") }}
        RECURRENT_SOURCE = {{ read_file("#{__DIR__}/kernels/recurrent_qwen35.metal") }}
        FULLATTN_SOURCE = {{ read_file("#{__DIR__}/kernels/fullattn_qwen35.metal") }}
        MTP_SOURCE = {{ read_file("#{__DIR__}/kernels/mtp_qwen35.metal") }}

        @@mv_pipeline   : ML::Metal::ComputePipeline?
        @@mv_add_pipeline : ML::Metal::ComputePipeline?
        @@embed_q4k_pipeline : ML::Metal::ComputePipeline?
        @@embed_q6k_pipeline : ML::Metal::ComputePipeline?
        @@mm_pipeline   : ML::Metal::ComputePipeline?
        @@mm_h16_pipeline : ML::Metal::ComputePipeline?
        @@mm_h16_b48_pipeline : ML::Metal::ComputePipeline?
        @@mm_h16_b64_pipeline : ML::Metal::ComputePipeline?
        @@mm_h16_b64_swiglu_pipeline : ML::Metal::ComputePipeline?
        @@mm_h16_b64_gelu_mul_pipeline : ML::Metal::ComputePipeline?
        @@mm_h16_b64_swiglu_h16_pipeline : ML::Metal::ComputePipeline?
        @@mm_h16_b80_pipeline : ML::Metal::ComputePipeline?
        @@mm_h16_b96_pipeline : ML::Metal::ComputePipeline?
        @@mm_h16_b112_pipeline : ML::Metal::ComputePipeline?
        @@mm_q4_tensor_f32out_pipeline : ML::Metal::ComputePipeline?
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
        @@mv_iq4_nl_pipeline : ML::Metal::ComputePipeline?
        @@mv_f32_pipeline : ML::Metal::ComputePipeline?
        @@mv8_top1_tiles_pipeline : ML::Metal::ComputePipeline?
        @@mv8_top2_tiles_pipeline : ML::Metal::ComputePipeline?
        @@mv6_top1_tiles_pipeline : ML::Metal::ComputePipeline?
        @@mv6_top1_allowed_tiles_pipeline : ML::Metal::ComputePipeline?
        @@mv6_top2_tiles_pipeline : ML::Metal::ComputePipeline?
        @@mv6_top1_tiles_batch_pipeline : ML::Metal::ComputePipeline?
        @@top1_reduce_tiles_pipeline : ML::Metal::ComputePipeline?
        @@top2_reduce_tiles_pipeline : ML::Metal::ComputePipeline?
        @@top1_reduce_tiles_batch_pipeline : ML::Metal::ComputePipeline?
        @@top1_reduce_f16_rows_pipeline : ML::Metal::ComputePipeline?
        @@top2_reduce_f16_rows_pipeline : ML::Metal::ComputePipeline?
        @@store_top1_token_id_pipeline : ML::Metal::ComputePipeline?
        @@bf16_gemv_pipeline : ML::Metal::ComputePipeline?
        @@bf16_q_gate_gemv_pipeline : ML::Metal::ComputePipeline?
        @@mtp_attn_gate_pipeline : ML::Metal::ComputePipeline?
        @@dn_pipeline   : ML::Metal::ComputePipeline?
        @@dn128_pipeline : ML::Metal::ComputePipeline?
        @@dn128_fused_pipeline : ML::Metal::ComputePipeline?
        @@dn128_fused_post_pipeline : ML::Metal::ComputePipeline?
        @@dn128_chunk_fused_pipeline : ML::Metal::ComputePipeline?
        @@dn128_chunk_rowwise_pipeline : ML::Metal::ComputePipeline?
        @@dn128_chunk_rowwise_checkpoint_pipeline : ML::Metal::ComputePipeline?
        @@dn128_chunk_rowwise_rollback_log_pipeline : ML::Metal::ComputePipeline?
        @@dn128_rollback_rowwise_pipeline : ML::Metal::ComputePipeline?
        @@dn_post_pipeline : ML::Metal::ComputePipeline?
        @@dn_post_chunk_pipeline : ML::Metal::ComputePipeline?
        @@dn_post_chunk_h16_pipeline : ML::Metal::ComputePipeline?
        @@lowrank_project_coeffs_pipeline : ML::Metal::ComputePipeline?
        @@lowrank_project_coeffs_chunk_pipeline : ML::Metal::ComputePipeline?
        @@lowrank_project_state_pipeline : ML::Metal::ComputePipeline?
        @@lowrank_reconstruct_state_pipeline : ML::Metal::ComputePipeline?
        @@lowrank_delta_pipeline : ML::Metal::ComputePipeline?
        @@lowrank_delta_chunk_pipeline : ML::Metal::ComputePipeline?
        @@ffn_pca_updown_coeffs_pipeline : ML::Metal::ComputePipeline?
        @@ffn_pca_updown_out_pipeline : ML::Metal::ComputePipeline?
        @@ffn_pca_updown_fused_pipeline : ML::Metal::ComputePipeline?
        @@ffn_pca_updown_fused_rows_pipeline : ML::Metal::ComputePipeline?
        @@ffn_pca_updown_fused_rows_q8_pipeline : ML::Metal::ComputePipeline?
        @@attn_pipeline : ML::Metal::ComputePipeline?
        @@attn_gqa4_pipeline : ML::Metal::ComputePipeline?
        @@attn_splitk_stage1_pipeline : ML::Metal::ComputePipeline?
        @@attn_splitk_stage2_pipeline : ML::Metal::ComputePipeline?
        @@f32_to_f16_pipeline : ML::Metal::ComputePipeline?
        @@f16_to_f32_pipeline : ML::Metal::ComputePipeline?
        @@ffn_swiglu_pipeline : ML::Metal::ComputePipeline?
        @@ffn_swiglu_h16_pipeline : ML::Metal::ComputePipeline?
        @@add_rmsnorm_pipeline : ML::Metal::ComputePipeline?
        @@add_rmsnorm_rows_pipeline : ML::Metal::ComputePipeline?
        @@add_rmsnorm_rows_h16_pipeline : ML::Metal::ComputePipeline?
        @@add_vec_pipeline : ML::Metal::ComputePipeline?
        @@rmsnorm_vec_pipeline : ML::Metal::ComputePipeline?
        @@rmsnorm_rows_pipeline : ML::Metal::ComputePipeline?
        @@rmsnorm_rows_f32_h16_pipeline : ML::Metal::ComputePipeline?
        @@recurrent_ab_pipeline : ML::Metal::ComputePipeline?
        @@recurrent_ab_chunk_pipeline : ML::Metal::ComputePipeline?
        @@recurrent_conv_pipeline : ML::Metal::ComputePipeline?
        @@recurrent_shift_pipeline : ML::Metal::ComputePipeline?
        @@recurrent_conv_shift_pipeline : ML::Metal::ComputePipeline?
        @@recurrent_conv_shift_chunk_pipeline : ML::Metal::ComputePipeline?
        @@recurrent_conv_shift_chunk_checkpoint_pipeline : ML::Metal::ComputePipeline?
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
        @@attn_rows_sg4_pipeline : ML::Metal::ComputePipeline?
        @@attn_rows_sg4_pregate_pipeline : ML::Metal::ComputePipeline?

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
          @@group_upload_bytes = Hash(String, Int64).new(0_i64)
          @@group_read_bytes = Hash(String, Int64).new(0_i64)
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
            @@group_upload_bytes.clear
            @@group_read_bytes.clear
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

          def self.gemv_wait_ms : Float64
            @@gemv_wait_ns / 1_000_000.0
          end

          def self.gemm_wait_ms : Float64
            @@gemm_wait_ns / 1_000_000.0
          end

          def self.matmul_wait_ms : Float64
            gemv_wait_ms + gemm_wait_ms
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

          def self.bump_group_transfer(label : String, upload_bytes : Int64, read_bytes : Int64) : Nil
            return unless @@enabled
            @@group_upload_bytes[label] += upload_bytes
            @@group_read_bytes[label] += read_bytes
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
              group_labels = Set(String).new
              @@group_counts.keys.each { |name| group_labels << name }
              @@group_upload_bytes.keys.each { |name| group_labels << name }
              @@group_read_bytes.keys.each { |name| group_labels << name }
              unless group_labels.empty?
                total_group_upload = @@group_upload_bytes.values.sum
                total_group_read = @@group_read_bytes.values.sum
                s << sprintf("  group boundary transfer: upload %.2f MiB  readback %.2f MiB\n",
                             total_group_upload / 1_048_576.0,
                             total_group_read / 1_048_576.0)
                s << "  grouped command buffers:\n"
                group_labels.to_a.sort_by { |name| {-@@group_wait_ns[name], name} }.each do |name|
                  s << sprintf("    %-18s %4d calls  encode %.2f ms  wait %.2f ms  read %.2f ms  upload %.2f MiB  readback %.2f MiB\n",
                               name, @@group_counts[name],
                               @@group_encode_ns[name] / 1_000_000.0,
                               @@group_wait_ns[name] / 1_000_000.0,
                               @@group_read_ns[name] / 1_000_000.0,
                               @@group_upload_bytes[name] / 1_048_576.0,
                               @@group_read_bytes[name] / 1_048_576.0)
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

        @@lane_command_queues = {} of String => ML::Metal::CommandQueue

        private def self.lane_command_queue(name : String) : ML::Metal::CommandQueue
          @@lane_command_queues[name] ||= ML::Metal::CommandQueue.new
        end

        def self.decode_wave_command_buffer(command_queue_name : String? = nil) : ML::Metal::CommandBuffer
          ML::Metal::Device.init!
          cmd_queue = command_queue_name ? lane_command_queue(command_queue_name.not_nil!) : nil
          ML::Metal::CommandBuffer.new(queue: cmd_queue, fast: wave_fast_command_buffer_enabled?)
        end

        # ── Phase 4.2 scratch pool ─────────────────────────────────────
        # Persistent per-dispatch scratch buffers keyed by (tag, bytes).
        # Each dispatch-site tag names a buffer slot that must not alias
        # another buffer alive in the same command buffer. Sizes vary
        # across call sites and layers, so key includes size; reuse
        # within a key is a cache hit. Buffers live until `clear`.
        module Scratch
          @@pool : Hash({Symbol, Int64}, ML::MetalBuffer) = {} of {Symbol, Int64} => ML::MetalBuffer
          @@pool_s : Hash({String, Int64}, ML::MetalBuffer) = {} of {String, Int64} => ML::MetalBuffer
          @@mutex = Mutex.new
          @@hits   = 0_i64
          @@misses = 0_i64
          @@fresh_collectors = {} of UInt64 => Array(Array(ML::MetalBuffer))
          @@namespaces = {} of UInt64 => Array(String)

          private def self.thread_key : UInt64
            Thread.current.object_id
          end

          private def self.fresh_stack : Array(Array(ML::MetalBuffer))
            key = thread_key
            @@mutex.synchronize { @@fresh_collectors[key] ||= [] of Array(ML::MetalBuffer) }
          end

          private def self.namespace_stack : Array(String)
            key = thread_key
            @@mutex.synchronize { @@namespaces[key] ||= [] of String }
          end

          def self.with_fresh(&)
            collector = [] of ML::MetalBuffer
            fresh_stack << collector
            yield collector
          ensure
            fresh_stack.pop?
          end

          private def self.fresh_buffer(byte_size : Int64) : ML::MetalBuffer
            @@mutex.synchronize { @@misses += 1 }
            buf = ML::MetalBuffer.new(byte_size)
            if collector = fresh_stack.last?
              collector << buf
            end
            buf
          end

          def self.with_namespace(namespace : String, &)
            namespace_stack << namespace
            yield
          ensure
            namespace_stack.pop?
          end

          private def self.get_string_raw(tag : String, byte_size : Int64) : ML::MetalBuffer
            if ENV["QWEN35_SCRATCH_OFF"]? == "1" || !fresh_stack.empty?
              return fresh_buffer(byte_size)
            end
            key = {tag, byte_size}
            @@mutex.synchronize do
              if buf = @@pool_s[key]?
                @@hits += 1
                return buf
              end
              @@misses += 1
              buf = ML::MetalBuffer.new(byte_size)
              @@pool_s[key] = buf
              buf
            end
          end

          def self.get(tag : Symbol, byte_size : Int64) : ML::MetalBuffer
            if namespace = namespace_stack.last?
              return get_string_raw("#{namespace}:#{tag}", byte_size)
            end
            # A/B gate — when set, always allocate fresh (emulates pre-4.2).
            if ENV["QWEN35_SCRATCH_OFF"]? == "1" || !fresh_stack.empty?
              return fresh_buffer(byte_size)
            end
            key = {tag, byte_size}
            @@mutex.synchronize do
              if buf = @@pool[key]?
                @@hits += 1
                return buf
              end
              @@misses += 1
              buf = ML::MetalBuffer.new(byte_size)
              @@pool[key] = buf
              buf
            end
          end

          def self.get(tag : String, byte_size : Int64) : ML::MetalBuffer
            if namespace = namespace_stack.last?
              tag = "#{namespace}:#{tag}"
            end
            get_string_raw(tag, byte_size)
          end

          def self.stats : {Int64, Int64}
            @@mutex.synchronize { {@@hits, @@misses} }
          end

          def self.clear : Nil
            @@mutex.synchronize do
              @@pool.clear
              @@pool_s.clear
              @@fresh_collectors.clear
              @@namespaces.clear
              @@hits = @@misses = 0_i64
            end
          end
        end

        module ConstCache
          @@written : Hash(String, Bool) = {} of String => Bool
          @@mutex = Mutex.new

          def self.write_once(tag : String, buf : ML::MetalBuffer, data : Array(Float32)) : Nil
            key = "#{tag}:#{buf.handle.address}:#{buf.size}:#{data.to_unsafe.address}:#{data.size}"
            @@mutex.synchronize do
              return if @@written[key]?
              buf.write(data)
              @@written[key] = true
            end
          end

          def self.write_zero_f32_once(tag : String, buf : ML::MetalBuffer, count : Int32) : Nil
            key = "#{tag}:#{buf.handle.address}:#{buf.size}:zero:#{count}"
            @@mutex.synchronize do
              return if @@written[key]?
              buf.contents.as(Pointer(UInt8)).clear(count.to_i64 * sizeof(Float32))
              @@written[key] = true
            end
          end

          def self.clear : Nil
            @@mutex.synchronize { @@written.clear }
          end
        end

        # Whole-mmap MetalBuffer. Registered once per model load via
        # `register_mmap`. All weights whose `raw` bytes are slices
        # inside this region dispatch against it with a byte offset —
        # true zero-copy on Apple Silicon unified memory.
        @@mmap_base_addr : UInt64 = 0_u64
        @@mmap_size      : Int64  = 0_i64
        @@mmap_buf       : ML::MetalBuffer? = nil
        @@bf16_weight_buffers = {} of String => ML::MetalBuffer
        @@bf16_weight_mutex = Mutex.new

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

        def self.embedding_q4k_from_token_id(token_embd_qw : QuantWeight,
                                             token_id : Int32) : Array(Float32)?
          return nil unless token_embd_qw.type.q4_k?
          token_buf = ML::MetalBuffer.new(sizeof(UInt32).to_i64)
          token_buf.contents.as(Pointer(UInt32)).value = token_id.to_u32
          out_buf = ML::MetalBuffer.new(token_embd_qw.in_dim.to_i64 * sizeof(Float32))
          embedding_q4k_from_token_id_buf(token_embd_qw, token_buf, out_buf)
          out_buf.read(token_embd_qw.in_dim)
        end

        def self.embedding_q4k_from_token_id_buf(token_embd_qw : QuantWeight,
                                                 token_ids_buf : ML::MetalBuffer,
                                                 out_buf : ML::MetalBuffer,
                                                 token_index : Int32 = 0,
                                                 command_queue_name : String? = nil) : Nil
          raise "embedding_q4k_from_token_id_buf requires Q4_K token embeddings" unless token_embd_qw.type.q4_k?
          raise "embedding dim #{token_embd_qw.in_dim} must be divisible by #{QK_K}" unless token_embd_qw.in_dim % QK_K == 0

          w_buf, w_off = weight_slot(token_embd_qw)
          cmd_queue = command_queue_name ? lane_command_queue(command_queue_name.not_nil!) : nil
          cmd = ML::Metal::CommandBuffer.new(queue: cmd_queue)
          enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_embedding_q4k_from_token_id(enc, w_buf, w_off, token_ids_buf, out_buf,
            token_embd_qw.in_dim, token_embd_qw.out_dim, token_index)
          enc.end_encoding
          cmd.commit
          cmd.wait
        end

        def self.embedding_q6k_from_token_id(token_embd_qw : QuantWeight,
                                             token_id : Int32) : Array(Float32)?
          return nil unless token_embd_qw.type.q6_k?
          token_buf = ML::MetalBuffer.new(sizeof(UInt32).to_i64)
          token_buf.contents.as(Pointer(UInt32)).value = token_id.to_u32
          out_buf = ML::MetalBuffer.new(token_embd_qw.in_dim.to_i64 * sizeof(Float32))
          embedding_q6k_from_token_id_buf(token_embd_qw, token_buf, out_buf)
          out_buf.read(token_embd_qw.in_dim)
        end

        def self.embedding_q6k_from_token_id_buf(token_embd_qw : QuantWeight,
                                                 token_ids_buf : ML::MetalBuffer,
                                                 out_buf : ML::MetalBuffer,
                                                 token_index : Int32 = 0,
                                                 command_queue_name : String? = nil) : Nil
          raise "embedding_q6k_from_token_id_buf requires Q6_K token embeddings" unless token_embd_qw.type.q6_k?
          raise "embedding dim #{token_embd_qw.in_dim} must be divisible by #{QK_K}" unless token_embd_qw.in_dim % QK_K == 0

          w_buf, w_off = weight_slot(token_embd_qw)
          cmd_queue = command_queue_name ? lane_command_queue(command_queue_name.not_nil!) : nil
          cmd = ML::Metal::CommandBuffer.new(queue: cmd_queue)
          enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_embedding_q6k_from_token_id(enc, w_buf, w_off, token_ids_buf, out_buf,
            token_embd_qw.in_dim, token_embd_qw.out_dim, token_index)
          enc.end_encoding
          cmd.commit
          cmd.wait
        end

        private def self.encode_embedding_q4k_from_token_id(enc : ML::Metal::ComputeEncoder,
                                                            w_buf : ML::MetalBuffer,
                                                            w_off : Int64,
                                                            token_ids_buf : ML::MetalBuffer,
                                                            out_buf : ML::MetalBuffer,
                                                            hidden_dim : Int32,
                                                            vocab_size : Int32,
                                                            token_index : Int32) : Nil
          enc.set_pipeline(embed_q4k_pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_off)
          enc.set_buffer(token_ids_buf, 1)
          enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_value(hidden_dim.to_u32, 3)
          enc.set_value(vocab_size.to_u32, 4)
          enc.set_value(token_index.to_u32, 5)
          enc.dispatch_1d(hidden_dim, 256)
        end

        private def self.encode_embedding_q6k_from_token_id(enc : ML::Metal::ComputeEncoder,
                                                            w_buf : ML::MetalBuffer,
                                                            w_off : Int64,
                                                            token_ids_buf : ML::MetalBuffer,
                                                            out_buf : ML::MetalBuffer,
                                                            hidden_dim : Int32,
                                                            vocab_size : Int32,
                                                            token_index : Int32) : Nil
          enc.set_pipeline(embed_q6k_pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_off)
          enc.set_buffer(token_ids_buf, 1)
          enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_value(hidden_dim.to_u32, 3)
          enc.set_value(vocab_size.to_u32, 4)
          enc.set_value(token_index.to_u32, 5)
          enc.dispatch_1d(hidden_dim, 256)
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

        private def self.embed_q4k_pipeline : ML::Metal::ComputePipeline
          @@embed_q4k_pipeline ||= ML::Metal::PipelineCache.get("embed_q4k_f32_from_token_id") {
            ML::Metal::ComputePipeline.new("embed_q4k_f32_from_token_id", GEMM_Q4K_SOURCE)
          }
        end

        private def self.embed_q6k_pipeline : ML::Metal::ComputePipeline
          @@embed_q6k_pipeline ||= ML::Metal::PipelineCache.get("embed_q6k_f32_from_token_id") {
            ML::Metal::ComputePipeline.new("embed_q6k_f32_from_token_id", GEMM_Q56K_SOURCE)
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

        private def self.mm_h16_b48_pipeline : ML::Metal::ComputePipeline
          @@mm_h16_b48_pipeline ||= ML::Metal::PipelineCache.get("simd_mm_q4k_h16_b48") {
            ML::Metal::ComputePipeline.new("simd_mm_q4k_h16_b48", GEMM_Q4K_SOURCE)
          }
        end

        private def self.mm_h16_b64_pipeline : ML::Metal::ComputePipeline
          @@mm_h16_b64_pipeline ||= ML::Metal::PipelineCache.get("simd_mm_q4k_h16_b64") {
            ML::Metal::ComputePipeline.new("simd_mm_q4k_h16_b64", GEMM_Q4K_SOURCE)
          }
        end

        private def self.mm_h16_b64_swiglu_pipeline : ML::Metal::ComputePipeline
          @@mm_h16_b64_swiglu_pipeline ||= ML::Metal::PipelineCache.get("simd_mm_q4k_h16_b64_swiglu") {
            ML::Metal::ComputePipeline.new("simd_mm_q4k_h16_b64_swiglu", GEMM_Q4K_SOURCE)
          }
        end

        private def self.mm_h16_b64_gelu_mul_pipeline : ML::Metal::ComputePipeline
          @@mm_h16_b64_gelu_mul_pipeline ||= ML::Metal::PipelineCache.get("simd_mm_q4k_h16_b64_gelu_mul") {
            ML::Metal::ComputePipeline.new("simd_mm_q4k_h16_b64_gelu_mul", GEMM_Q4K_SOURCE)
          }
        end

        private def self.mm_h16_b64_swiglu_h16_pipeline : ML::Metal::ComputePipeline
          @@mm_h16_b64_swiglu_h16_pipeline ||= ML::Metal::PipelineCache.get("simd_mm_q4k_h16_b64_swiglu_h16") {
            ML::Metal::ComputePipeline.new("simd_mm_q4k_h16_b64_swiglu_h16", GEMM_Q4K_SOURCE)
          }
        end

        private def self.mm_h16_b80_pipeline : ML::Metal::ComputePipeline
          @@mm_h16_b80_pipeline ||= ML::Metal::PipelineCache.get("simd_mm_q4k_h16_b80") {
            ML::Metal::ComputePipeline.new("simd_mm_q4k_h16_b80", GEMM_Q4K_SOURCE)
          }
        end

        private def self.mm_h16_b96_pipeline : ML::Metal::ComputePipeline
          @@mm_h16_b96_pipeline ||= ML::Metal::PipelineCache.get("simd_mm_q4k_h16_b96") {
            ML::Metal::ComputePipeline.new("simd_mm_q4k_h16_b96", GEMM_Q4K_SOURCE)
          }
        end

        private def self.mm_h16_b112_pipeline : ML::Metal::ComputePipeline
          @@mm_h16_b112_pipeline ||= ML::Metal::PipelineCache.get("simd_mm_q4k_h16_b112") {
            ML::Metal::ComputePipeline.new("simd_mm_q4k_h16_b112", GEMM_Q4K_SOURCE)
          }
        end

        private def self.mm_q4_tensor_f32out_pipeline : ML::Metal::ComputePipeline
          @@mm_q4_tensor_f32out_pipeline ||= ML::Metal::PipelineCache.get("simd_mm_q4k_tensor_f32out") {
            ML::Metal::ComputePipeline.new("simd_mm_q4k_tensor_f32out", GEMM_Q4K_SOURCE)
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

        private def self.mv_iq4_nl_pipeline : ML::Metal::ComputePipeline
          @@mv_iq4_nl_pipeline ||= ML::Metal::PipelineCache.get("simd_mv_iq4_nl_f32") {
            ML::Metal::ComputePipeline.new("simd_mv_iq4_nl_f32", GEMM_Q56K_SOURCE)
          }
        end

        private def self.mv_f32_pipeline : ML::Metal::ComputePipeline
          @@mv_f32_pipeline ||= ML::Metal::PipelineCache.get("simd_mv_f32_f32") {
            ML::Metal::ComputePipeline.new("simd_mv_f32_f32", GEMM_Q56K_SOURCE)
          }
        end

        private def self.mv6_top1_tiles_pipeline : ML::Metal::ComputePipeline
          @@mv6_top1_tiles_pipeline ||= ML::Metal::PipelineCache.get("simd_mv_q6k_top1_tiles_f32") {
            ML::Metal::ComputePipeline.new("simd_mv_q6k_top1_tiles_f32", GEMM_Q56K_SOURCE)
          }
        end

        private def self.mv6_top1_allowed_tiles_pipeline : ML::Metal::ComputePipeline
          @@mv6_top1_allowed_tiles_pipeline ||= ML::Metal::PipelineCache.get("simd_mv_q6k_top1_allowed_tiles_f32") {
            ML::Metal::ComputePipeline.new("simd_mv_q6k_top1_allowed_tiles_f32", GEMM_Q56K_SOURCE)
          }
        end

        private def self.mv6_top2_tiles_pipeline : ML::Metal::ComputePipeline
          @@mv6_top2_tiles_pipeline ||= ML::Metal::PipelineCache.get("simd_mv_q6k_top2_tiles_f32") {
            ML::Metal::ComputePipeline.new("simd_mv_q6k_top2_tiles_f32", GEMM_Q56K_SOURCE)
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

        private def self.mv8_top2_tiles_pipeline : ML::Metal::ComputePipeline
          @@mv8_top2_tiles_pipeline ||= ML::Metal::PipelineCache.get("simd_mv_q8_0_top2_tiles_f32") {
            ML::Metal::ComputePipeline.new("simd_mv_q8_0_top2_tiles_f32", GEMM_Q56K_SOURCE)
          }
        end

        private def self.top1_reduce_tiles_pipeline : ML::Metal::ComputePipeline
          @@top1_reduce_tiles_pipeline ||= ML::Metal::PipelineCache.get("qwen35_top1_reduce_tiles") {
            ML::Metal::ComputePipeline.new("qwen35_top1_reduce_tiles", GEMM_Q56K_SOURCE)
          }
        end

        private def self.top2_reduce_tiles_pipeline : ML::Metal::ComputePipeline
          @@top2_reduce_tiles_pipeline ||= ML::Metal::PipelineCache.get("qwen35_top2_reduce_tiles") {
            ML::Metal::ComputePipeline.new("qwen35_top2_reduce_tiles", GEMM_Q56K_SOURCE)
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

        private def self.store_top1_token_id_pipeline : ML::Metal::ComputePipeline
          @@store_top1_token_id_pipeline ||= ML::Metal::PipelineCache.get("qwen35_store_top1_token_id") {
            ML::Metal::ComputePipeline.new("qwen35_store_top1_token_id", GEMM_Q56K_SOURCE)
          }
        end

        private def self.bf16_gemv_pipeline : ML::Metal::ComputePipeline
          @@bf16_gemv_pipeline ||= ML::Metal::PipelineCache.get("qwen35_bf16_gemv_f32") {
            ML::Metal::ComputePipeline.new("qwen35_bf16_gemv_f32", MTP_SOURCE)
          }
        end

        private def self.bf16_q_gate_gemv_pipeline : ML::Metal::ComputePipeline
          @@bf16_q_gate_gemv_pipeline ||= ML::Metal::PipelineCache.get("qwen35_bf16_q_gate_gemv_f32") {
            ML::Metal::ComputePipeline.new("qwen35_bf16_q_gate_gemv_f32", MTP_SOURCE)
          }
        end

        private def self.mtp_attn_gate_pipeline : ML::Metal::ComputePipeline
          @@mtp_attn_gate_pipeline ||= ML::Metal::PipelineCache.get("qwen35_mtp_attn_gate_one") {
            ML::Metal::ComputePipeline.new("qwen35_mtp_attn_gate_one", MTP_SOURCE)
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

        private def self.dn128_chunk_rowwise_checkpoint_pipeline : ML::Metal::ComputePipeline
          @@dn128_chunk_rowwise_checkpoint_pipeline ||= ML::Metal::PipelineCache.get("delta_net_chunk_128_rowwise_checkpoint") {
            ML::Metal::ComputePipeline.new("delta_net_chunk_128_rowwise_checkpoint", DELTA_NET_SOURCE)
          }
        end

        private def self.dn128_chunk_rowwise_rollback_log_pipeline : ML::Metal::ComputePipeline
          @@dn128_chunk_rowwise_rollback_log_pipeline ||= ML::Metal::PipelineCache.get("delta_net_chunk_128_rowwise_rollback_log") {
            ML::Metal::ComputePipeline.new("delta_net_chunk_128_rowwise_rollback_log", DELTA_NET_SOURCE)
          }
        end

        private def self.dn128_rollback_rowwise_pipeline : ML::Metal::ComputePipeline
          @@dn128_rollback_rowwise_pipeline ||= ML::Metal::PipelineCache.get("delta_net_rollback_128_rowwise_from_log") {
            ML::Metal::ComputePipeline.new("delta_net_rollback_128_rowwise_from_log", DELTA_NET_SOURCE)
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

        private def self.dn_post_chunk_h16_pipeline : ML::Metal::ComputePipeline
          @@dn_post_chunk_h16_pipeline ||= ML::Metal::PipelineCache.get("delta_net_post_norm_gate_chunk_h16") {
            ML::Metal::ComputePipeline.new("delta_net_post_norm_gate_chunk_h16", DELTA_NET_SOURCE)
          }
        end

        private def self.lowrank_delta_pipeline : ML::Metal::ComputePipeline
          @@lowrank_delta_pipeline ||= ML::Metal::PipelineCache.get("lowrank_delta_step") {
            ML::Metal::ComputePipeline.new("lowrank_delta_step", DELTA_NET_SOURCE)
          }
        end

        private def self.lowrank_project_coeffs_pipeline : ML::Metal::ComputePipeline
          @@lowrank_project_coeffs_pipeline ||= ML::Metal::PipelineCache.get("lowrank_project_coeffs") {
            ML::Metal::ComputePipeline.new("lowrank_project_coeffs", DELTA_NET_SOURCE)
          }
        end

        private def self.lowrank_project_coeffs_chunk_pipeline : ML::Metal::ComputePipeline
          @@lowrank_project_coeffs_chunk_pipeline ||= ML::Metal::PipelineCache.get("lowrank_project_coeffs_chunk") {
            ML::Metal::ComputePipeline.new("lowrank_project_coeffs_chunk", DELTA_NET_SOURCE)
          }
        end

        private def self.lowrank_project_state_pipeline : ML::Metal::ComputePipeline
          @@lowrank_project_state_pipeline ||= ML::Metal::PipelineCache.get("lowrank_project_state") {
            ML::Metal::ComputePipeline.new("lowrank_project_state", DELTA_NET_SOURCE)
          }
        end

        private def self.lowrank_reconstruct_state_pipeline : ML::Metal::ComputePipeline
          @@lowrank_reconstruct_state_pipeline ||= ML::Metal::PipelineCache.get("lowrank_reconstruct_state") {
            ML::Metal::ComputePipeline.new("lowrank_reconstruct_state", DELTA_NET_SOURCE)
          }
        end

        private def self.lowrank_delta_chunk_pipeline : ML::Metal::ComputePipeline
          @@lowrank_delta_chunk_pipeline ||= ML::Metal::PipelineCache.get("lowrank_delta_chunk_step_parallel") {
            ML::Metal::ComputePipeline.new("lowrank_delta_chunk_step_parallel", DELTA_NET_SOURCE)
          }
        end

        private def self.ffn_pca_updown_coeffs_pipeline : ML::Metal::ComputePipeline
          @@ffn_pca_updown_coeffs_pipeline ||= ML::Metal::PipelineCache.get("ffn_pca_updown_coeffs") {
            ML::Metal::ComputePipeline.new("ffn_pca_updown_coeffs", DELTA_NET_SOURCE)
          }
        end

        private def self.ffn_pca_updown_out_pipeline : ML::Metal::ComputePipeline
          @@ffn_pca_updown_out_pipeline ||= ML::Metal::PipelineCache.get("ffn_pca_updown_out") {
            ML::Metal::ComputePipeline.new("ffn_pca_updown_out", DELTA_NET_SOURCE)
          }
        end

        private def self.ffn_pca_updown_fused_pipeline : ML::Metal::ComputePipeline
          @@ffn_pca_updown_fused_pipeline ||= ML::Metal::PipelineCache.get("ffn_pca_updown_fused") {
            ML::Metal::ComputePipeline.new("ffn_pca_updown_fused", DELTA_NET_SOURCE)
          }
        end

        private def self.ffn_pca_updown_fused_rows_pipeline : ML::Metal::ComputePipeline
          @@ffn_pca_updown_fused_rows_pipeline ||= ML::Metal::PipelineCache.get("ffn_pca_updown_fused_rows") {
            ML::Metal::ComputePipeline.new("ffn_pca_updown_fused_rows", DELTA_NET_SOURCE)
          }
        end

        private def self.ffn_pca_updown_fused_rows_q8_pipeline : ML::Metal::ComputePipeline
          @@ffn_pca_updown_fused_rows_q8_pipeline ||= ML::Metal::PipelineCache.get("ffn_pca_updown_fused_rows_q8") {
            ML::Metal::ComputePipeline.new("ffn_pca_updown_fused_rows_q8", FFN_UPDOWN_Q8_SOURCE)
          }
        end

        private def self.attn_pipeline : ML::Metal::ComputePipeline
          @@attn_pipeline ||= ML::Metal::PipelineCache.get("qwen35_attn_decode") {
            ML::Metal::ComputePipeline.new("qwen35_attn_decode", ATTN_DECODE_SOURCE)
          }
        end

        private def self.attn_gqa4_pipeline : ML::Metal::ComputePipeline
          @@attn_gqa4_pipeline ||= ML::Metal::PipelineCache.get("qwen35_attn_decode_gqa4") {
            ML::Metal::ComputePipeline.new("qwen35_attn_decode_gqa4", ATTN_DECODE_SOURCE)
          }
        end

        private def self.attn_splitk_stage1_pipeline : ML::Metal::ComputePipeline
          @@attn_splitk_stage1_pipeline ||= ML::Metal::PipelineCache.get("qwen35_attn_decode_splitk_stage1") {
            ML::Metal::ComputePipeline.new("qwen35_attn_decode_splitk_stage1", ATTN_DECODE_SOURCE)
          }
        end

        private def self.attn_splitk_stage2_pipeline : ML::Metal::ComputePipeline
          @@attn_splitk_stage2_pipeline ||= ML::Metal::PipelineCache.get("qwen35_attn_decode_splitk_stage2") {
            ML::Metal::ComputePipeline.new("qwen35_attn_decode_splitk_stage2", ATTN_DECODE_SOURCE)
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

        private def self.ffn_swiglu_h16_pipeline : ML::Metal::ComputePipeline
          @@ffn_swiglu_h16_pipeline ||= ML::Metal::PipelineCache.get("qwen35_swiglu_mul_h16") {
            ML::Metal::ComputePipeline.new("qwen35_swiglu_mul_h16", FFN_SOURCE)
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

        private def self.add_rmsnorm_rows_h16_pipeline : ML::Metal::ComputePipeline
          @@add_rmsnorm_rows_h16_pipeline ||= ML::Metal::PipelineCache.get("qwen35_add_rmsnorm_rows_h16") {
            ML::Metal::ComputePipeline.new("qwen35_add_rmsnorm_rows_h16", FFN_SOURCE)
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

        private def self.rmsnorm_rows_f32_h16_pipeline : ML::Metal::ComputePipeline
          @@rmsnorm_rows_f32_h16_pipeline ||= ML::Metal::PipelineCache.get("qwen35_rmsnorm_rows_f32_h16") {
            ML::Metal::ComputePipeline.new("qwen35_rmsnorm_rows_f32_h16", FFN_SOURCE)
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

        private def self.recurrent_conv_shift_chunk_checkpoint_pipeline : ML::Metal::ComputePipeline
          @@recurrent_conv_shift_chunk_checkpoint_pipeline ||= ML::Metal::PipelineCache.get("qwen35_recurrent_conv_shift_chunk_checkpoint") {
            ML::Metal::ComputePipeline.new("qwen35_recurrent_conv_shift_chunk_checkpoint", RECURRENT_SOURCE)
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

        private def self.attn_rows_sg4_pipeline : ML::Metal::ComputePipeline
          @@attn_rows_sg4_pipeline ||= ML::Metal::PipelineCache.get("qwen35_attn_decode_rows_sg4") {
            ML::Metal::ComputePipeline.new("qwen35_attn_decode_rows_sg4", FULLATTN_SOURCE)
          }
        end

        private def self.attn_rows_sg4_pregate_pipeline : ML::Metal::ComputePipeline
          @@attn_rows_sg4_pregate_pipeline ||= ML::Metal::PipelineCache.get("qwen35_attn_decode_rows_sg4_pregate") {
            ML::Metal::ComputePipeline.new("qwen35_attn_decode_rows_sg4_pregate", FULLATTN_SOURCE)
          }
        end

        private def self.gemv_pipeline_for(qw : QuantWeight) : ML::Metal::ComputePipeline?
          case qw.type
          when .q4_k? then mv_pipeline
          when .q5_k? then mv5_pipeline
          when .q6_k? then mv6_pipeline
          when .q8_0? then mv8_pipeline
          when .iq4_nl? then mv_iq4_nl_pipeline
          when .f32? then mv_f32_pipeline
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
          when .same?(mv_iq4_nl_pipeline)
            MV_IQ4_NL_NSG * MV_IQ4_NL_NR0
          when .same?(mv_f32_pipeline)
            MV_F32_NSG * MV_F32_NR0
          else
            MV_Q4_NSG * MV_Q4_NR0
          end
        end

        private def self.gemv_threads_per_tg_for(pipeline : ML::Metal::ComputePipeline) : Int32
          case pipeline
          when .same?(mv8_pipeline), .same?(mv8_add_pipeline), .same?(mv8_top1_tiles_pipeline), .same?(mv8_top2_tiles_pipeline)
            MV_Q8_NSG * 32
          when .same?(mv_iq4_nl_pipeline)
            MV_IQ4_NL_NSG * 32
          when .same?(mv_f32_pipeline)
            MV_F32_NSG * 32
          else
            64
          end
        end

        private def self.gemv_profile_quant(pipeline : ML::Metal::ComputePipeline) : {String, Int32, Int32}
          case pipeline
          when .same?(mv5_pipeline)
            {"Q5_K", Q5K_BLOCK_BYTES, QK_K}
          when .same?(mv6_pipeline), .same?(mv6_add_pipeline), .same?(mv6_top1_tiles_pipeline), .same?(mv6_top2_tiles_pipeline)
            {"Q6_K", Q6K_BLOCK_BYTES, QK_K}
          when .same?(mv8_pipeline), .same?(mv8_add_pipeline), .same?(mv8_top1_tiles_pipeline), .same?(mv8_top2_tiles_pipeline)
            {"Q8_0", Q8_0_BLOCK_BYTES, Q8_0_QK}
          when .same?(mv_iq4_nl_pipeline)
            {"IQ4_NL", IQ4_NL_BLOCK_BYTES, IQ4_NL_QK}
          when .same?(mv_f32_pipeline)
            {"F32", 4, 1}
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
          if q4_tensor_mm_enabled? && q4_tensor_ffn_candidate?(out_dim) && batch >= Q4_TENSOR_NR1
            enc.set_pipeline(mm_q4_tensor_f32out_pipeline)
            enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
            enc.set_buffer(x16_buf, 1)
            enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
            enc.set_value(in_dim.to_u32, 3)
            enc.set_value(out_dim.to_u32, 4)
            enc.set_value(batch.to_u32, 5)
            enc.set_threadgroup_memory(Q4_TENSOR_SHMEM, 0)
            enc.dispatch_threadgroups({
              (batch + Q4_TENSOR_NR1 - 1) // Q4_TENSOR_NR1,
              (out_dim + MM_NR0 - 1) // MM_NR0,
              1,
            }, {Q4_TENSOR_TG, 1, 1})
            return
          end

          if q4_h16_b48_gemm_enabled? && batch == MM48_NR1
            enc.set_pipeline(mm_h16_b48_pipeline)
            enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
            enc.set_buffer(x16_buf, 1)
            enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
            enc.set_value(in_dim.to_u32, 3)
            enc.set_value(out_dim.to_u32, 4)
            enc.set_value(batch.to_u32, 5)
            enc.set_threadgroup_memory(MM48_SHMEM, 0)
            enc.dispatch_threadgroups({
              1,
              (out_dim + MM_NR0 - 1) // MM_NR0,
              1,
            }, {MM48_TG, 1, 1})
            return
          end

          if q4_h16_b80_gemm_enabled? && batch == MM80_NR1
            enc.set_pipeline(mm_h16_b80_pipeline)
            enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
            enc.set_buffer(x16_buf, 1)
            enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
            enc.set_value(in_dim.to_u32, 3)
            enc.set_value(out_dim.to_u32, 4)
            enc.set_value(batch.to_u32, 5)
            enc.set_threadgroup_memory(MM80_SHMEM, 0)
            enc.dispatch_threadgroups({
              1,
              (out_dim + MM_NR0 - 1) // MM_NR0,
              1,
            }, {MM80_TG, 1, 1})
            return
          end

          if q4_h16_b96_gemm_enabled? && batch == MM96_NR1
            enc.set_pipeline(mm_h16_b96_pipeline)
            enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
            enc.set_buffer(x16_buf, 1)
            enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
            enc.set_value(in_dim.to_u32, 3)
            enc.set_value(out_dim.to_u32, 4)
            enc.set_value(batch.to_u32, 5)
            enc.set_threadgroup_memory(MM96_SHMEM, 0)
            enc.dispatch_threadgroups({
              (batch + MM96_NR1 - 1) // MM96_NR1,
              (out_dim + MM_NR0 - 1) // MM_NR0,
              1,
            }, {MM96_TG, 1, 1})
            return
          end

          if q4_h16_b112_gemm_enabled? && batch == MM112_NR1
            enc.set_pipeline(mm_h16_b112_pipeline)
            enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
            enc.set_buffer(x16_buf, 1)
            enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
            enc.set_value(in_dim.to_u32, 3)
            enc.set_value(out_dim.to_u32, 4)
            enc.set_value(batch.to_u32, 5)
            enc.set_threadgroup_memory(MM112_SHMEM, 0)
            enc.dispatch_threadgroups({
              1,
              (out_dim + MM_NR0 - 1) // MM_NR0,
              1,
            }, {MM112_TG, 1, 1})
            return
          end

          use_b64_tail = q4_h16_b64_tail_candidate?(batch)
          if q4_h16_b64_gemm_enabled? && batch >= MM64_NR1 && ((batch % MM64_NR1) == 0 || use_b64_tail)
            enc.set_pipeline(mm_h16_b64_pipeline)
            enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
            enc.set_buffer(x16_buf, 1)
            enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
            enc.set_value(in_dim.to_u32, 3)
            enc.set_value(out_dim.to_u32, 4)
            enc.set_value(batch.to_u32, 5)
            enc.set_threadgroup_memory(MM64_SHMEM, 0)
            enc.dispatch_threadgroups({
              (batch + MM64_NR1 - 1) // MM64_NR1,
              (out_dim + MM_NR0 - 1) // MM_NR0,
              1,
            }, {MM64_TG, 1, 1})
            return
          end

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

        private def self.encode_q4k_gemm_h16_b64_swiglu_from_h16(enc : ML::Metal::ComputeEncoder,
                                                                 x16_buf : ML::MetalBuffer,
                                                                 gate_buf : ML::MetalBuffer,
                                                                 out_buf : ML::MetalBuffer,
                                                                 w_buf : ML::MetalBuffer,
                                                                 w_offset : Int64,
                                                                 in_dim : Int32,
                                                                 out_dim : Int32,
                                                                 batch : Int32) : Nil
          enc.set_pipeline(mm_h16_b64_swiglu_pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
          enc.set_buffer(x16_buf, 1)
          enc.set_buffer(gate_buf, 2)
          enc.set_buffer(out_buf, 3, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32, 4)
          enc.set_value(out_dim.to_u32, 5)
          enc.set_value(batch.to_u32, 6)
          enc.set_threadgroup_memory(MM64_SHMEM, 0)
          enc.dispatch_threadgroups({
            (batch + MM64_NR1 - 1) // MM64_NR1,
            (out_dim + MM_NR0 - 1) // MM_NR0,
            1,
          }, {MM64_TG, 1, 1})
        end

        private def self.encode_q4k_gemm_h16_b64_gelu_mul_from_h16(enc : ML::Metal::ComputeEncoder,
                                                                   x16_buf : ML::MetalBuffer,
                                                                   gate_buf : ML::MetalBuffer,
                                                                   out_buf : ML::MetalBuffer,
                                                                   w_buf : ML::MetalBuffer,
                                                                   w_offset : Int64,
                                                                   in_dim : Int32,
                                                                   out_dim : Int32,
                                                                   batch : Int32) : Nil
          enc.set_pipeline(mm_h16_b64_gelu_mul_pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
          enc.set_buffer(x16_buf, 1)
          enc.set_buffer(gate_buf, 2)
          enc.set_buffer(out_buf, 3, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32, 4)
          enc.set_value(out_dim.to_u32, 5)
          enc.set_value(batch.to_u32, 6)
          enc.set_threadgroup_memory(MM64_SHMEM, 0)
          enc.dispatch_threadgroups({
            (batch + MM64_NR1 - 1) // MM64_NR1,
            (out_dim + MM_NR0 - 1) // MM_NR0,
            1,
          }, {MM64_TG, 1, 1})
        end

        private def self.encode_q4k_gemm_h16_b64_swiglu_h16_from_h16(enc : ML::Metal::ComputeEncoder,
                                                                     x16_buf : ML::MetalBuffer,
                                                                     gate_buf : ML::MetalBuffer,
                                                                     out_h16_buf : ML::MetalBuffer,
                                                                     w_buf : ML::MetalBuffer,
                                                                     w_offset : Int64,
                                                                     in_dim : Int32,
                                                                     out_dim : Int32,
                                                                     batch : Int32) : Nil
          enc.set_pipeline(mm_h16_b64_swiglu_h16_pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_offset)
          enc.set_buffer(x16_buf, 1)
          enc.set_buffer(gate_buf, 2)
          enc.set_buffer(out_h16_buf, 3, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32, 4)
          enc.set_value(out_dim.to_u32, 5)
          enc.set_value(batch.to_u32, 6)
          enc.set_threadgroup_memory(MM64_SHMEM, 0)
          enc.dispatch_threadgroups({
            (batch + MM64_NR1 - 1) // MM64_NR1,
            (out_dim + MM_NR0 - 1) // MM_NR0,
            1,
          }, {MM64_TG, 1, 1})
        end

        private def self.encode_q4k_gemm_h16_pair_b64_swiglu(enc : ML::Metal::ComputeEncoder,
                                                             x_buf : ML::MetalBuffer,
                                                             gate_buf : ML::MetalBuffer,
                                                             act_buf : ML::MetalBuffer,
                                                             gate_w_buf : ML::MetalBuffer,
                                                             gate_w_offset : Int64,
                                                             up_w_buf : ML::MetalBuffer,
                                                             up_w_offset : Int64,
                                                             in_dim : Int32,
                                                             out_dim : Int32,
                                                             batch : Int32) : Nil
          x16_buf = Scratch.get(:mm4_pair_swiglu_x16, (batch * in_dim).to_i64 * 2_i64)

          Profile.bump_conversion("f32_to_f16 q4_pair_swiglu_input #{in_dim} b#{batch}", (batch * in_dim).to_i64 * 6_i64)
          enc.set_pipeline(f32_to_f16_pipeline)
          enc.set_buffer(x_buf, 0)
          enc.set_buffer(x16_buf, 1, ML::Metal::BufferAccess::Write)
          enc.set_value((batch * in_dim).to_u32, 2)
          enc.dispatch_1d(batch * in_dim, 256)

          encode_q4k_gemm_h16_from_h16(enc, x16_buf, gate_buf, gate_w_buf, gate_w_offset, in_dim, out_dim, batch)
          encode_q4k_gemm_h16_b64_swiglu_from_h16(enc, x16_buf, gate_buf, act_buf, up_w_buf, up_w_offset, in_dim, out_dim, batch)
        end

        private def self.encode_q4k_gemm_h16_pair_b64_swiglu_h16(enc : ML::Metal::ComputeEncoder,
                                                                 x_buf : ML::MetalBuffer,
                                                                 gate_buf : ML::MetalBuffer,
                                                                 act_h16_buf : ML::MetalBuffer,
                                                                 gate_w_buf : ML::MetalBuffer,
                                                                 gate_w_offset : Int64,
                                                                 up_w_buf : ML::MetalBuffer,
                                                                 up_w_offset : Int64,
                                                                 in_dim : Int32,
                                                                 out_dim : Int32,
                                                                 batch : Int32) : Nil
          x16_buf = Scratch.get(:mm4_pair_swiglu_h16_x16, (batch * in_dim).to_i64 * 2_i64)

          Profile.bump_conversion("f32_to_f16 q4_pair_swiglu_h16_input #{in_dim} b#{batch}", (batch * in_dim).to_i64 * 6_i64)
          enc.set_pipeline(f32_to_f16_pipeline)
          enc.set_buffer(x_buf, 0)
          enc.set_buffer(x16_buf, 1, ML::Metal::BufferAccess::Write)
          enc.set_value((batch * in_dim).to_u32, 2)
          enc.dispatch_1d(batch * in_dim, 256)

          encode_q4k_gemm_h16_from_h16(enc, x16_buf, gate_buf, gate_w_buf, gate_w_offset, in_dim, out_dim, batch)
          encode_q4k_gemm_h16_b64_swiglu_h16_from_h16(enc, x16_buf, gate_buf, act_h16_buf, up_w_buf, up_w_offset, in_dim, out_dim, batch)
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

        private def self.encode_q56k_gemm_f32_from_h16(enc : ML::Metal::ComputeEncoder,
                                                       pipeline : ML::Metal::ComputePipeline,
                                                       x16_buf : ML::MetalBuffer,
                                                       out_buf : ML::MetalBuffer,
                                                       w_buf : ML::MetalBuffer,
                                                       w_offset : Int64,
                                                       in_dim : Int32,
                                                       out_dim : Int32,
                                                       batch : Int32) : Nil
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

        private def self.encode_q56k_gemm_f32_add_from_h16(enc : ML::Metal::ComputeEncoder,
                                                           pipeline : ML::Metal::ComputePipeline,
                                                           x16_buf : ML::MetalBuffer,
                                                           residual_buf : ML::MetalBuffer,
                                                           out_buf : ML::MetalBuffer,
                                                           w_buf : ML::MetalBuffer,
                                                           w_offset : Int64,
                                                           in_dim : Int32,
                                                           out_dim : Int32,
                                                           batch : Int32) : Nil
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

        private def self.prefill_swiglu_h16_down_candidate?(qw : QuantWeight, batch : Int32) : Bool
          prefill_swiglu_h16_down_enabled? && h16_batch_gemm_candidate?(qw, batch)
        end

        private def self.h16_batch_gemm_candidate?(qw : QuantWeight, batch : Int32) : Bool
          return false unless batch > GEMM_BATCH_THRESHOLD
          force_small_q4_gemv = small_q4_gemv_enabled? && qw.type.q4_k? && qw.out_dim <= 64
          (q4_h16_gemm_enabled? && qw.type.q4_k? && !force_small_q4_gemv) ||
            (q56_batch_gemm_enabled? && (qw.type.q5_k? || qw.type.q6_k?))
        end

        private def self.encode_matmul_from_h16(enc : ML::Metal::ComputeEncoder,
                                                qw : QuantWeight,
                                                x16_buf : ML::MetalBuffer,
                                                out_buf : ML::MetalBuffer,
                                                w_buf : ML::MetalBuffer,
                                                w_offset : Int64,
                                                in_dim : Int32,
                                                out_dim : Int32,
                                                batch : Int32) : Bool
          force_small_q4_gemv = small_q4_gemv_enabled? && qw.type.q4_k? && out_dim <= 64
          if q4_h16_gemm_enabled? && qw.type.q4_k? && batch > GEMM_BATCH_THRESHOLD && !force_small_q4_gemv
            Profile.bump_matmul_shape("q4_h16_gemm #{qw.type.name} #{in_dim}x#{out_dim} b#{batch}", qw.raw.size.to_i64)
            encode_q4k_gemm_h16_from_h16(enc, x16_buf, out_buf, w_buf, w_offset, in_dim, out_dim, batch)
          elsif q56_batch_gemm_enabled? && qw.type.q5_k? && batch > GEMM_BATCH_THRESHOLD
            Profile.bump_matmul_shape("q5_gemm #{qw.type.name} #{in_dim}x#{out_dim} b#{batch}", qw.raw.size.to_i64)
            encode_q56k_gemm_f32_from_h16(enc, mm5_f32out_pipeline, x16_buf, out_buf, w_buf, w_offset, in_dim, out_dim, batch)
          elsif q56_batch_gemm_enabled? && qw.type.q6_k? && batch > GEMM_BATCH_THRESHOLD
            Profile.bump_matmul_shape("q6_gemm #{qw.type.name} #{in_dim}x#{out_dim} b#{batch}", qw.raw.size.to_i64)
            encode_q56k_gemm_f32_from_h16(enc, mm6_f32out_pipeline, x16_buf, out_buf, w_buf, w_offset, in_dim, out_dim, batch)
          else
            return false
          end
          true
        end

        private def self.encode_matmul_add_from_h16(enc : ML::Metal::ComputeEncoder,
                                                    qw : QuantWeight,
                                                    x16_buf : ML::MetalBuffer,
                                                    residual_buf : ML::MetalBuffer,
                                                    out_buf : ML::MetalBuffer,
                                                    w_buf : ML::MetalBuffer,
                                                    w_offset : Int64,
                                                    in_dim : Int32,
                                                    out_dim : Int32,
                                                    batch : Int32) : Bool
          return false unless q56_batch_gemm_enabled? && qw.type.q6_k? && batch > GEMM_BATCH_THRESHOLD

          Profile.bump_matmul_shape("q6_gemm_add #{qw.type.name} #{in_dim}x#{out_dim} b#{batch}", qw.raw.size.to_i64)
          encode_q56k_gemm_f32_add_from_h16(enc, mm6_f32out_add_pipeline, x16_buf, residual_buf, out_buf, w_buf, w_offset, in_dim, out_dim, batch)
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

        private def self.encode_rmsnorm_rows_f32_h16(enc : ML::Metal::ComputeEncoder,
                                                     x_buf : ML::MetalBuffer,
                                                     weight_buf : ML::MetalBuffer,
                                                     out_buf : ML::MetalBuffer,
                                                     out_h16_buf : ML::MetalBuffer,
                                                     dim : Int32,
                                                     n_rows : Int32,
                                                     eps : Float32) : Nil
          enc.set_pipeline(rmsnorm_rows_f32_h16_pipeline)
          enc.set_buffer(x_buf, 0)
          enc.set_buffer(weight_buf, 1)
          enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_buffer(out_h16_buf, 3, ML::Metal::BufferAccess::Write)
          enc.set_value(dim.to_u32, 4)
          enc.set_value(n_rows.to_u32, 5)
          enc.set_value(eps, 6)
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

        private def self.encode_add_rmsnorm_rows_h16(enc : ML::Metal::ComputeEncoder,
                                                     x_buf : ML::MetalBuffer,
                                                     y_buf : ML::MetalBuffer,
                                                     weight_buf : ML::MetalBuffer,
                                                     residual_buf : ML::MetalBuffer,
                                                     normed_h16_buf : ML::MetalBuffer,
                                                     dim : Int32,
                                                     n_rows : Int32,
                                                     eps : Float32) : Nil
          enc.set_pipeline(add_rmsnorm_rows_h16_pipeline)
          enc.set_buffer(x_buf, 0)
          enc.set_buffer(y_buf, 1)
          enc.set_buffer(weight_buf, 2)
          enc.set_buffer(residual_buf, 3, ML::Metal::BufferAccess::Write)
          enc.set_buffer(normed_h16_buf, 4, ML::Metal::BufferAccess::Write)
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

        private def self.prefill_phase_profile_enabled? : Bool
          ENV["QWEN35_PREFILL_PHASE_PROFILE"]? == "1" || ENV["QWEN35_METAL_PROFILE"]? == "1"
        end

        private def self.prefill_full_detail_profile_enabled? : Bool
          ENV["QWEN35_PREFILL_FULL_DETAIL_PROFILE"]? == "1"
        end

        private def self.prefill_attn_rows_sg4_enabled? : Bool
          ENV["QWEN35_PREFILL_ATTN_ROWS_SG4_OFF"]? != "1"
        end

        private def self.prefill_attn_rows_sg4_pregate_enabled? : Bool
          ENV["QWEN35_PREFILL_ATTN_ROWS_SG4_PREGATE"]? == "1"
        end

        private def self.prefill_attn_rows_sg4_direct_gate_min_tokens : Int32
          (ENV["QWEN35_PREFILL_ATTN_ROWS_SG4_DIRECT_GATE_MIN"]? || "1024").to_i32
        end

        private def self.prefill_attn_rows_sg4_direct_gate_enabled?(n_tokens : Int32) : Bool
          ENV["QWEN35_PREFILL_ATTN_ROWS_SG4_DIRECT_GATE_OFF"]? != "1" &&
            n_tokens >= prefill_attn_rows_sg4_direct_gate_min_tokens
        end

        private def self.prefill_phase_checkpoint(cmd : ML::Metal::CommandBuffer,
                                                  label : String,
                                                  started_at : Time::Instant) : {ML::Metal::CommandBuffer, Time::Instant}
          tenc = Time.instant
          cmd.commit
          cmd.wait
          twait = Time.instant
          Profile.bump_group(label,
            (tenc - started_at).total_nanoseconds.to_i64,
            (twait - tenc).total_nanoseconds.to_i64,
            0_i64)
          {ML::Metal::CommandBuffer.new, Time.instant}
        end

        private def self.small_q4_gemv_enabled? : Bool
          ENV["QWEN35_SMALL_Q4_GEMV_OFF"]? != "1"
        end

        private def self.q4_h16_gemm_enabled? : Bool
          ENV["QWEN35_Q4K_H16_GEMM_OFF"]? != "1"
        end

        private def self.q4_tensor_mm_enabled? : Bool
          ENV["QWEN35_Q4K_TENSOR_MM"]? == "1" &&
            ENV["GEMMA4_ROW_PREFILL_Q4_GELU_FUSE"]? != "1"
        end

        private def self.q4_tensor_ffn_candidate?(out_dim : Int32) : Bool
          # Keep this experimental route to large FFN gate/up projections.
          # Smaller attention-sized projections and the Gemma GELU-fuse
          # composition are not promoted by the current evidence.
          out_dim >= 8192
        end

        private def self.q4_h16_b64_gemm_enabled? : Bool
          ENV["QWEN35_Q4K_H16_B64_OFF"]? != "1"
        end

        private def self.q4_h16_b64_tail_min_batch : Int32
          (ENV["QWEN35_Q4K_H16_B64_TAIL_MIN"]? || "0").to_i32
        end

        private def self.q4_h16_b64_tail_candidate?(batch : Int32) : Bool
          min_batch = q4_h16_b64_tail_min_batch
          min_batch > 0 && batch >= min_batch
        end

        private def self.q4_h16_exact_rowpack_candidate?(batch : Int32) : Bool
          (q4_h16_b48_gemm_enabled? && batch == MM48_NR1) ||
            (q4_h16_b80_gemm_enabled? && batch == MM80_NR1) ||
            (q4_h16_b96_gemm_enabled? && batch == MM96_NR1) ||
            (q4_h16_b112_gemm_enabled? && batch == MM112_NR1)
        end

        private def self.q4_h16_b64_swiglu_batch_candidate?(batch : Int32) : Bool
          batch >= MM64_NR1 &&
            ((batch % MM64_NR1) == 0 ||
              (q4_h16_b64_tail_candidate?(batch) && !q4_h16_exact_rowpack_candidate?(batch)))
        end

        private def self.q4_h16_b48_gemm_enabled? : Bool
          ENV["QWEN35_Q4K_H16_B48_OFF"]? != "1"
        end

        private def self.q4_h16_b80_gemm_enabled? : Bool
          ENV["QWEN35_Q4K_H16_B80_OFF"]? != "1"
        end

        private def self.q4_h16_b96_gemm_enabled? : Bool
          ENV["QWEN35_Q4K_H16_B96_OFF"]? != "1"
        end

        private def self.q4_h16_b112_gemm_enabled? : Bool
          ENV["QWEN35_Q4K_H16_B112_OFF"]? != "1"
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

        private def self.prefill_addnorm_h16_ffn_enabled? : Bool
          ENV["QWEN35_ADDNORM_H16_FFN"]? == "1"
        end

        private def self.prefill_swiglu_h16_down_enabled? : Bool
          ENV["QWEN35_SWIGLU_H16_DOWN"]? == "1"
        end

        private def self.prefill_rmsnorm_h16_proj_enabled? : Bool
          ENV["QWEN35_RMSNORM_H16_PROJ"]? == "1"
        end

        private def self.prefill_dn_post_h16_oproj_enabled? : Bool
          ENV["QWEN35_DN_POST_H16_OPROJ"]? == "1"
        end

        private def self.prefill_q4_b64_up_swiglu_enabled? : Bool
          ENV["QWEN35_Q4K_B64_UP_SWIGLU"]? == "1"
        end

        private def self.prefill_q4_b64_up_swiglu_h16_enabled? : Bool
          ENV["QWEN35_Q4K_B64_UP_SWIGLU_H16_OFF"]? != "1"
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

        private def self.q4_b64_up_swiglu_candidate?(gate_qw : QuantWeight,
                                                     up_qw : QuantWeight,
                                                     batch : Int32,
                                                     down_h16 : Bool) : Bool
          prefill_q4_b64_up_swiglu_enabled? &&
            !down_h16 &&
            q4_h16_b64_gemm_enabled? &&
            q4_pair_h16_gemm_candidate?(gate_qw, up_qw, batch) &&
            q4_h16_b64_swiglu_batch_candidate?(batch) &&
            (up_qw.out_dim % MM_NR0) == 0
        end

        private def self.q4_b64_up_swiglu_h16_candidate?(gate_qw : QuantWeight,
                                                         up_qw : QuantWeight,
                                                         batch : Int32,
                                                         down_h16 : Bool) : Bool
          prefill_q4_b64_up_swiglu_h16_enabled? &&
            down_h16 &&
            q4_h16_b64_gemm_enabled? &&
            q4_pair_h16_gemm_candidate?(gate_qw, up_qw, batch) &&
            q4_h16_b64_swiglu_batch_candidate?(batch) &&
            (up_qw.out_dim % MM_NR0) == 0
        end

        private def self.q4_b64_up_swiglu_h16_down_candidate?(gate_qw : QuantWeight,
                                                              up_qw : QuantWeight,
                                                              down_qw : QuantWeight,
                                                              batch : Int32) : Bool
          prefill_q4_b64_up_swiglu_h16_enabled? &&
            h16_batch_gemm_candidate?(down_qw, batch) &&
            q4_h16_b64_gemm_enabled? &&
            q4_pair_h16_gemm_candidate?(gate_qw, up_qw, batch) &&
            q4_h16_b64_swiglu_batch_candidate?(batch) &&
            (up_qw.out_dim % MM_NR0) == 0
        end

        private def self.read_shared_top1(id_buf : ML::MetalBuffer, value_buf : ML::MetalBuffer) : Array(Float32)
          id = id_buf.contents.as(Pointer(UInt32)).value
          value = value_buf.contents.as(Pointer(Float32)).value
          [id.to_f32, value]
        end

        private def self.read_shared_top2(id_buf : ML::MetalBuffer,
                                          value_buf : ML::MetalBuffer,
                                          second_id_buf : ML::MetalBuffer,
                                          second_value_buf : ML::MetalBuffer) : Array(Float32)
          id = id_buf.contents.as(Pointer(UInt32)).value
          value = value_buf.contents.as(Pointer(Float32)).value
          second_id = second_id_buf.contents.as(Pointer(UInt32)).value
          second_value = second_value_buf.contents.as(Pointer(Float32)).value
          [id.to_f32, value, second_id.to_f32, second_value]
        end

        private def self.read_shared_top1_rows(id_buf : ML::MetalBuffer, value_buf : ML::MetalBuffer, rows : Int32) : Array({Int32, Float32})
          ids = id_buf.contents.as(Pointer(UInt32))
          values = value_buf.contents.as(Pointer(Float32))
          Array({Int32, Float32}).new(rows) do |i|
            {ids[i].to_i32, values[i]}
          end
        end

        record DecodeWaveSubmission,
          cmd : ML::Metal::CommandBuffer,
          pending_cmds : Array(ML::Metal::CommandBuffer),
          emit_head : Bool,
          use_head_top1 : Bool,
          use_head_top2 : Bool,
          logits_buf : ML::MetalBuffer?,
          top1_id_buf : ML::MetalBuffer?,
          top1_value_buf : ML::MetalBuffer?,
          second_id_buf : ML::MetalBuffer?,
          second_value_buf : ML::MetalBuffer?,
          output_dim : Int32,
          retained_bufs : Array(ML::MetalBuffer),
          profile_t0 : Time::Instant?,
          profile_tenc : Time::Instant?

        def self.wait_forward_decode_wave(submission : DecodeWaveSubmission) : Array(Float32)
          submission.pending_cmds.each(&.wait)
          submission.cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = if submission.emit_head
                     if submission.use_head_top2
                       read_shared_top2(submission.top1_id_buf.not_nil!, submission.top1_value_buf.not_nil!,
                         submission.second_id_buf.not_nil!, submission.second_value_buf.not_nil!)
                     elsif submission.use_head_top1
                       read_shared_top1(submission.top1_id_buf.not_nil!, submission.top1_value_buf.not_nil!)
                     else
                       read_shared_f32(submission.logits_buf.not_nil!, submission.output_dim)
                     end
                   else
                     [] of Float32
                   end
          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_wave(
              (submission.profile_tenc.not_nil! - submission.profile_t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - submission.profile_tenc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          result
        end

        record LowRankLayerChunkSubmission,
          cmd : ML::Metal::CommandBuffer,
          output_buf : ML::MetalBuffer,
          output_size : Int32,
          retained_bufs : Array(ML::MetalBuffer)

        def self.wait_lowrank_layer_chunk(submission : LowRankLayerChunkSubmission) : Array(Float32)
          submission.cmd.wait
          read_shared_f32(submission.output_buf, submission.output_size)
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

        private def self.attn_splitk_min_context : Int32
          (ENV["QWEN35_ATTN_SPLITK_MIN_CTX"]? || "128").to_i32
        end

        private def self.attn_splitk_chunk_size : Int32
          value = (ENV["QWEN35_ATTN_SPLITK_CHUNK"]? || "64").to_i32
          value > 0 ? value : 64
        end

        private def self.attn_gqa4_enabled? : Bool
          ENV["QWEN35_ATTN_GQA4"]? == "1" && ENV["QWEN35_ATTN_GQA4_OFF"]? != "1"
        end

        private def self.can_use_head_top1_fused?(output_qw : QuantWeight) : Bool
          head_top1_fused_enabled? &&
            ((output_qw.type.q6_k? && output_qw.in_dim % QK_K == 0) ||
              (output_qw.type.q8_0? && output_qw.in_dim % Q8_0_QK == 0))
        end

        private def self.profile_bump_head_top1_shape(label : String,
                                                      output_qw : QuantWeight,
                                                      rows : Int32? = nil) : Nil
          return unless Profile.enabled?

          weight_bytes = if rows
                           row_bytes = output_qw.raw.size.to_i64 // output_qw.out_dim
                           row_bytes * rows.not_nil!
                         else
                           output_qw.raw.size.to_i64
                         end
          Profile.bump_matmul_shape("#{label} #{output_qw.type.name} #{output_qw.in_dim}x#{output_qw.out_dim} b1", weight_bytes)
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

        # Projected-K low-rank DeltaNet state update for the self-spec draft
        # branch. `m_state_buf` is [h_v, s, rank] and is updated in place.
        # `c` and `qbar` are pre-projected K/Q basis coefficients [h_k, rank].
        def self.lowrank_delta_step(m_state_buf : ML::MetalBuffer,
                                    c : Array(Float32),
                                    qbar : Array(Float32),
                                    v_conv : Array(Float32),
                                    ghead : Array(Float32),
                                    beta : Array(Float32),
                                    h_k : Int32, h_v : Int32, s : Int32,
                                    rank : Int32,
                                    scale : Float32) : Array(Float32)
          ML::Metal::Device.init!
          raise "lowrank_delta_step rank must be positive" unless rank > 0
          raise "lowrank_delta_step c size mismatch" unless c.size == h_k * rank
          raise "lowrank_delta_step qbar size mismatch" unless qbar.size == h_k * rank
          raise "lowrank_delta_step v size mismatch" unless v_conv.size == h_v * s
          raise "lowrank_delta_step g size mismatch" unless ghead.size == h_v
          raise "lowrank_delta_step beta size mismatch" unless beta.size == h_v
          raise "lowrank_delta_step state size mismatch" unless m_state_buf.size >= (h_v * s * rank).to_i64 * sizeof(Float32)

          c_buf = Scratch.get(:lowrank_delta_c, c.size.to_i64 * sizeof(Float32))
          qbar_buf = Scratch.get(:lowrank_delta_qbar, qbar.size.to_i64 * sizeof(Float32))
          v_buf = Scratch.get(:lowrank_delta_v, v_conv.size.to_i64 * sizeof(Float32))
          g_buf = Scratch.get(:lowrank_delta_g, ghead.size.to_i64 * sizeof(Float32))
          b_buf = Scratch.get(:lowrank_delta_b, beta.size.to_i64 * sizeof(Float32))
          out_buf = Scratch.get(:lowrank_delta_out, (h_v * s).to_i64 * sizeof(Float32))

          c_buf.write(c)
          qbar_buf.write(qbar)
          v_buf.write(v_conv)
          g_buf.write(ghead)
          b_buf.write(beta)

          cmd = ML::Metal::CommandBuffer.new
          enc = ML::Metal::ComputeEncoder.new(cmd)
          enc.set_pipeline(lowrank_delta_pipeline)
          enc.set_buffer(m_state_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          enc.set_buffer(c_buf, 1)
          enc.set_buffer(qbar_buf, 2)
          enc.set_buffer(v_buf, 3)
          enc.set_buffer(g_buf, 4)
          enc.set_buffer(b_buf, 5)
          enc.set_buffer(out_buf, 6, ML::Metal::BufferAccess::Write)
          enc.set_value(h_k.to_u32, 7)
          enc.set_value(h_v.to_u32, 8)
          enc.set_value(s.to_u32, 9)
          enc.set_value(rank.to_u32, 10)
          enc.set_value(scale, 11)
          enc.dispatch_threadgroups({s, h_v, 1}, {1, 1, 1})
          enc.end_encoding
          cmd.commit
          cmd.wait
          read_shared_f32(out_buf, h_v * s)
        end

        # Same low-rank state update as `lowrank_delta_step`, but computes the
        # Q/K basis coefficients on GPU first and keeps them in temporary Metal
        # buffers for the update dispatch in the same command buffer.
        def self.lowrank_delta_step_projected(m_state_buf : ML::MetalBuffer,
                                              q_conv : Array(Float32),
                                              k_conv : Array(Float32),
                                              basis : Array(Float32),
                                              v_conv : Array(Float32),
                                              ghead : Array(Float32),
                                              beta : Array(Float32),
                                              h_k : Int32, h_v : Int32, s : Int32,
                                              rank : Int32,
                                              scale : Float32) : Array(Float32)
          ML::Metal::Device.init!
          raise "lowrank_delta_step_projected rank must be positive" unless rank > 0
          raise "lowrank_delta_step_projected q size mismatch" unless q_conv.size == h_k * s
          raise "lowrank_delta_step_projected k size mismatch" unless k_conv.size == h_k * s
          raise "lowrank_delta_step_projected basis size mismatch" unless basis.size == h_k * rank * s
          raise "lowrank_delta_step_projected v size mismatch" unless v_conv.size == h_v * s
          raise "lowrank_delta_step_projected g size mismatch" unless ghead.size == h_v
          raise "lowrank_delta_step_projected beta size mismatch" unless beta.size == h_v
          raise "lowrank_delta_step_projected state size mismatch" unless m_state_buf.size >= (h_v * s * rank).to_i64 * sizeof(Float32)

          basis_buf = Scratch.get(:lowrank_project_basis, basis.size.to_i64 * sizeof(Float32))
          basis_buf.write(basis)
          lowrank_delta_step_projected_buf(m_state_buf, q_conv, k_conv, basis_buf, v_conv, ghead, beta,
            h_k, h_v, s, rank, scale)
        end

        def self.lowrank_delta_step_projected_buf(m_state_buf : ML::MetalBuffer,
                                                  q_conv : Array(Float32),
                                                  k_conv : Array(Float32),
                                                  basis_buf : ML::MetalBuffer,
                                                  v_conv : Array(Float32),
                                                  ghead : Array(Float32),
                                                  beta : Array(Float32),
                                                  h_k : Int32, h_v : Int32, s : Int32,
                                                  rank : Int32,
                                                  scale : Float32) : Array(Float32)
          ML::Metal::Device.init!
          raise "lowrank_delta_step_projected_buf rank must be positive" unless rank > 0
          raise "lowrank_delta_step_projected_buf q size mismatch" unless q_conv.size == h_k * s
          raise "lowrank_delta_step_projected_buf k size mismatch" unless k_conv.size == h_k * s
          raise "lowrank_delta_step_projected_buf basis size mismatch" unless basis_buf.size >= (h_k * rank * s).to_i64 * sizeof(Float32)
          raise "lowrank_delta_step_projected_buf v size mismatch" unless v_conv.size == h_v * s
          raise "lowrank_delta_step_projected_buf g size mismatch" unless ghead.size == h_v
          raise "lowrank_delta_step_projected_buf beta size mismatch" unless beta.size == h_v
          raise "lowrank_delta_step_projected_buf state size mismatch" unless m_state_buf.size >= (h_v * s * rank).to_i64 * sizeof(Float32)

          q_buf = Scratch.get(:lowrank_project_q, q_conv.size.to_i64 * sizeof(Float32))
          k_buf = Scratch.get(:lowrank_project_k, k_conv.size.to_i64 * sizeof(Float32))
          c_buf = Scratch.get(:lowrank_project_c, (h_k * rank).to_i64 * sizeof(Float32))
          qbar_buf = Scratch.get(:lowrank_project_qbar, (h_k * rank).to_i64 * sizeof(Float32))
          v_buf = Scratch.get(:lowrank_delta_v, v_conv.size.to_i64 * sizeof(Float32))
          g_buf = Scratch.get(:lowrank_delta_g, ghead.size.to_i64 * sizeof(Float32))
          b_buf = Scratch.get(:lowrank_delta_b, beta.size.to_i64 * sizeof(Float32))
          out_buf = Scratch.get(:lowrank_delta_out, (h_v * s).to_i64 * sizeof(Float32))

          q_buf.write(q_conv)
          k_buf.write(k_conv)
          v_buf.write(v_conv)
          g_buf.write(ghead)
          b_buf.write(beta)

          cmd = ML::Metal::CommandBuffer.new

          proj_enc = ML::Metal::ComputeEncoder.new(cmd)
          proj_enc.set_pipeline(lowrank_project_coeffs_pipeline)
          proj_enc.set_buffer(q_buf, 0)
          proj_enc.set_buffer(k_buf, 1)
          proj_enc.set_buffer(basis_buf, 2)
          proj_enc.set_buffer(c_buf, 3, ML::Metal::BufferAccess::Write)
          proj_enc.set_buffer(qbar_buf, 4, ML::Metal::BufferAccess::Write)
          proj_enc.set_value(h_k.to_u32, 5)
          proj_enc.set_value(s.to_u32, 6)
          proj_enc.set_value(rank.to_u32, 7)
          proj_enc.dispatch_threadgroups({(rank + 7) // 8, h_k, 2}, {8, 1, 1})
          proj_enc.end_encoding

          step_enc = ML::Metal::ComputeEncoder.new(cmd)
          step_enc.set_pipeline(lowrank_delta_pipeline)
          step_enc.set_buffer(m_state_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          step_enc.set_buffer(c_buf, 1)
          step_enc.set_buffer(qbar_buf, 2)
          step_enc.set_buffer(v_buf, 3)
          step_enc.set_buffer(g_buf, 4)
          step_enc.set_buffer(b_buf, 5)
          step_enc.set_buffer(out_buf, 6, ML::Metal::BufferAccess::Write)
          step_enc.set_value(h_k.to_u32, 7)
          step_enc.set_value(h_v.to_u32, 8)
          step_enc.set_value(s.to_u32, 9)
          step_enc.set_value(rank.to_u32, 10)
          step_enc.set_value(scale, 11)
          step_enc.dispatch_threadgroups({s, h_v, 1}, {1, 1, 1})
          step_enc.end_encoding

          cmd.commit
          cmd.wait
          read_shared_f32(out_buf, h_v * s)
        end

        def self.lowrank_project_state_buf(full_state_buf : ML::MetalBuffer,
                                           basis_buf : ML::MetalBuffer,
                                           h_k : Int32, h_v : Int32, s : Int32,
                                           rank : Int32,
                                           command_queue_name : String? = nil) : ML::MetalBuffer
          ML::Metal::Device.init!
          raise "lowrank_project_state rank must be positive" unless rank > 0
          raise "lowrank_project_state full state size mismatch" unless full_state_buf.size >= (h_v * s * s).to_i64 * sizeof(Float32)
          raise "lowrank_project_state basis size mismatch" unless basis_buf.size >= (h_k * rank * s).to_i64 * sizeof(Float32)

          cmd_queue = command_queue_name ? lane_command_queue(command_queue_name.not_nil!) : nil
          cmd = ML::Metal::CommandBuffer.new(queue: cmd_queue, fast: wave_fast_command_buffer_enabled?)
          out_buf = lowrank_project_state_append(full_state_buf, basis_buf, h_k, h_v, s, rank, cmd)
          cmd.commit
          cmd.wait
          out_buf
        end

        def self.lowrank_project_state_append(full_state_buf : ML::MetalBuffer,
                                              basis_buf : ML::MetalBuffer,
                                              h_k : Int32, h_v : Int32, s : Int32,
                                              rank : Int32,
                                              cmd : ML::Metal::CommandBuffer) : ML::MetalBuffer
          ML::Metal::Device.init!
          raise "lowrank_project_state rank must be positive" unless rank > 0
          raise "lowrank_project_state full state size mismatch" unless full_state_buf.size >= (h_v * s * s).to_i64 * sizeof(Float32)
          raise "lowrank_project_state basis size mismatch" unless basis_buf.size >= (h_k * rank * s).to_i64 * sizeof(Float32)

          out_buf = ML::MetalBuffer.new((h_v * s * rank).to_i64 * sizeof(Float32))
          enc = ML::Metal::ComputeEncoder.new(cmd)
          enc.set_pipeline(lowrank_project_state_pipeline)
          enc.set_buffer(full_state_buf, 0)
          enc.set_buffer(basis_buf, 1)
          enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_value(h_k.to_u32, 3)
          enc.set_value(h_v.to_u32, 4)
          enc.set_value(s.to_u32, 5)
          enc.set_value(rank.to_u32, 6)
          enc.dispatch_threadgroups({(rank + 7) // 8, s, h_v}, {8, 1, 1})
          enc.end_encoding
          out_buf
        end

        def self.lowrank_reconstruct_state_append(lowrank_state_buf : ML::MetalBuffer,
                                                  basis_buf : ML::MetalBuffer,
                                                  full_state_buf : ML::MetalBuffer,
                                                  h_k : Int32, h_v : Int32, s : Int32,
                                                  rank : Int32,
                                                  cmd : ML::Metal::CommandBuffer) : Nil
          ML::Metal::Device.init!
          raise "lowrank_reconstruct_state rank must be positive" unless rank > 0
          raise "lowrank_reconstruct_state lowrank state size mismatch" unless lowrank_state_buf.size >= (h_v * s * rank).to_i64 * sizeof(Float32)
          raise "lowrank_reconstruct_state basis size mismatch" unless basis_buf.size >= (h_k * rank * s).to_i64 * sizeof(Float32)
          raise "lowrank_reconstruct_state full state size mismatch" unless full_state_buf.size >= (h_v * s * s).to_i64 * sizeof(Float32)

          enc = ML::Metal::ComputeEncoder.new(cmd)
          enc.set_pipeline(lowrank_reconstruct_state_pipeline)
          enc.set_buffer(lowrank_state_buf, 0)
          enc.set_buffer(basis_buf, 1)
          enc.set_buffer(full_state_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_value(h_k.to_u32, 3)
          enc.set_value(h_v.to_u32, 4)
          enc.set_value(s.to_u32, 5)
          enc.set_value(rank.to_u32, 6)
          enc.dispatch_threadgroups({(s + 7) // 8, s, h_v}, {8, 1, 1})
          enc.end_encoding
          nil
        end

        # Token-major chunk version of `lowrank_delta_step_projected_buf`.
        # Inputs contain precomputed recurrent q/k/v/g/beta for a known token
        # span. The low-rank M state is scanned in token order inside one command
        # buffer and `out` is returned as [n_tokens, h_v, s].
        def self.lowrank_delta_chunk_projected_buf(m_state_buf : ML::MetalBuffer,
                                                   q_conv : Array(Float32),
                                                   k_conv : Array(Float32),
                                                   basis_buf : ML::MetalBuffer,
                                                   v_conv : Array(Float32),
                                                   ghead : Array(Float32),
                                                   beta : Array(Float32),
                                                   h_k : Int32, h_v : Int32, s : Int32,
                                                   rank : Int32,
                                                   n_tokens : Int32,
                                                   scale : Float32) : Array(Float32)
          ML::Metal::Device.init!
          raise "lowrank_delta_chunk_projected_buf n_tokens must be positive" unless n_tokens > 0
          raise "lowrank_delta_chunk_projected_buf rank must be positive" unless rank > 0
          raise "lowrank_delta_chunk_projected_buf q size mismatch" unless q_conv.size == n_tokens * h_k * s
          raise "lowrank_delta_chunk_projected_buf k size mismatch" unless k_conv.size == n_tokens * h_k * s
          raise "lowrank_delta_chunk_projected_buf basis size mismatch" unless basis_buf.size >= (h_k * rank * s).to_i64 * sizeof(Float32)
          raise "lowrank_delta_chunk_projected_buf v size mismatch" unless v_conv.size == n_tokens * h_v * s
          raise "lowrank_delta_chunk_projected_buf g size mismatch" unless ghead.size == n_tokens * h_v
          raise "lowrank_delta_chunk_projected_buf beta size mismatch" unless beta.size == n_tokens * h_v
          raise "lowrank_delta_chunk_projected_buf state size mismatch" unless m_state_buf.size >= (h_v * s * rank).to_i64 * sizeof(Float32)

          q_buf = Scratch.get(:lowrank_chunk_q, q_conv.size.to_i64 * sizeof(Float32))
          k_buf = Scratch.get(:lowrank_chunk_k, k_conv.size.to_i64 * sizeof(Float32))
          c_buf = Scratch.get(:lowrank_chunk_c, (n_tokens * h_k * rank).to_i64 * sizeof(Float32))
          qbar_buf = Scratch.get(:lowrank_chunk_qbar, (n_tokens * h_k * rank).to_i64 * sizeof(Float32))
          v_buf = Scratch.get(:lowrank_chunk_v, v_conv.size.to_i64 * sizeof(Float32))
          g_buf = Scratch.get(:lowrank_chunk_g, ghead.size.to_i64 * sizeof(Float32))
          b_buf = Scratch.get(:lowrank_chunk_b, beta.size.to_i64 * sizeof(Float32))
          out_buf = Scratch.get(:lowrank_chunk_out, (n_tokens * h_v * s).to_i64 * sizeof(Float32))

          q_buf.write(q_conv)
          k_buf.write(k_conv)
          v_buf.write(v_conv)
          g_buf.write(ghead)
          b_buf.write(beta)

          cmd = ML::Metal::CommandBuffer.new

          proj_enc = ML::Metal::ComputeEncoder.new(cmd)
          proj_enc.set_pipeline(lowrank_project_coeffs_chunk_pipeline)
          proj_enc.set_buffer(q_buf, 0)
          proj_enc.set_buffer(k_buf, 1)
          proj_enc.set_buffer(basis_buf, 2)
          proj_enc.set_buffer(c_buf, 3, ML::Metal::BufferAccess::Write)
          proj_enc.set_buffer(qbar_buf, 4, ML::Metal::BufferAccess::Write)
          proj_enc.set_value(h_k.to_u32, 5)
          proj_enc.set_value(s.to_u32, 6)
          proj_enc.set_value(rank.to_u32, 7)
          proj_enc.set_value(n_tokens.to_u32, 8)
          proj_enc.dispatch_threadgroups({(rank + 7) // 8, h_k, n_tokens * 2}, {8, 1, 1})
          proj_enc.end_encoding

          scan_enc = ML::Metal::ComputeEncoder.new(cmd)
          scan_enc.set_pipeline(lowrank_delta_chunk_pipeline)
          scan_enc.set_buffer(m_state_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          scan_enc.set_buffer(c_buf, 1)
          scan_enc.set_buffer(qbar_buf, 2)
          scan_enc.set_buffer(v_buf, 3)
          scan_enc.set_buffer(g_buf, 4)
          scan_enc.set_buffer(b_buf, 5)
          scan_enc.set_buffer(out_buf, 6, ML::Metal::BufferAccess::Write)
          scan_enc.set_value(h_k.to_u32, 7)
          scan_enc.set_value(h_v.to_u32, 8)
          scan_enc.set_value(s.to_u32, 9)
          scan_enc.set_value(rank.to_u32, 10)
          scan_enc.set_value(n_tokens.to_u32, 11)
          scan_enc.set_value(scale, 12)
          scan_enc.dispatch_threadgroups({s, h_v, 1}, {128, 1, 1})
          scan_enc.end_encoding

          cmd.commit
          cmd.wait
          read_shared_f32(out_buf, n_tokens * h_v * s)
        end

        # Low-rank recurrent attention chunk through the output projection:
        #   project Q/K -> c/qbar -> low-rank M scan -> RMSNorm(y)*SiLU(z)
        #   -> ssm_out projection.
        #
        # Returns token-major `attn_out` [n_tokens, hidden_dim].
        def self.lowrank_delta_chunk_projected_out_buf(m_state_buf : ML::MetalBuffer,
                                                       q_conv : Array(Float32),
                                                       k_conv : Array(Float32),
                                                       basis_buf : ML::MetalBuffer,
                                                       v_conv : Array(Float32),
                                                       ghead : Array(Float32),
                                                       beta : Array(Float32),
                                                       z : Array(Float32),
                                                       ssm_norm : Array(Float32),
                                                       out_qw : QuantWeight,
                                                       h_k : Int32, h_v : Int32, s : Int32,
                                                       rank : Int32,
                                                       n_tokens : Int32,
                                                       eps : Float32,
                                                       scale : Float32) : Array(Float32)?
          ML::Metal::Device.init!
          out_pipe = gemv_pipeline_for(out_qw)
          return nil if out_pipe.nil?
          inner_dim = h_v * s
          raise "lowrank_delta_chunk_projected_out n_tokens must be positive" unless n_tokens > 0
          raise "lowrank_delta_chunk_projected_out rank must be positive" unless rank > 0
          raise "lowrank_delta_chunk_projected_out q size mismatch" unless q_conv.size == n_tokens * h_k * s
          raise "lowrank_delta_chunk_projected_out k size mismatch" unless k_conv.size == n_tokens * h_k * s
          raise "lowrank_delta_chunk_projected_out basis size mismatch" unless basis_buf.size >= (h_k * rank * s).to_i64 * sizeof(Float32)
          raise "lowrank_delta_chunk_projected_out v size mismatch" unless v_conv.size == n_tokens * h_v * s
          raise "lowrank_delta_chunk_projected_out g size mismatch" unless ghead.size == n_tokens * h_v
          raise "lowrank_delta_chunk_projected_out beta size mismatch" unless beta.size == n_tokens * h_v
          raise "lowrank_delta_chunk_projected_out z size mismatch" unless z.size == n_tokens * inner_dim
          raise "lowrank_delta_chunk_projected_out norm size mismatch" unless ssm_norm.size == s
          raise "lowrank_delta_chunk_projected_out projection in_dim mismatch" unless out_qw.in_dim == inner_dim
          raise "lowrank_delta_chunk_projected_out state size mismatch" unless m_state_buf.size >= (h_v * s * rank).to_i64 * sizeof(Float32)

          q_buf = Scratch.get(:lowrank_chunk_q, q_conv.size.to_i64 * sizeof(Float32))
          k_buf = Scratch.get(:lowrank_chunk_k, k_conv.size.to_i64 * sizeof(Float32))
          c_buf = Scratch.get(:lowrank_chunk_c, (n_tokens * h_k * rank).to_i64 * sizeof(Float32))
          qbar_buf = Scratch.get(:lowrank_chunk_qbar, (n_tokens * h_k * rank).to_i64 * sizeof(Float32))
          v_buf = Scratch.get(:lowrank_chunk_v, v_conv.size.to_i64 * sizeof(Float32))
          g_buf = Scratch.get(:lowrank_chunk_g, ghead.size.to_i64 * sizeof(Float32))
          b_buf = Scratch.get(:lowrank_chunk_b, beta.size.to_i64 * sizeof(Float32))
          z_buf = Scratch.get(:lowrank_chunk_z, z.size.to_i64 * sizeof(Float32))
          norm_buf = Scratch.get(:lowrank_chunk_norm, ssm_norm.size.to_i64 * sizeof(Float32))
          mid_buf = Scratch.get(:lowrank_chunk_out, (n_tokens * inner_dim).to_i64 * sizeof(Float32))
          attn_out_buf = Scratch.get(:lowrank_chunk_attn_out, (n_tokens * out_qw.out_dim).to_i64 * sizeof(Float32))

          q_buf.write(q_conv)
          k_buf.write(k_conv)
          v_buf.write(v_conv)
          g_buf.write(ghead)
          b_buf.write(beta)
          z_buf.write(z)
          norm_buf.write(ssm_norm)
          out_w_buf, out_w_off = weight_slot(out_qw)

          cmd = ML::Metal::CommandBuffer.new

          proj_enc = ML::Metal::ComputeEncoder.new(cmd)
          proj_enc.set_pipeline(lowrank_project_coeffs_chunk_pipeline)
          proj_enc.set_buffer(q_buf, 0)
          proj_enc.set_buffer(k_buf, 1)
          proj_enc.set_buffer(basis_buf, 2)
          proj_enc.set_buffer(c_buf, 3, ML::Metal::BufferAccess::Write)
          proj_enc.set_buffer(qbar_buf, 4, ML::Metal::BufferAccess::Write)
          proj_enc.set_value(h_k.to_u32, 5)
          proj_enc.set_value(s.to_u32, 6)
          proj_enc.set_value(rank.to_u32, 7)
          proj_enc.set_value(n_tokens.to_u32, 8)
          proj_enc.dispatch_threadgroups({(rank + 7) // 8, h_k, n_tokens * 2}, {8, 1, 1})
          proj_enc.end_encoding

          scan_enc = ML::Metal::ComputeEncoder.new(cmd)
          scan_enc.set_pipeline(lowrank_delta_chunk_pipeline)
          scan_enc.set_buffer(m_state_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          scan_enc.set_buffer(c_buf, 1)
          scan_enc.set_buffer(qbar_buf, 2)
          scan_enc.set_buffer(v_buf, 3)
          scan_enc.set_buffer(g_buf, 4)
          scan_enc.set_buffer(b_buf, 5)
          scan_enc.set_buffer(mid_buf, 6, ML::Metal::BufferAccess::Write)
          scan_enc.set_value(h_k.to_u32, 7)
          scan_enc.set_value(h_v.to_u32, 8)
          scan_enc.set_value(s.to_u32, 9)
          scan_enc.set_value(rank.to_u32, 10)
          scan_enc.set_value(n_tokens.to_u32, 11)
          scan_enc.set_value(scale, 12)
          scan_enc.dispatch_threadgroups({s, h_v, 1}, {128, 1, 1})
          scan_enc.end_encoding

          post_enc = ML::Metal::ComputeEncoder.new(cmd)
          post_enc.set_pipeline(dn_post_chunk_pipeline)
          post_enc.set_buffer(mid_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          post_enc.set_buffer(z_buf, 1)
          post_enc.set_buffer(norm_buf, 2)
          post_enc.set_value(h_v.to_u32, 3)
          post_enc.set_value(s.to_u32, 4)
          post_enc.set_value(eps, 5)
          post_enc.set_value(n_tokens.to_u32, 6)
          post_enc.dispatch_threadgroups({h_v, n_tokens, 1}, {32, 1, 1})
          post_enc.end_encoding

          out_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_matmul(out_enc, out_pipe.not_nil!, out_qw, mid_buf, attn_out_buf, out_w_buf, out_w_off, out_qw.in_dim, out_qw.out_dim, n_tokens)
          out_enc.end_encoding

          cmd.commit
          cmd.wait
          read_shared_f32(attn_out_buf, n_tokens * out_qw.out_dim)
        end

        # Same low-rank recurrent attention chunk as above, fused through the
        # post-attention residual/RMSNorm and FFN tail. This is a known-token-span
        # building block for the self-spec draft/verifier lane experiments.
        def self.lowrank_delta_chunk_projected_layer_buf(m_state_buf : ML::MetalBuffer,
                                                         inp : Array(Float32),
                                                         q_conv : Array(Float32),
                                                         k_conv : Array(Float32),
                                                         basis_buf : ML::MetalBuffer,
                                                         v_conv : Array(Float32),
                                                         ghead : Array(Float32),
                                                         beta : Array(Float32),
                                                         z : Array(Float32),
                                                         ssm_norm : Array(Float32),
                                                         out_qw : QuantWeight,
                                                         post_attention_norm : Array(Float32),
                                                         ffn_gate_qw : QuantWeight,
                                                         ffn_up_qw : QuantWeight,
                                                         ffn_down_qw : QuantWeight,
                                                         h_k : Int32, h_v : Int32, s : Int32,
                                                         rank : Int32,
                                                         n_tokens : Int32,
                                                         eps : Float32,
                                                         scale : Float32) : Array(Float32)?
          submission = lowrank_delta_chunk_projected_layer_async(m_state_buf, inp, q_conv, k_conv, basis_buf, v_conv, ghead, beta, z,
            ssm_norm, out_qw, post_attention_norm, ffn_gate_qw, ffn_up_qw, ffn_down_qw,
            h_k, h_v, s, rank, n_tokens, eps, scale)
          return nil if submission.nil?
          wait_lowrank_layer_chunk(submission)
        end

        def self.lowrank_delta_chunk_projected_layer_updown_buf(m_state_buf : ML::MetalBuffer,
                                                                inp : Array(Float32),
                                                                q_conv : Array(Float32),
                                                                k_conv : Array(Float32),
                                                                basis_buf : ML::MetalBuffer,
                                                                v_conv : Array(Float32),
                                                                ghead : Array(Float32),
                                                                beta : Array(Float32),
                                                                z : Array(Float32),
                                                                ssm_norm : Array(Float32),
                                                                out_qw : QuantWeight,
                                                                post_attention_norm : Array(Float32),
                                                                ffn_gate_qw : QuantWeight,
                                                                ffn_up_qw : QuantWeight,
                                                                ffn_down_qw : QuantWeight,
                                                                updown_x_mean_buf : ML::MetalBuffer,
                                                                updown_c_mean_buf : ML::MetalBuffer,
                                                                updown_coeff_w_buf : ML::MetalBuffer,
                                                                updown_down_buf : ML::MetalBuffer,
                                                                h_k : Int32, h_v : Int32, s : Int32,
                                                                rank : Int32,
                                                                n_tokens : Int32,
                                                                updown_rank : Int32,
                                                                eps : Float32,
                                                                scale : Float32) : Array(Float32)?
          submission = lowrank_delta_chunk_projected_layer_async(m_state_buf, inp, q_conv, k_conv, basis_buf, v_conv, ghead, beta, z,
            ssm_norm, out_qw, post_attention_norm, ffn_gate_qw, ffn_up_qw, ffn_down_qw,
            h_k, h_v, s, rank, n_tokens, eps, scale,
            updown_x_mean_buf: updown_x_mean_buf,
            updown_c_mean_buf: updown_c_mean_buf,
            updown_coeff_w_buf: updown_coeff_w_buf,
            updown_down_buf: updown_down_buf,
            updown_rank: updown_rank)
          return nil if submission.nil?
          wait_lowrank_layer_chunk(submission)
        end

        # Probe-only FFN PCA up/down microkernel:
        #   coeffs = c_mean + (x - x_mean) * coeff_weights^T
        #   out    = coeffs * down_basis
        #
        # This is the cheap draft FFN tail used by `lowrank-ffn-pca-updown-R`.
        # The current wrapper uploads and reads back for correctness/timing
        # gates; production use should keep adapter buffers GPU-resident.
        def self.ffn_pca_updown_out(x : Array(Float32),
                                    x_mean : Array(Float32),
                                    c_mean : Array(Float32),
                                    coeff_weights : Array(Float32),
                                    down_basis : Array(Float32),
                                    hidden_dim : Int32,
                                    rank : Int32) : Array(Float32)
          ML::Metal::Device.init!
          raise "ffn_pca_updown hidden_dim must be positive" unless hidden_dim > 0
          raise "ffn_pca_updown rank must be positive" unless rank > 0
          raise "ffn_pca_updown x size mismatch" unless x.size == hidden_dim
          raise "ffn_pca_updown x_mean size mismatch" unless x_mean.size == hidden_dim
          raise "ffn_pca_updown c_mean size mismatch" unless c_mean.size >= rank
          raise "ffn_pca_updown coeff_weights size mismatch" unless coeff_weights.size >= rank * hidden_dim
          raise "ffn_pca_updown down_basis size mismatch" unless down_basis.size >= rank * hidden_dim

          x_buf = Scratch.get(:ffn_updown_x, hidden_dim.to_i64 * sizeof(Float32))
          x_mean_buf = Scratch.get(:ffn_updown_x_mean, hidden_dim.to_i64 * sizeof(Float32))
          c_mean_buf = Scratch.get(:ffn_updown_c_mean, rank.to_i64 * sizeof(Float32))
          coeff_w_buf = Scratch.get(:ffn_updown_coeff_w, (rank * hidden_dim).to_i64 * sizeof(Float32))
          down_buf = Scratch.get(:ffn_updown_down, (rank * hidden_dim).to_i64 * sizeof(Float32))
          coeff_buf = Scratch.get(:ffn_updown_coeffs, rank.to_i64 * sizeof(Float32))
          out_buf = Scratch.get(:ffn_updown_out, hidden_dim.to_i64 * sizeof(Float32))

          x_buf.write(x)
          x_mean_buf.write(x_mean)
          c_mean_buf.write(c_mean[0, rank])
          coeff_w_buf.write(coeff_weights[0, rank * hidden_dim])
          down_buf.write(down_basis[0, rank * hidden_dim])

          cmd = ML::Metal::CommandBuffer.new

          coeff_enc = ML::Metal::ComputeEncoder.new(cmd)
          coeff_enc.set_pipeline(ffn_pca_updown_coeffs_pipeline)
          coeff_enc.set_buffer(x_buf, 0)
          coeff_enc.set_buffer(x_mean_buf, 1)
          coeff_enc.set_buffer(c_mean_buf, 2)
          coeff_enc.set_buffer(coeff_w_buf, 3)
          coeff_enc.set_buffer(coeff_buf, 4, ML::Metal::BufferAccess::Write)
          coeff_enc.set_value(hidden_dim.to_u32, 5)
          coeff_enc.set_value(rank.to_u32, 6)
          coeff_enc.dispatch_1d(rank, 64)
          coeff_enc.end_encoding

          out_enc = ML::Metal::ComputeEncoder.new(cmd)
          out_enc.set_pipeline(ffn_pca_updown_out_pipeline)
          out_enc.set_buffer(coeff_buf, 0)
          out_enc.set_buffer(down_buf, 1)
          out_enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          out_enc.set_value(hidden_dim.to_u32, 3)
          out_enc.set_value(rank.to_u32, 4)
          out_enc.dispatch_1d(hidden_dim, 256)
          out_enc.end_encoding

          cmd.commit
          cmd.wait
          read_shared_f32(out_buf, hidden_dim)
        end

        def self.ffn_pca_updown_out_resident(x : Array(Float32),
                                             x_mean_buf : ML::MetalBuffer,
                                             c_mean_buf : ML::MetalBuffer,
                                             coeff_w_buf : ML::MetalBuffer,
                                             down_buf : ML::MetalBuffer,
                                             hidden_dim : Int32,
                                             rank : Int32) : Array(Float32)
          ML::Metal::Device.init!
          raise "ffn_pca_updown resident hidden_dim must be positive" unless hidden_dim > 0
          raise "ffn_pca_updown resident rank must be positive" unless rank > 0
          raise "ffn_pca_updown resident x size mismatch" unless x.size == hidden_dim
          raise "ffn_pca_updown resident x_mean buffer too small" unless x_mean_buf.size >= hidden_dim.to_i64 * sizeof(Float32)
          raise "ffn_pca_updown resident c_mean buffer too small" unless c_mean_buf.size >= rank.to_i64 * sizeof(Float32)
          raise "ffn_pca_updown resident coeff buffer too small" unless coeff_w_buf.size >= (rank * hidden_dim).to_i64 * sizeof(Float32)
          raise "ffn_pca_updown resident down buffer too small" unless down_buf.size >= (rank * hidden_dim).to_i64 * sizeof(Float32)

          x_buf = Scratch.get(:ffn_updown_resident_x, hidden_dim.to_i64 * sizeof(Float32))
          out_buf = Scratch.get(:ffn_updown_resident_out, hidden_dim.to_i64 * sizeof(Float32))
          x_buf.write(x)

          cmd = ML::Metal::CommandBuffer.new

          enc = ML::Metal::ComputeEncoder.new(cmd)
          enc.set_pipeline(ffn_pca_updown_fused_pipeline)
          enc.set_buffer(x_buf, 0)
          enc.set_buffer(x_mean_buf, 1)
          enc.set_buffer(c_mean_buf, 2)
          enc.set_buffer(coeff_w_buf, 3)
          enc.set_buffer(down_buf, 4)
          enc.set_buffer(out_buf, 5, ML::Metal::BufferAccess::Write)
          enc.set_value(hidden_dim.to_u32, 6)
          enc.set_value(rank.to_u32, 7)
          enc.dispatch_threadgroups({1, 1, 1}, {256, 1, 1})
          enc.end_encoding

          cmd.commit
          cmd.wait
          read_shared_f32(out_buf, hidden_dim)
        end

        def self.lowrank_delta_chunk_projected_layer_async(m_state_buf : ML::MetalBuffer,
                                                           inp : Array(Float32),
                                                           q_conv : Array(Float32),
                                                           k_conv : Array(Float32),
                                                           basis_buf : ML::MetalBuffer,
                                                           v_conv : Array(Float32),
                                                           ghead : Array(Float32),
                                                           beta : Array(Float32),
                                                           z : Array(Float32),
                                                           ssm_norm : Array(Float32),
                                                           out_qw : QuantWeight,
                                                           post_attention_norm : Array(Float32),
                                                           ffn_gate_qw : QuantWeight,
                                                           ffn_up_qw : QuantWeight,
                                                           ffn_down_qw : QuantWeight,
                                                           h_k : Int32, h_v : Int32, s : Int32,
                                                           rank : Int32,
                                                           n_tokens : Int32,
                                                           eps : Float32,
                                                           scale : Float32,
                                                           fresh_scratch : Bool = false,
                                                           scratch_namespace : String? = nil,
                                                           command_queue_name : String? = nil,
                                                           retained_scratch : Array(ML::MetalBuffer)? = nil,
                                                           updown_x_mean_buf : ML::MetalBuffer? = nil,
                                                           updown_c_mean_buf : ML::MetalBuffer? = nil,
                                                           updown_coeff_w_buf : ML::MetalBuffer? = nil,
                                                           updown_down_buf : ML::MetalBuffer? = nil,
                                                           updown_rank : Int32 = 0) : LowRankLayerChunkSubmission?
          if fresh_scratch
            return Scratch.with_fresh do |retained|
              lowrank_delta_chunk_projected_layer_async(m_state_buf, inp, q_conv, k_conv, basis_buf, v_conv, ghead, beta, z,
                ssm_norm, out_qw, post_attention_norm, ffn_gate_qw, ffn_up_qw, ffn_down_qw,
                h_k, h_v, s, rank, n_tokens, eps, scale, command_queue_name: command_queue_name, retained_scratch: retained,
                updown_x_mean_buf: updown_x_mean_buf, updown_c_mean_buf: updown_c_mean_buf,
                updown_coeff_w_buf: updown_coeff_w_buf, updown_down_buf: updown_down_buf, updown_rank: updown_rank)
            end
          end
          if namespace = scratch_namespace
            return Scratch.with_namespace(namespace) do
              lowrank_delta_chunk_projected_layer_async(m_state_buf, inp, q_conv, k_conv, basis_buf, v_conv, ghead, beta, z,
                ssm_norm, out_qw, post_attention_norm, ffn_gate_qw, ffn_up_qw, ffn_down_qw,
                h_k, h_v, s, rank, n_tokens, eps, scale, command_queue_name: command_queue_name,
                updown_x_mean_buf: updown_x_mean_buf, updown_c_mean_buf: updown_c_mean_buf,
                updown_coeff_w_buf: updown_coeff_w_buf, updown_down_buf: updown_down_buf, updown_rank: updown_rank)
            end
          end
          ML::Metal::Device.init!
          use_updown = !updown_x_mean_buf.nil? && !updown_c_mean_buf.nil? && !updown_coeff_w_buf.nil? && !updown_down_buf.nil? && updown_rank > 0
          out_pipe = gemv_pipeline_for(out_qw)
          ffn_gate_pipe = use_updown ? nil : gemv_pipeline_for(ffn_gate_qw)
          ffn_up_pipe = use_updown ? nil : gemv_pipeline_for(ffn_up_qw)
          ffn_down_pipe = use_updown ? nil : gemv_pipeline_for(ffn_down_qw)
          return nil if out_pipe.nil?
          return nil if !use_updown && (ffn_gate_pipe.nil? || ffn_up_pipe.nil? || ffn_down_pipe.nil?)
          inner_dim = h_v * s
          hidden_dim = out_qw.out_dim
          ffn_dim = ffn_gate_qw.out_dim
          raise "lowrank_delta_chunk_projected_layer n_tokens must be positive" unless n_tokens > 0
          raise "lowrank_delta_chunk_projected_layer rank must be positive" unless rank > 0
          raise "lowrank_delta_chunk_projected_layer input size mismatch" unless inp.size == n_tokens * hidden_dim
          raise "lowrank_delta_chunk_projected_layer q size mismatch" unless q_conv.size == n_tokens * h_k * s
          raise "lowrank_delta_chunk_projected_layer k size mismatch" unless k_conv.size == n_tokens * h_k * s
          raise "lowrank_delta_chunk_projected_layer basis size mismatch" unless basis_buf.size >= (h_k * rank * s).to_i64 * sizeof(Float32)
          raise "lowrank_delta_chunk_projected_layer v size mismatch" unless v_conv.size == n_tokens * h_v * s
          raise "lowrank_delta_chunk_projected_layer g size mismatch" unless ghead.size == n_tokens * h_v
          raise "lowrank_delta_chunk_projected_layer beta size mismatch" unless beta.size == n_tokens * h_v
          raise "lowrank_delta_chunk_projected_layer z size mismatch" unless z.size == n_tokens * inner_dim
          raise "lowrank_delta_chunk_projected_layer norm size mismatch" unless ssm_norm.size == s
          raise "lowrank_delta_chunk_projected_layer post norm size mismatch" unless post_attention_norm.size == hidden_dim
          raise "lowrank_delta_chunk_projected_layer projection in_dim mismatch" unless out_qw.in_dim == inner_dim
          raise "lowrank_delta_chunk_projected_layer ffn gate/up mismatch" unless ffn_gate_qw.in_dim == hidden_dim && ffn_up_qw.in_dim == hidden_dim && ffn_gate_qw.out_dim == ffn_up_qw.out_dim
          raise "lowrank_delta_chunk_projected_layer ffn down mismatch" unless ffn_down_qw.in_dim == ffn_dim && ffn_down_qw.out_dim == hidden_dim
          if use_updown
            raise "lowrank_delta_chunk_projected_layer updown rank too large" if updown_rank > 64
            raise "lowrank_delta_chunk_projected_layer updown x_mean buffer too small" unless updown_x_mean_buf.not_nil!.size >= hidden_dim.to_i64 * sizeof(Float32)
            raise "lowrank_delta_chunk_projected_layer updown c_mean buffer too small" unless updown_c_mean_buf.not_nil!.size >= updown_rank.to_i64 * sizeof(Float32)
            raise "lowrank_delta_chunk_projected_layer updown coeff buffer too small" unless updown_coeff_w_buf.not_nil!.size >= (updown_rank * hidden_dim).to_i64 * sizeof(Float32)
            raise "lowrank_delta_chunk_projected_layer updown down buffer too small" unless updown_down_buf.not_nil!.size >= (updown_rank * hidden_dim).to_i64 * sizeof(Float32)
          end
          raise "lowrank_delta_chunk_projected_layer state size mismatch" unless m_state_buf.size >= (h_v * s * rank).to_i64 * sizeof(Float32)

          inp_buf = Scratch.get(:lowrank_layer_inp, inp.size.to_i64 * sizeof(Float32))
          q_buf = Scratch.get(:lowrank_layer_q, q_conv.size.to_i64 * sizeof(Float32))
          k_buf = Scratch.get(:lowrank_layer_k, k_conv.size.to_i64 * sizeof(Float32))
          c_buf = Scratch.get(:lowrank_layer_c, (n_tokens * h_k * rank).to_i64 * sizeof(Float32))
          qbar_buf = Scratch.get(:lowrank_layer_qbar, (n_tokens * h_k * rank).to_i64 * sizeof(Float32))
          v_buf = Scratch.get(:lowrank_layer_v, v_conv.size.to_i64 * sizeof(Float32))
          g_buf = Scratch.get(:lowrank_layer_g, ghead.size.to_i64 * sizeof(Float32))
          b_buf = Scratch.get(:lowrank_layer_b, beta.size.to_i64 * sizeof(Float32))
          z_buf = Scratch.get(:lowrank_layer_z, z.size.to_i64 * sizeof(Float32))
          norm_buf = Scratch.get(:lowrank_layer_norm, ssm_norm.size.to_i64 * sizeof(Float32))
          mid_buf = Scratch.get(:lowrank_layer_mid, (n_tokens * inner_dim).to_i64 * sizeof(Float32))
          attn_out_buf = Scratch.get(:lowrank_layer_attn_out, (n_tokens * hidden_dim).to_i64 * sizeof(Float32))
          post_w_buf = Scratch.get(:lowrank_layer_post_w, post_attention_norm.size.to_i64 * sizeof(Float32))
          residual_buf = Scratch.get(:lowrank_layer_residual, inp.size.to_i64 * sizeof(Float32))
          normed_buf = Scratch.get(:lowrank_layer_normed, inp.size.to_i64 * sizeof(Float32))
          ffn_gate_buf = Scratch.get(:lowrank_layer_ffn_gate, (n_tokens * ffn_dim).to_i64 * sizeof(Float32))
          ffn_up_buf = Scratch.get(:lowrank_layer_ffn_up, (n_tokens * ffn_dim).to_i64 * sizeof(Float32))
          ffn_comb_buf = Scratch.get(:lowrank_layer_ffn_comb, (n_tokens * ffn_dim).to_i64 * sizeof(Float32))
          ffn_out_buf = Scratch.get(:lowrank_layer_ffn_out, (n_tokens * hidden_dim).to_i64 * sizeof(Float32))
          out_buf = Scratch.get(:lowrank_layer_out, inp.size.to_i64 * sizeof(Float32))

          inp_buf.write(inp)
          q_buf.write(q_conv)
          k_buf.write(k_conv)
          v_buf.write(v_conv)
          g_buf.write(ghead)
          b_buf.write(beta)
          z_buf.write(z)
          norm_buf.write(ssm_norm)
          post_w_buf.write(post_attention_norm)
          out_w_buf, out_w_off = weight_slot(out_qw)
          ffn_gate_w_buf, ffn_gate_w_off = weight_slot(ffn_gate_qw)
          ffn_up_w_buf, ffn_up_w_off = weight_slot(ffn_up_qw)
          ffn_down_w_buf, ffn_down_w_off = weight_slot(ffn_down_qw)

          cmd_queue = command_queue_name ? lane_command_queue(command_queue_name.not_nil!) : nil
          cmd = ML::Metal::CommandBuffer.new(queue: cmd_queue)

          proj_enc = ML::Metal::ComputeEncoder.new(cmd)
          proj_enc.set_pipeline(lowrank_project_coeffs_chunk_pipeline)
          proj_enc.set_buffer(q_buf, 0)
          proj_enc.set_buffer(k_buf, 1)
          proj_enc.set_buffer(basis_buf, 2)
          proj_enc.set_buffer(c_buf, 3, ML::Metal::BufferAccess::Write)
          proj_enc.set_buffer(qbar_buf, 4, ML::Metal::BufferAccess::Write)
          proj_enc.set_value(h_k.to_u32, 5)
          proj_enc.set_value(s.to_u32, 6)
          proj_enc.set_value(rank.to_u32, 7)
          proj_enc.set_value(n_tokens.to_u32, 8)
          proj_enc.dispatch_threadgroups({(rank + 7) // 8, h_k, n_tokens * 2}, {8, 1, 1})
          proj_enc.end_encoding

          scan_enc = ML::Metal::ComputeEncoder.new(cmd)
          scan_enc.set_pipeline(lowrank_delta_chunk_pipeline)
          scan_enc.set_buffer(m_state_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          scan_enc.set_buffer(c_buf, 1)
          scan_enc.set_buffer(qbar_buf, 2)
          scan_enc.set_buffer(v_buf, 3)
          scan_enc.set_buffer(g_buf, 4)
          scan_enc.set_buffer(b_buf, 5)
          scan_enc.set_buffer(mid_buf, 6, ML::Metal::BufferAccess::Write)
          scan_enc.set_value(h_k.to_u32, 7)
          scan_enc.set_value(h_v.to_u32, 8)
          scan_enc.set_value(s.to_u32, 9)
          scan_enc.set_value(rank.to_u32, 10)
          scan_enc.set_value(n_tokens.to_u32, 11)
          scan_enc.set_value(scale, 12)
          scan_enc.dispatch_threadgroups({s, h_v, 1}, {128, 1, 1})
          scan_enc.end_encoding

          post_enc = ML::Metal::ComputeEncoder.new(cmd)
          post_enc.set_pipeline(dn_post_chunk_pipeline)
          post_enc.set_buffer(mid_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          post_enc.set_buffer(z_buf, 1)
          post_enc.set_buffer(norm_buf, 2)
          post_enc.set_value(h_v.to_u32, 3)
          post_enc.set_value(s.to_u32, 4)
          post_enc.set_value(eps, 5)
          post_enc.set_value(n_tokens.to_u32, 6)
          post_enc.dispatch_threadgroups({h_v, n_tokens, 1}, {32, 1, 1})
          post_enc.end_encoding

          out_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_matmul(out_enc, out_pipe.not_nil!, out_qw, mid_buf, attn_out_buf, out_w_buf, out_w_off, out_qw.in_dim, out_qw.out_dim, n_tokens)
          out_enc.end_encoding

          addnorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_add_rmsnorm_rows(addnorm_enc, inp_buf, attn_out_buf, post_w_buf, residual_buf, normed_buf, hidden_dim, n_tokens, eps)
          addnorm_enc.end_encoding

          if use_updown
            updown_enc = ML::Metal::ComputeEncoder.new(cmd)
            updown_enc.set_pipeline(ffn_pca_updown_fused_rows_pipeline)
            updown_enc.set_buffer(normed_buf, 0)
            updown_enc.set_buffer(updown_x_mean_buf.not_nil!, 1)
            updown_enc.set_buffer(updown_c_mean_buf.not_nil!, 2)
            updown_enc.set_buffer(updown_coeff_w_buf.not_nil!, 3)
            updown_enc.set_buffer(updown_down_buf.not_nil!, 4)
            updown_enc.set_buffer(ffn_out_buf, 5, ML::Metal::BufferAccess::Write)
            updown_enc.set_value(hidden_dim.to_u32, 6)
            updown_enc.set_value(updown_rank.to_u32, 7)
            updown_enc.set_value(n_tokens.to_u32, 8)
            updown_enc.dispatch_threadgroups({n_tokens, 1, 1}, {256, 1, 1})
            updown_enc.end_encoding

            add_enc = ML::Metal::ComputeEncoder.new(cmd)
            add_enc.set_pipeline(add_vec_pipeline)
            add_enc.set_buffer(residual_buf, 0)
            add_enc.set_buffer(ffn_out_buf, 1)
            add_enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
            add_enc.set_value((n_tokens * hidden_dim).to_u32, 3)
            add_enc.dispatch_1d(n_tokens * hidden_dim, 256)
            add_enc.end_encoding
          else
            ffn_proj_enc = ML::Metal::ComputeEncoder.new(cmd)
            encode_matmul(ffn_proj_enc, ffn_gate_pipe.not_nil!, ffn_gate_qw, normed_buf, ffn_gate_buf, ffn_gate_w_buf, ffn_gate_w_off, ffn_gate_qw.in_dim, ffn_gate_qw.out_dim, n_tokens)
            encode_matmul(ffn_proj_enc, ffn_up_pipe.not_nil!, ffn_up_qw, normed_buf, ffn_up_buf, ffn_up_w_buf, ffn_up_w_off, ffn_up_qw.in_dim, ffn_up_qw.out_dim, n_tokens)
            ffn_proj_enc.end_encoding

            swiglu_enc = ML::Metal::ComputeEncoder.new(cmd)
            swiglu_enc.set_pipeline(ffn_swiglu_pipeline)
            swiglu_enc.set_buffer(ffn_gate_buf, 0)
            swiglu_enc.set_buffer(ffn_up_buf, 1)
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
              add_enc.set_buffer(ffn_out_buf, 1)
              add_enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
              add_enc.set_value((n_tokens * hidden_dim).to_u32, 3)
              add_enc.dispatch_1d(n_tokens * hidden_dim, 256)
              add_enc.end_encoding
            end
          end

          cmd.commit
          LowRankLayerChunkSubmission.new(cmd, out_buf, n_tokens * hidden_dim, retained_scratch || [] of ML::MetalBuffer)
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

        def self.delta_net_chunk_checkpoint(state_buf : ML::MetalBuffer,
                                            q_conv : Array(Float32),
                                            k_conv : Array(Float32),
                                            v_conv : Array(Float32),
                                            ghead : Array(Float32),
                                            beta : Array(Float32),
                                            h_k : Int32, h_v : Int32, s : Int32,
                                            n_tokens : Int32,
                                            scale : Float32,
                                            checkpoint_index : Int32) : NamedTuple(out: Array(Float32), checkpoint_state: Array(Float32))
          ML::Metal::Device.init!
          raise "delta_net_chunk_checkpoint currently requires rowwise s=128" unless s == 128
          raise "delta_net_chunk_checkpoint n_tokens must be positive" unless n_tokens > 0
          raise "delta_net_chunk_checkpoint checkpoint index out of range" unless checkpoint_index >= 0 && checkpoint_index < n_tokens
          raise "delta_net_chunk_checkpoint q size mismatch" unless q_conv.size == n_tokens * h_k * s
          raise "delta_net_chunk_checkpoint k size mismatch" unless k_conv.size == n_tokens * h_k * s
          raise "delta_net_chunk_checkpoint v size mismatch" unless v_conv.size == n_tokens * h_v * s
          raise "delta_net_chunk_checkpoint g size mismatch" unless ghead.size == n_tokens * h_v
          raise "delta_net_chunk_checkpoint beta size mismatch" unless beta.size == n_tokens * h_v

          q_buf   = Scratch.get(:dn_chunk_checkpoint_q, q_conv.size.to_i64 * sizeof(Float32))
          k_buf   = Scratch.get(:dn_chunk_checkpoint_k, k_conv.size.to_i64 * sizeof(Float32))
          v_buf   = Scratch.get(:dn_chunk_checkpoint_v, v_conv.size.to_i64 * sizeof(Float32))
          g_buf   = Scratch.get(:dn_chunk_checkpoint_g, ghead.size.to_i64 * sizeof(Float32))
          b_buf   = Scratch.get(:dn_chunk_checkpoint_b, beta.size.to_i64 * sizeof(Float32))
          out_buf = Scratch.get(:dn_chunk_checkpoint_out, (n_tokens * h_v * s).to_i64 * sizeof(Float32))
          checkpoint_buf = Scratch.get(:dn_chunk_checkpoint_state, (h_v * s * s).to_i64 * sizeof(Float32))

          q_buf.write(q_conv)
          k_buf.write(k_conv)
          v_buf.write(v_conv)
          g_buf.write(ghead)
          b_buf.write(beta)

          cmd = ML::Metal::CommandBuffer.new
          enc = ML::Metal::ComputeEncoder.new(cmd)
          enc.set_pipeline(dn128_chunk_rowwise_checkpoint_pipeline)
          enc.set_buffer(state_buf,      0, ML::Metal::BufferAccess::ReadWrite)
          enc.set_buffer(q_buf,          1)
          enc.set_buffer(k_buf,          2)
          enc.set_buffer(v_buf,          3)
          enc.set_buffer(g_buf,          4)
          enc.set_buffer(b_buf,          5)
          enc.set_buffer(out_buf,        6, ML::Metal::BufferAccess::Write)
          enc.set_value(h_k.to_u32,      7)
          enc.set_value(h_v.to_u32,      8)
          enc.set_value(s.to_u32,        9)
          enc.set_value(scale,          10)
          enc.set_value(n_tokens.to_u32, 11)
          enc.set_buffer(checkpoint_buf, 12, ML::Metal::BufferAccess::Write)
          enc.set_value(checkpoint_index.to_u32, 13)
          enc.dispatch_threadgroups({(s + 3) // 4, h_v, 1}, {32, 4, 1})
          enc.end_encoding

          cmd.commit
          cmd.wait

          {
            out:              read_shared_f32(out_buf, n_tokens * h_v * s),
            checkpoint_state: read_shared_f32(checkpoint_buf, h_v * s * s),
          }
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

        def self.recurrent_prep_chunk_checkpoint(conv_state_buf : ML::MetalBuffer,
                                                 qkv_mixed : Array(Float32),
                                                 alpha : Array(Float32),
                                                 beta : Array(Float32),
                                                 ssm_conv1d : Array(Float32),
                                                 ssm_dt_bias : Array(Float32),
                                                 ssm_a : Array(Float32),
                                                 h_k : Int32, h_v : Int32, s : Int32,
                                                 conv_k : Int32,
                                                 n_tokens : Int32,
                                                 eps : Float32,
                                                 checkpoint_index : Int32) : NamedTuple(q: Array(Float32), k: Array(Float32), v: Array(Float32), g: Array(Float32), beta: Array(Float32), checkpoint_conv: Array(Float32))
          ML::Metal::Device.init!
          qkv_dim = 2 * h_k * s + h_v * s
          q_dim = h_k * s
          v_dim = h_v * s
          raise "recurrent_prep_chunk_checkpoint n_tokens must be positive" unless n_tokens > 0
          raise "recurrent_prep_chunk_checkpoint checkpoint index out of range" unless checkpoint_index >= 0 && checkpoint_index < n_tokens
          raise "recurrent_prep_chunk_checkpoint needs a non-empty conv state" unless conv_k > 1
          raise "recurrent_prep_chunk_checkpoint qkv size mismatch" unless qkv_mixed.size == n_tokens * qkv_dim
          raise "recurrent_prep_chunk_checkpoint alpha size mismatch" unless alpha.size == n_tokens * h_v
          raise "recurrent_prep_chunk_checkpoint beta size mismatch" unless beta.size == n_tokens * h_v
          raise "recurrent_prep_chunk_checkpoint conv1d size mismatch" unless ssm_conv1d.size == qkv_dim * conv_k

          qkv_buf = Scratch.get(:rec_chunk_checkpoint_qkv, qkv_mixed.size.to_i64 * sizeof(Float32))
          conv_w_buf = Scratch.get(:rec_chunk_checkpoint_conv_w, ssm_conv1d.size.to_i64 * sizeof(Float32))
          q_buf = Scratch.get(:rec_chunk_checkpoint_q, (n_tokens * q_dim).to_i64 * sizeof(Float32))
          k_buf = Scratch.get(:rec_chunk_checkpoint_k, (n_tokens * q_dim).to_i64 * sizeof(Float32))
          v_buf = Scratch.get(:rec_chunk_checkpoint_v, (n_tokens * v_dim).to_i64 * sizeof(Float32))
          alpha_buf = Scratch.get(:rec_chunk_checkpoint_alpha, alpha.size.to_i64 * sizeof(Float32))
          beta_buf = Scratch.get(:rec_chunk_checkpoint_beta, beta.size.to_i64 * sizeof(Float32))
          dt_bias_buf = Scratch.get(:rec_chunk_checkpoint_dt_bias, ssm_dt_bias.size.to_i64 * sizeof(Float32))
          ssm_a_buf = Scratch.get(:rec_chunk_checkpoint_ssm_a, ssm_a.size.to_i64 * sizeof(Float32))
          g_buf = Scratch.get(:rec_chunk_checkpoint_g, (n_tokens * h_v).to_i64 * sizeof(Float32))
          checkpoint_buf = Scratch.get(:rec_chunk_checkpoint_conv, ((conv_k - 1) * qkv_dim).to_i64 * sizeof(Float32))

          qkv_buf.write(qkv_mixed)
          conv_w_buf.write(ssm_conv1d)
          alpha_buf.write(alpha)
          beta_buf.write(beta)
          dt_bias_buf.write(ssm_dt_bias)
          ssm_a_buf.write(ssm_a)

          cmd = ML::Metal::CommandBuffer.new

          conv_enc = ML::Metal::ComputeEncoder.new(cmd)
          conv_enc.set_pipeline(recurrent_conv_shift_chunk_checkpoint_pipeline)
          conv_enc.set_buffer(conv_state_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          conv_enc.set_buffer(qkv_buf, 1)
          conv_enc.set_buffer(conv_w_buf, 2)
          conv_enc.set_buffer(q_buf, 3, ML::Metal::BufferAccess::Write)
          conv_enc.set_buffer(k_buf, 4, ML::Metal::BufferAccess::Write)
          conv_enc.set_buffer(v_buf, 5, ML::Metal::BufferAccess::Write)
          conv_enc.set_value(h_k.to_u32, 6)
          conv_enc.set_value(h_v.to_u32, 7)
          conv_enc.set_value(s.to_u32, 8)
          conv_enc.set_value(conv_k.to_u32, 9)
          conv_enc.set_value(n_tokens.to_u32, 10)
          conv_enc.set_buffer(checkpoint_buf, 11, ML::Metal::BufferAccess::Write)
          conv_enc.set_value(checkpoint_index.to_u32, 12)
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

          cmd.commit
          cmd.wait

          {
            q:               read_shared_f32(q_buf, n_tokens * q_dim),
            k:               read_shared_f32(k_buf, n_tokens * q_dim),
            v:               read_shared_f32(v_buf, n_tokens * v_dim),
            g:               read_shared_f32(g_buf, n_tokens * h_v),
            beta:            read_shared_f32(beta_buf, n_tokens * h_v),
            checkpoint_conv: read_shared_f32(checkpoint_buf, (conv_k - 1) * qkv_dim),
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

        def self.ffn_project_residual_top1(x : Array(Float32),
                                           residual : Array(Float32),
                                           gate_qw : QuantWeight,
                                           up_qw : QuantWeight,
                                           down_qw : QuantWeight,
                                           norm_weight : Array(Float32),
                                           head_qw : QuantWeight,
                                           eps : Float32) : NamedTuple(hidden: Array(Float32), top1: {Int32, Float32})?
          return nil unless ENV["QWEN35_MTP_FFN_HEAD_FUSE"]? == "1"
          gate_pipe = gemv_pipeline_for(gate_qw)
          up_pipe = gemv_pipeline_for(up_qw)
          down_pipe = gemv_pipeline_for(down_qw)
          return nil if gate_pipe.nil? || up_pipe.nil? || down_pipe.nil?
          return nil unless can_use_head_top1_fused?(head_qw)

          ML::Metal::Device.init!

          hidden_dim = x.size
          ffn_dim = gate_qw.out_dim
          raise "ffn/head residual size mismatch" unless residual.size == hidden_dim
          raise "ffn/head norm size mismatch" unless norm_weight.size == hidden_dim
          raise "ffn gate in_dim mismatch: expected #{hidden_dim}, got #{gate_qw.in_dim}" unless gate_qw.in_dim == hidden_dim
          raise "ffn up shape mismatch" unless up_qw.in_dim == hidden_dim && up_qw.out_dim == ffn_dim
          raise "ffn down shape mismatch" unless down_qw.in_dim == ffn_dim && down_qw.out_dim == hidden_dim
          raise "ffn/head shape mismatch" unless head_qw.in_dim == hidden_dim

          t0 = Time.instant if Profile.enabled?
          x_buf = Scratch.get(:ffn_head_x, hidden_dim.to_i64 * sizeof(Float32))
          residual_buf = Scratch.get(:ffn_head_residual, hidden_dim.to_i64 * sizeof(Float32))
          norm_w_buf = Scratch.get(:ffn_head_norm_w, hidden_dim.to_i64 * sizeof(Float32))
          gate_buf = Scratch.get(:ffn_head_gate, ffn_dim.to_i64 * sizeof(Float32))
          up_buf = Scratch.get(:ffn_head_up, ffn_dim.to_i64 * sizeof(Float32))
          comb_buf = Scratch.get(:ffn_head_comb, ffn_dim.to_i64 * sizeof(Float32))
          ffn_out_buf = Scratch.get(:ffn_head_ffn_out, hidden_dim.to_i64 * sizeof(Float32))
          after_ffn_buf = Scratch.get(:ffn_head_after_ffn, hidden_dim.to_i64 * sizeof(Float32))
          normed_buf = Scratch.get(:ffn_head_normed, hidden_dim.to_i64 * sizeof(Float32))
          tile_count = (head_qw.out_dim + HEAD_TOP1_ROWS_PER_TG - 1) // HEAD_TOP1_ROWS_PER_TG
          tile_values_buf = Scratch.get(:ffn_head_tile_values, tile_count.to_i64 * sizeof(Float32))
          tile_ids_buf = Scratch.get(:ffn_head_tile_ids, tile_count.to_i64 * sizeof(UInt32))
          top1_id_buf = Scratch.get(:ffn_head_top1_id, sizeof(UInt32).to_i64)
          top1_value_buf = Scratch.get(:ffn_head_top1_value, sizeof(Float32).to_i64)

          x_buf.write(x)
          residual_buf.write(residual)
          norm_w_buf.write(norm_weight)

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
          head_w_buf, head_w_off = if slot = mmap_slot_for(head_qw.raw)
                                     slot
                                   else
                                     {head_qw.fallback_metal_buffer, 0_i64}
                                   end

          cmd = ML::Metal::CommandBuffer.new

          proj_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(proj_enc, gate_pipe.not_nil!, x_buf, gate_buf, gate_w_buf, gate_w_off, gate_qw.in_dim, gate_qw.out_dim)
          encode_gemv(proj_enc, up_pipe.not_nil!, x_buf, up_buf, up_w_buf, up_w_off, up_qw.in_dim, up_qw.out_dim)
          proj_enc.end_encoding

          swiglu_enc = ML::Metal::ComputeEncoder.new(cmd)
          swiglu_enc.set_pipeline(ffn_swiglu_pipeline)
          swiglu_enc.set_buffer(gate_buf, 0)
          swiglu_enc.set_buffer(up_buf, 1)
          swiglu_enc.set_buffer(comb_buf, 2, ML::Metal::BufferAccess::Write)
          swiglu_enc.set_value(ffn_dim.to_u32, 3)
          swiglu_enc.dispatch_1d(ffn_dim, 256)
          swiglu_enc.end_encoding

          down_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(down_enc, down_pipe.not_nil!, comb_buf, ffn_out_buf, down_w_buf, down_w_off, down_qw.in_dim, down_qw.out_dim)
          down_enc.end_encoding

          norm_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_add_rmsnorm(norm_enc, residual_buf, ffn_out_buf, norm_w_buf, after_ffn_buf, normed_buf, hidden_dim, eps)
          norm_enc.end_encoding

          head_enc = ML::Metal::ComputeEncoder.new(cmd)
          head_enc.set_pipeline(head_qw.type.q8_0? ? mv8_top1_tiles_pipeline : mv6_top1_tiles_pipeline)
          head_enc.set_buffer(head_w_buf, 0, ML::Metal::BufferAccess::Read, offset: head_w_off)
          head_enc.set_buffer(normed_buf, 1)
          head_enc.set_buffer(tile_values_buf, 2, ML::Metal::BufferAccess::Write)
          head_enc.set_buffer(tile_ids_buf, 3, ML::Metal::BufferAccess::Write)
          head_enc.set_value(head_qw.in_dim.to_u32, 4)
          head_enc.set_value(head_qw.out_dim.to_u32, 5)
          head_enc.dispatch_threadgroups({tile_count, 1, 1}, {head_qw.type.q8_0? ? MV_Q8_NSG * 32 : 64, 1, 1})
          head_enc.end_encoding

          reduce_enc = ML::Metal::ComputeEncoder.new(cmd)
          reduce_enc.set_pipeline(top1_reduce_tiles_pipeline)
          reduce_enc.set_buffer(tile_values_buf, 0)
          reduce_enc.set_buffer(tile_ids_buf, 1)
          reduce_enc.set_buffer(top1_id_buf, 2, ML::Metal::BufferAccess::Write)
          reduce_enc.set_buffer(top1_value_buf, 3, ML::Metal::BufferAccess::Write)
          reduce_enc.set_value(tile_count.to_u32, 4)
          reduce_enc.dispatch_threadgroups({1, 1, 1}, {256, 1, 1})
          reduce_enc.end_encoding

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          hidden = read_shared_f32(after_ffn_buf, hidden_dim)
          top1_raw = read_shared_top1(top1_id_buf, top1_value_buf)
          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_gemv(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          {hidden: hidden, top1: {top1_raw[0].to_i32, top1_raw[1]}}
        end

        def self.ffn_project_residual_top2(x : Array(Float32),
                                           residual : Array(Float32),
                                           gate_qw : QuantWeight,
                                           up_qw : QuantWeight,
                                           down_qw : QuantWeight,
                                           norm_weight : Array(Float32),
          head_qw : QuantWeight,
          eps : Float32) : NamedTuple(hidden: Array(Float32), top2: Array(Float32))?
          return nil unless ENV["QWEN35_MTP_FFN_HEAD_FUSE"]? == "1"
          return nil if ENV["QWEN35_MTP_FFN_HEAD_TOP2_FUSE_OFF"]? == "1"
          gate_pipe = gemv_pipeline_for(gate_qw)
          up_pipe = gemv_pipeline_for(up_qw)
          down_pipe = gemv_pipeline_for(down_qw)
          return nil if gate_pipe.nil? || up_pipe.nil? || down_pipe.nil?
          return nil unless can_use_head_top1_fused?(head_qw)

          ML::Metal::Device.init!

          hidden_dim = x.size
          ffn_dim = gate_qw.out_dim
          raise "ffn/head residual size mismatch" unless residual.size == hidden_dim
          raise "ffn/head norm size mismatch" unless norm_weight.size == hidden_dim
          raise "ffn gate in_dim mismatch: expected #{hidden_dim}, got #{gate_qw.in_dim}" unless gate_qw.in_dim == hidden_dim
          raise "ffn up shape mismatch" unless up_qw.in_dim == hidden_dim && up_qw.out_dim == ffn_dim
          raise "ffn down shape mismatch" unless down_qw.in_dim == ffn_dim && down_qw.out_dim == hidden_dim
          raise "ffn/head shape mismatch" unless head_qw.in_dim == hidden_dim

          t0 = Time.instant if Profile.enabled?
          x_buf = Scratch.get(:ffn_head2_x, hidden_dim.to_i64 * sizeof(Float32))
          residual_buf = Scratch.get(:ffn_head2_residual, hidden_dim.to_i64 * sizeof(Float32))
          norm_w_buf = Scratch.get(:ffn_head2_norm_w, hidden_dim.to_i64 * sizeof(Float32))
          gate_buf = Scratch.get(:ffn_head2_gate, ffn_dim.to_i64 * sizeof(Float32))
          up_buf = Scratch.get(:ffn_head2_up, ffn_dim.to_i64 * sizeof(Float32))
          comb_buf = Scratch.get(:ffn_head2_comb, ffn_dim.to_i64 * sizeof(Float32))
          ffn_out_buf = Scratch.get(:ffn_head2_ffn_out, hidden_dim.to_i64 * sizeof(Float32))
          after_ffn_buf = Scratch.get(:ffn_head2_after_ffn, hidden_dim.to_i64 * sizeof(Float32))
          normed_buf = Scratch.get(:ffn_head2_normed, hidden_dim.to_i64 * sizeof(Float32))
          tile_count = (head_qw.out_dim + HEAD_TOP1_ROWS_PER_TG - 1) // HEAD_TOP1_ROWS_PER_TG
          tile_values_buf = Scratch.get(:ffn_head2_tile_values, tile_count.to_i64 * sizeof(Float32))
          tile_ids_buf = Scratch.get(:ffn_head2_tile_ids, tile_count.to_i64 * sizeof(UInt32))
          second_tile_values_buf = Scratch.get(:ffn_head2_second_tile_values, tile_count.to_i64 * sizeof(Float32))
          second_tile_ids_buf = Scratch.get(:ffn_head2_second_tile_ids, tile_count.to_i64 * sizeof(UInt32))
          top1_id_buf = Scratch.get(:ffn_head2_top1_id, sizeof(UInt32).to_i64)
          top1_value_buf = Scratch.get(:ffn_head2_top1_value, sizeof(Float32).to_i64)
          second_id_buf = Scratch.get(:ffn_head2_second_id, sizeof(UInt32).to_i64)
          second_value_buf = Scratch.get(:ffn_head2_second_value, sizeof(Float32).to_i64)

          x_buf.write(x)
          residual_buf.write(residual)
          norm_w_buf.write(norm_weight)

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
          head_w_buf, head_w_off = if slot = mmap_slot_for(head_qw.raw)
                                     slot
                                   else
                                     {head_qw.fallback_metal_buffer, 0_i64}
                                   end

          cmd = ML::Metal::CommandBuffer.new

          proj_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(proj_enc, gate_pipe.not_nil!, x_buf, gate_buf, gate_w_buf, gate_w_off, gate_qw.in_dim, gate_qw.out_dim)
          encode_gemv(proj_enc, up_pipe.not_nil!, x_buf, up_buf, up_w_buf, up_w_off, up_qw.in_dim, up_qw.out_dim)
          proj_enc.end_encoding

          swiglu_enc = ML::Metal::ComputeEncoder.new(cmd)
          swiglu_enc.set_pipeline(ffn_swiglu_pipeline)
          swiglu_enc.set_buffer(gate_buf, 0)
          swiglu_enc.set_buffer(up_buf, 1)
          swiglu_enc.set_buffer(comb_buf, 2, ML::Metal::BufferAccess::Write)
          swiglu_enc.set_value(ffn_dim.to_u32, 3)
          swiglu_enc.dispatch_1d(ffn_dim, 256)
          swiglu_enc.end_encoding

          down_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(down_enc, down_pipe.not_nil!, comb_buf, ffn_out_buf, down_w_buf, down_w_off, down_qw.in_dim, down_qw.out_dim)
          down_enc.end_encoding

          norm_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_add_rmsnorm(norm_enc, residual_buf, ffn_out_buf, norm_w_buf, after_ffn_buf, normed_buf, hidden_dim, eps)
          norm_enc.end_encoding

          head_enc = ML::Metal::ComputeEncoder.new(cmd)
          head_enc.set_pipeline(head_qw.type.q8_0? ? mv8_top2_tiles_pipeline : mv6_top2_tiles_pipeline)
          head_enc.set_buffer(head_w_buf, 0, ML::Metal::BufferAccess::Read, offset: head_w_off)
          head_enc.set_buffer(normed_buf, 1)
          head_enc.set_buffer(tile_values_buf, 2, ML::Metal::BufferAccess::Write)
          head_enc.set_buffer(tile_ids_buf, 3, ML::Metal::BufferAccess::Write)
          head_enc.set_buffer(second_tile_values_buf, 4, ML::Metal::BufferAccess::Write)
          head_enc.set_buffer(second_tile_ids_buf, 5, ML::Metal::BufferAccess::Write)
          head_enc.set_value(head_qw.in_dim.to_u32, 6)
          head_enc.set_value(head_qw.out_dim.to_u32, 7)
          head_enc.dispatch_threadgroups({tile_count, 1, 1}, {head_qw.type.q8_0? ? MV_Q8_NSG * 32 : 64, 1, 1})
          head_enc.end_encoding

          reduce_enc = ML::Metal::ComputeEncoder.new(cmd)
          reduce_enc.set_pipeline(top2_reduce_tiles_pipeline)
          reduce_enc.set_buffer(tile_values_buf, 0)
          reduce_enc.set_buffer(tile_ids_buf, 1)
          reduce_enc.set_buffer(second_tile_values_buf, 2)
          reduce_enc.set_buffer(second_tile_ids_buf, 3)
          reduce_enc.set_buffer(top1_id_buf, 4, ML::Metal::BufferAccess::Write)
          reduce_enc.set_buffer(top1_value_buf, 5, ML::Metal::BufferAccess::Write)
          reduce_enc.set_buffer(second_id_buf, 6, ML::Metal::BufferAccess::Write)
          reduce_enc.set_buffer(second_value_buf, 7, ML::Metal::BufferAccess::Write)
          reduce_enc.set_value(tile_count.to_u32, 8)
          reduce_enc.dispatch_threadgroups({1, 1, 1}, {256, 1, 1})
          reduce_enc.end_encoding

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          hidden = read_shared_f32(after_ffn_buf, hidden_dim)
          top2_raw = read_shared_top2(top1_id_buf, top1_value_buf, second_id_buf, second_value_buf)
          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_gemv(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          {hidden: hidden, top2: top2_raw}
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

        def self.rollback_delta_net_state_from_log(state_buf : ML::MetalBuffer,
                                                   log_buf : ML::MetalBuffer,
                                                   h_k : Int32,
                                                   h_v : Int32,
                                                   s : Int32) : Nil
          rollback_delta_net_states_from_logs([state_buf], [log_buf], h_k, h_v, s)
        end

        def self.rollback_delta_net_states_from_logs(state_bufs : Array(ML::MetalBuffer),
                                                     log_bufs : Array(ML::MetalBuffer),
                                                     h_k : Int32,
                                                     h_v : Int32,
                                                     s : Int32) : Nil
          return unless s == 128 && dn_chunk_rowwise_enabled?(s)
          raise ArgumentError.new("rollback state/log count mismatch") unless state_bufs.size == log_bufs.size
          return if state_bufs.empty?

          ML::Metal::Device.init!
          cmd = ML::Metal::CommandBuffer.new
          state_bufs.each_with_index do |state_buf, i|
            enc = ML::Metal::ComputeEncoder.new(cmd)
            enc.set_pipeline(dn128_rollback_rowwise_pipeline)
            enc.set_buffer(state_buf, 0, ML::Metal::BufferAccess::ReadWrite)
            enc.set_buffer(log_bufs[i], 1)
            enc.set_value(h_k.to_u32, 2)
            enc.set_value(h_v.to_u32, 3)
            enc.set_value(s.to_u32, 4)
            enc.dispatch_threadgroups({(s + 3) // 4, h_v, 1}, {32, 4, 1})
            enc.end_encoding
          end
          cmd.commit
          cmd.wait
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
                                                    profile_label : String = "rec_chunk_many",
                                                    checkpoint_index : Int32? = nil,
                                                    checkpoint_conv_state_bufs : Array(ML::MetalBuffer)? = nil,
                                                    checkpoint_ssm_state_bufs : Array(ML::MetalBuffer)? = nil,
                                                    checkpoint_rollback_log : Bool = false,
                                                    input_buf : ML::MetalBuffer? = nil,
                                                    output_buf : ML::MetalBuffer? = nil,
                                                    read_output : Bool = true,
                                                    append_command_buffer : ML::Metal::CommandBuffer? = nil) : Array(Float32)?
          return nil unless n_tokens > 0
          return nil if layers.empty?
          return nil if append_command_buffer && read_output
          checkpoint_requested = !checkpoint_index.nil?
          if checkpoint_requested
            cp = checkpoint_index.not_nil!
            return nil unless cp >= 0 && cp < n_tokens
            return nil if checkpoint_rollback_log && cp + 1 >= n_tokens
            return nil unless conv_k > 1 && s == 128 && dn_chunk_rowwise_enabled?(s)
            conv_chk = checkpoint_conv_state_bufs
            ssm_chk = checkpoint_ssm_state_bufs
            return nil if conv_chk.nil? || conv_chk.size != layers.size
            return nil if ssm_chk.nil? || ssm_chk.size != layers.size
          end

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
          hidden_elems = n_tokens * hidden_dim
          hidden_bytes = hidden_elems.to_i64 * sizeof(Float32)
          if ib = input_buf
            raise "recurrent_layer_chunk_many input buffer too small" if ib.size < hidden_bytes
          else
            raise "recurrent_layer_chunk_many input size mismatch" unless inp.size == hidden_elems
          end
          if ob = output_buf
            raise "recurrent_layer_chunk_many output buffer too small" if ob.size < hidden_bytes
          end
          raise "recurrent_layer_chunk_many state size mismatch" unless conv_state_bufs.size == layers.size && ssm_state_bufs.size == layers.size

          src_buf = input_buf || Scratch.get(:rec_chunk_many_hidden_a, hidden_bytes)
          dst_buf = Scratch.get(:rec_chunk_many_hidden_b, hidden_bytes)
          cur_buf = Scratch.get(:rec_chunk_many_cur, hidden_bytes)
          cur_h16_buf = Scratch.get(:rec_chunk_many_cur_h16, hidden_elems.to_i64 * 2_i64)
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
          attn_mid_h16_buf = Scratch.get(:rec_chunk_many_mid_h16, (n_tokens * d_inner).to_i64 * 2_i64)
          attn_out_buf = Scratch.get(:rec_chunk_many_attn_out, hidden_bytes)
          residual_buf = Scratch.get(:rec_chunk_many_residual, hidden_bytes)
          normed_buf = Scratch.get(:rec_chunk_many_normed, hidden_bytes)
          normed_h16_buf = Scratch.get(:rec_chunk_many_normed_h16, hidden_elems.to_i64 * 2_i64)
          ffn_gate_buf = Scratch.get(:rec_chunk_many_ffn_gate, (n_tokens * ffn_dim).to_i64 * sizeof(Float32))
          ffn_up_buf = Scratch.get(:rec_chunk_many_ffn_up, (n_tokens * ffn_dim).to_i64 * sizeof(Float32))
          ffn_comb_buf = Scratch.get(:rec_chunk_many_ffn_comb, (n_tokens * ffn_dim).to_i64 * sizeof(Float32))
          ffn_comb_h16_buf = Scratch.get(:rec_chunk_many_ffn_comb_h16, (n_tokens * ffn_dim).to_i64 * 2_i64)
          ffn_out_buf = Scratch.get(:rec_chunk_many_ffn_out, hidden_bytes)

          unless input_buf
            src_buf.write(inp)
            Profile.bump_group_transfer("#{profile_label}.boundary", hidden_bytes, 0_i64)
          end

          t0 = Time.instant if Profile.enabled?
          cmd = append_command_buffer || ML::Metal::CommandBuffer.new
          appended = !append_command_buffer.nil?
          phase_profile = !appended && prefill_phase_profile_enabled? && Profile.enabled?
          full_detail_profile = prefill_full_detail_profile_enabled? && phase_profile
          phase_t0 = Time.instant

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
            norm_h16_proj = prefill_rmsnorm_h16_proj_enabled? && n_tokens > GEMM_BATCH_THRESHOLD
            if norm_h16_proj
              encode_rmsnorm_rows_f32_h16(norm_enc, src_buf, norm_w_buf, cur_buf, cur_h16_buf, hidden_dim, n_tokens, eps)
            else
              encode_rmsnorm_rows(norm_enc, src_buf, norm_w_buf, cur_buf, hidden_dim, n_tokens, eps)
            end
            norm_enc.end_encoding

            Profile.trace("prefill.rec.proj") do
              proj_enc = ML::Metal::ComputeEncoder.new(cmd)
              # The checkpoint conv kernel currently consumes f32 qkv rows.
              # Keep checkpointed verifier experiments exact by using the f32
              # projection route even when the normal prefill path would use
              # the h16 qkv fast path.
              qkv_h16 = !checkpoint_requested && q5_qkv_h16_conv_enabled? && q56_batch_gemm_enabled? && lw.attn_qkv_qw.type.q5_k? && n_tokens > GEMM_BATCH_THRESHOLD
              shared_h16 = rec_proj_shared_h16_enabled? && qkv_h16 && q4_h16_gemm_enabled? &&
                           lw.attn_gate_qw.type.q4_k? && n_tokens > GEMM_BATCH_THRESHOLD
              if shared_h16 && norm_h16_proj
                Profile.bump_matmul_shape("q5_h16_gemm #{lw.attn_qkv_qw.type.name} #{lw.attn_qkv_qw.in_dim}x#{lw.attn_qkv_qw.out_dim} b#{n_tokens}", lw.attn_qkv_qw.raw.size.to_i64)
                encode_q56k_gemm_h16_from_h16(proj_enc, mm5_pipeline, cur_h16_buf, qkv_h16_buf, qkv_w_buf, qkv_w_off, lw.attn_qkv_qw.in_dim, lw.attn_qkv_qw.out_dim, n_tokens)
                Profile.bump_matmul_shape("q4_h16_gemm #{lw.attn_gate_qw.type.name} #{lw.attn_gate_qw.in_dim}x#{lw.attn_gate_qw.out_dim} b#{n_tokens}", lw.attn_gate_qw.raw.size.to_i64)
                encode_q4k_gemm_h16_from_h16(proj_enc, cur_h16_buf, z_buf, gate_w_buf, gate_w_off, lw.attn_gate_qw.in_dim, lw.attn_gate_qw.out_dim, n_tokens)
              elsif shared_h16
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
              elsif qkv_h16 && norm_h16_proj
                Profile.bump_matmul_shape("q5_h16_gemm #{lw.attn_qkv_qw.type.name} #{lw.attn_qkv_qw.in_dim}x#{lw.attn_qkv_qw.out_dim} b#{n_tokens}", lw.attn_qkv_qw.raw.size.to_i64)
                encode_q56k_gemm_h16_from_h16(proj_enc, mm5_pipeline, cur_h16_buf, qkv_h16_buf, qkv_w_buf, qkv_w_off, lw.attn_qkv_qw.in_dim, lw.attn_qkv_qw.out_dim, n_tokens)
                encode_matmul(proj_enc, gemv_pipeline_for(lw.attn_gate_qw).not_nil!, lw.attn_gate_qw, cur_buf, z_buf, gate_w_buf, gate_w_off, lw.attn_gate_qw.in_dim, lw.attn_gate_qw.out_dim, n_tokens)
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
            if full_detail_profile
              checked = prefill_phase_checkpoint(cmd, "#{profile_label}.rec#{local_i}.proj", phase_t0)
              cmd = checked[0]
              phase_t0 = checked[1]
            end

            conv_enc = ML::Metal::ComputeEncoder.new(cmd)
            qkv_h16 = !checkpoint_requested && q5_qkv_h16_conv_enabled? && q56_batch_gemm_enabled? && lw.attn_qkv_qw.type.q5_k? && n_tokens > GEMM_BATCH_THRESHOLD
            conv_enc.set_pipeline(checkpoint_requested ? recurrent_conv_shift_chunk_checkpoint_pipeline : (qkv_h16 ? recurrent_conv_shift_chunk_h16_pipeline : recurrent_conv_shift_chunk_pipeline))
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
            if checkpoint_requested
              conv_enc.set_buffer(checkpoint_conv_state_bufs.not_nil![local_i], 11, ML::Metal::BufferAccess::Write)
              conv_enc.set_value(checkpoint_index.not_nil!.to_u32, 12)
            end
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
            if full_detail_profile
              checked = prefill_phase_checkpoint(cmd, "#{profile_label}.rec#{local_i}.prep", phase_t0)
              cmd = checked[0]
              phase_t0 = checked[1]
            end

            dn_enc = ML::Metal::ComputeEncoder.new(cmd)
            use_dn_rowwise = dn_chunk_rowwise_enabled?(s)
            dn_enc.set_pipeline(checkpoint_requested ? (checkpoint_rollback_log ? dn128_chunk_rowwise_rollback_log_pipeline : dn128_chunk_rowwise_checkpoint_pipeline) : (use_dn_rowwise ? dn128_chunk_rowwise_pipeline : dn128_chunk_fused_pipeline))
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
            if checkpoint_requested
              dn_enc.set_buffer(checkpoint_ssm_state_bufs.not_nil![local_i], 12, ML::Metal::BufferAccess::Write)
              log_index = checkpoint_rollback_log ? checkpoint_index.not_nil! + 1 : checkpoint_index.not_nil!
              dn_enc.set_value(log_index.to_u32, 13)
            end
            if use_dn_rowwise
              dn_enc.dispatch_threadgroups({(s + 3) // 4, h_v, 1}, {32, 4, 1})
            else
              dn_enc.dispatch_threadgroups({h_v, 1, 1}, {128, 1, 1})
            end
            dn_enc.end_encoding
            if full_detail_profile
              checked = prefill_phase_checkpoint(cmd, "#{profile_label}.rec#{local_i}.dn", phase_t0)
              cmd = checked[0]
              phase_t0 = checked[1]
            end

            post_enc = ML::Metal::ComputeEncoder.new(cmd)
            o_proj_h16 = prefill_dn_post_h16_oproj_enabled? && h16_batch_gemm_candidate?(lw.ssm_out_qw, n_tokens)
            post_enc.set_pipeline(o_proj_h16 ? dn_post_chunk_h16_pipeline : dn_post_chunk_pipeline)
            post_enc.set_buffer(attn_mid_buf, 0, ML::Metal::BufferAccess::ReadWrite)
            post_enc.set_buffer(z_buf, 1)
            post_enc.set_buffer(ssm_norm_buf, 2)
            if o_proj_h16
              post_enc.set_buffer(attn_mid_h16_buf, 3, ML::Metal::BufferAccess::Write)
              post_enc.set_value(h_v.to_u32, 4)
              post_enc.set_value(s.to_u32, 5)
              post_enc.set_value(eps, 6)
              post_enc.set_value(n_tokens.to_u32, 7)
            else
              post_enc.set_value(h_v.to_u32, 3)
              post_enc.set_value(s.to_u32, 4)
              post_enc.set_value(eps, 5)
              post_enc.set_value(n_tokens.to_u32, 6)
            end
            post_enc.dispatch_threadgroups({h_v, n_tokens, 1}, {32, 1, 1})
            post_enc.end_encoding

            Profile.trace("prefill.rec.o_proj") do
              out_enc = ML::Metal::ComputeEncoder.new(cmd)
              if o_proj_h16
                raise "unsupported h16 recurrent o_proj route" unless encode_matmul_from_h16(out_enc, lw.ssm_out_qw, attn_mid_h16_buf, attn_out_buf, out_w_buf, out_w_off, lw.ssm_out_qw.in_dim, lw.ssm_out_qw.out_dim, n_tokens)
              else
                encode_matmul(out_enc, gemv_pipeline_for(lw.ssm_out_qw).not_nil!, lw.ssm_out_qw, attn_mid_buf, attn_out_buf, out_w_buf, out_w_off, lw.ssm_out_qw.in_dim, lw.ssm_out_qw.out_dim, n_tokens)
              end
              out_enc.end_encoding
            end
            if full_detail_profile
              checked = prefill_phase_checkpoint(cmd, "#{profile_label}.rec#{local_i}.post_oproj", phase_t0)
              cmd = checked[0]
              phase_t0 = checked[1]
            end

            ffn_pair_h16 = prefill_addnorm_h16_ffn_enabled? && q4_pair_h16_gemm_candidate?(lw.ffn_gate_qw, lw.ffn_up_qw, n_tokens)
            addnorm_enc = ML::Metal::ComputeEncoder.new(cmd)
            if ffn_pair_h16
              encode_add_rmsnorm_rows_h16(addnorm_enc, src_buf, attn_out_buf, post_w_buf, residual_buf, normed_h16_buf, hidden_dim, n_tokens, eps)
            else
              encode_add_rmsnorm_rows(addnorm_enc, src_buf, attn_out_buf, post_w_buf, residual_buf, normed_buf, hidden_dim, n_tokens, eps)
            end
            addnorm_enc.end_encoding

            ffn_act_buf = swiglu_inplace_enabled? ? ffn_up_buf : ffn_comb_buf
            ffn_down_h16 = prefill_swiglu_h16_down_candidate?(lw.ffn_down_qw, n_tokens) ||
              q4_b64_up_swiglu_h16_down_candidate?(lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw, n_tokens)
            up_swiglu_fused = false

            Profile.trace("prefill.rec.ffn_upgate") do
              ffn_proj_enc = ML::Metal::ComputeEncoder.new(cmd)
              pair_q4 = q4_pair_h16_gemm_candidate?(lw.ffn_gate_qw, lw.ffn_up_qw, n_tokens)
              if pair_q4
                Profile.bump_matmul_shape("q4_h16_gemm #{lw.ffn_gate_qw.type.name} #{lw.ffn_gate_qw.in_dim}x#{lw.ffn_gate_qw.out_dim} b#{n_tokens}", lw.ffn_gate_qw.raw.size.to_i64)
                Profile.bump_matmul_shape("q4_h16_gemm #{lw.ffn_up_qw.type.name} #{lw.ffn_up_qw.in_dim}x#{lw.ffn_up_qw.out_dim} b#{n_tokens}", lw.ffn_up_qw.raw.size.to_i64)
                if q4_b64_up_swiglu_h16_candidate?(lw.ffn_gate_qw, lw.ffn_up_qw, n_tokens, ffn_down_h16)
                  up_swiglu_fused = true
                  if ffn_pair_h16
                    encode_q4k_gemm_h16_from_h16(ffn_proj_enc, normed_h16_buf, ffn_gate_buf, ffn_gate_w_buf, ffn_gate_w_off, lw.ffn_gate_qw.in_dim, lw.ffn_gate_qw.out_dim, n_tokens)
                    encode_q4k_gemm_h16_b64_swiglu_h16_from_h16(ffn_proj_enc, normed_h16_buf, ffn_gate_buf, ffn_comb_h16_buf, ffn_up_w_buf, ffn_up_w_off, lw.ffn_up_qw.in_dim, lw.ffn_up_qw.out_dim, n_tokens)
                  else
                    encode_q4k_gemm_h16_pair_b64_swiglu_h16(ffn_proj_enc, normed_buf, ffn_gate_buf, ffn_comb_h16_buf, ffn_gate_w_buf, ffn_gate_w_off, ffn_up_w_buf, ffn_up_w_off, lw.ffn_gate_qw.in_dim, lw.ffn_gate_qw.out_dim, n_tokens)
                  end
                elsif q4_b64_up_swiglu_candidate?(lw.ffn_gate_qw, lw.ffn_up_qw, n_tokens, ffn_down_h16)
                  up_swiglu_fused = true
                  if ffn_pair_h16
                    encode_q4k_gemm_h16_from_h16(ffn_proj_enc, normed_h16_buf, ffn_gate_buf, ffn_gate_w_buf, ffn_gate_w_off, lw.ffn_gate_qw.in_dim, lw.ffn_gate_qw.out_dim, n_tokens)
                    encode_q4k_gemm_h16_b64_swiglu_from_h16(ffn_proj_enc, normed_h16_buf, ffn_gate_buf, ffn_act_buf, ffn_up_w_buf, ffn_up_w_off, lw.ffn_up_qw.in_dim, lw.ffn_up_qw.out_dim, n_tokens)
                  else
                    encode_q4k_gemm_h16_pair_b64_swiglu(ffn_proj_enc, normed_buf, ffn_gate_buf, ffn_act_buf, ffn_gate_w_buf, ffn_gate_w_off, ffn_up_w_buf, ffn_up_w_off, lw.ffn_gate_qw.in_dim, lw.ffn_gate_qw.out_dim, n_tokens)
                  end
                elsif ffn_pair_h16
                  encode_q4k_gemm_h16_from_h16(ffn_proj_enc, normed_h16_buf, ffn_gate_buf, ffn_gate_w_buf, ffn_gate_w_off, lw.ffn_gate_qw.in_dim, lw.ffn_gate_qw.out_dim, n_tokens)
                  encode_q4k_gemm_h16_from_h16(ffn_proj_enc, normed_h16_buf, ffn_up_buf, ffn_up_w_buf, ffn_up_w_off, lw.ffn_up_qw.in_dim, lw.ffn_up_qw.out_dim, n_tokens)
                else
                  encode_q4k_gemm_h16_pair(ffn_proj_enc, normed_buf, ffn_gate_buf, ffn_up_buf, ffn_gate_w_buf, ffn_gate_w_off, ffn_up_w_buf, ffn_up_w_off, lw.ffn_gate_qw.in_dim, lw.ffn_gate_qw.out_dim, n_tokens)
                end
              else
                encode_matmul(ffn_proj_enc, gemv_pipeline_for(lw.ffn_gate_qw).not_nil!, lw.ffn_gate_qw, normed_buf, ffn_gate_buf, ffn_gate_w_buf, ffn_gate_w_off, lw.ffn_gate_qw.in_dim, lw.ffn_gate_qw.out_dim, n_tokens)
                encode_matmul(ffn_proj_enc, gemv_pipeline_for(lw.ffn_up_qw).not_nil!, lw.ffn_up_qw, normed_buf, ffn_up_buf, ffn_up_w_buf, ffn_up_w_off, lw.ffn_up_qw.in_dim, lw.ffn_up_qw.out_dim, n_tokens)
              end
              ffn_proj_enc.end_encoding
            end
            if full_detail_profile
              checked = prefill_phase_checkpoint(cmd, "#{profile_label}.rec#{local_i}.ffn_upgate", phase_t0)
              cmd = checked[0]
              phase_t0 = checked[1]
            end

            unless up_swiglu_fused
              swiglu_enc = ML::Metal::ComputeEncoder.new(cmd)
              swiglu_enc.set_pipeline(ffn_down_h16 ? ffn_swiglu_h16_pipeline : ffn_swiglu_pipeline)
              swiglu_enc.set_buffer(ffn_gate_buf, 0)
              swiglu_enc.set_buffer(ffn_up_buf, 1)
              swiglu_enc.set_buffer(ffn_down_h16 ? ffn_comb_h16_buf : ffn_act_buf, 2, ML::Metal::BufferAccess::Write)
              swiglu_enc.set_value((n_tokens * ffn_dim).to_u32, 3)
              swiglu_enc.dispatch_1d(n_tokens * ffn_dim, 256)
              swiglu_enc.end_encoding
            end

            fused_down_add = false
            if prefill_ffn_down_add_fused_enabled?
              Profile.trace("prefill.rec.ffn_down_add") do
                ffn_down_enc = ML::Metal::ComputeEncoder.new(cmd)
                fused_down_add = if ffn_down_h16
                                   encode_matmul_add_from_h16(ffn_down_enc, lw.ffn_down_qw, ffn_comb_h16_buf, residual_buf, dst_buf, ffn_down_w_buf, ffn_down_w_off, lw.ffn_down_qw.in_dim, lw.ffn_down_qw.out_dim, n_tokens)
                                 else
                                   encode_matmul_add(ffn_down_enc, gemv_pipeline_for(lw.ffn_down_qw).not_nil!, lw.ffn_down_qw, ffn_act_buf, residual_buf, dst_buf, ffn_down_w_buf, ffn_down_w_off, lw.ffn_down_qw.in_dim, lw.ffn_down_qw.out_dim, n_tokens)
                                 end
                ffn_down_enc.end_encoding
              end
            end

            unless fused_down_add
              Profile.trace("prefill.rec.ffn_down") do
                ffn_down_enc = ML::Metal::ComputeEncoder.new(cmd)
                if ffn_down_h16
                  raise "unsupported h16 FFN-down route" unless encode_matmul_from_h16(ffn_down_enc, lw.ffn_down_qw, ffn_comb_h16_buf, ffn_out_buf, ffn_down_w_buf, ffn_down_w_off, lw.ffn_down_qw.in_dim, lw.ffn_down_qw.out_dim, n_tokens)
                else
                  encode_matmul(ffn_down_enc, gemv_pipeline_for(lw.ffn_down_qw).not_nil!, lw.ffn_down_qw, ffn_act_buf, ffn_out_buf, ffn_down_w_buf, ffn_down_w_off, lw.ffn_down_qw.in_dim, lw.ffn_down_qw.out_dim, n_tokens)
                end
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
            if full_detail_profile
              checkpoint_name = fused_down_add ? "ffn_down_add" : "ffn_down"
              checked = prefill_phase_checkpoint(cmd, "#{profile_label}.rec#{local_i}.#{checkpoint_name}", phase_t0)
              cmd = checked[0]
              phase_t0 = checked[1]
            end

            src_buf, dst_buf = dst_buf, src_buf
          end

          if full_detail_profile
            if ob = output_buf
              ob.copy_from(src_buf, hidden_bytes)
            end
            t_read0 = Time.instant
            result = read_output ? read_shared_f32(src_buf, hidden_elems) : [] of Float32
            t_read = Time.instant
            Profile.bump_group_transfer("#{profile_label}.boundary", 0_i64, hidden_bytes) if read_output
            Profile.bump_group("#{profile_label}.read", 0_i64, 0_i64, (t_read - t_read0).total_nanoseconds.to_i64)
            return result
          end

          if ob = output_buf
            blit = ML::Metal::BlitEncoder.new(cmd)
            blit.copy_buffer(src_buf, 0, ob, 0, hidden_bytes.to_i32)
            blit.end_encoding
          end
          return [] of Float32 if appended
          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_output ? read_shared_f32(src_buf, hidden_elems) : [] of Float32
          if Profile.enabled?
            t_read = Time.instant
            encode_ns = (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64
            wait_ns = (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64
            read_ns = read_output ? (t_read - t_wait.not_nil!).total_nanoseconds.to_i64 : 0_i64
            Profile.bump_group_transfer("#{profile_label}.boundary", 0_i64, hidden_bytes) if read_output
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
                                                    scale : Float32,
                                                    input_buf : ML::MetalBuffer? = nil) : Array(Float32)?
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
          hidden_bytes = (n_tokens * hidden_dim).to_i64 * sizeof(Float32)
          if ib = input_buf
            raise "final full-attn input buffer too small" if ib.size < hidden_bytes
          else
            raise "final full-attn input size mismatch" unless inp.size == n_tokens * hidden_dim
          end

          final_pos = start_pos + n_tokens - 1
          last_offset = (n_tokens - 1) * hidden_dim
          last_byte_offset = last_offset.to_i64 * sizeof(Float32)
          last_x = input_buf ? nil : inp[last_offset, hidden_dim]

          inp_buf = input_buf || Scratch.get(:full_last_inp, hidden_bytes)
          last_inp_buf = Scratch.get(:full_last_last_inp, hidden_dim.to_i64 * sizeof(Float32))
          norm_w_buf = Scratch.get(:full_last_norm_w, attn_norm.size.to_i64 * sizeof(Float32))
          cur_buf = Scratch.get(:full_last_cur, hidden_bytes)
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

          inp_buf.write(inp) unless input_buf
          last_inp_buf.write(last_x.not_nil!) unless input_buf
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

          if input_buf
            blit = ML::Metal::BlitEncoder.new(cmd)
            blit.copy_buffer(inp_buf, last_byte_offset.to_i32, last_inp_buf, 0, (hidden_dim * sizeof(Float32)).to_i32)
            blit.end_encoding
          end

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
        def self.full_attn_layer_chunk_project_last_top1(inp : Array(Float32),
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
                                                    output_norm : Array(Float32),
                                                    output_qw : QuantWeight,
                                                    input_buf : ML::MetalBuffer? = nil) : {Int32, Float32}?
          q_pipe = gemv_pipeline_for(q_qw)
          k_pipe = gemv_pipeline_for(k_qw)
          v_pipe = gemv_pipeline_for(v_qw)
          out_pipe = gemv_pipeline_for(out_qw)
          ffn_gate_pipe = gemv_pipeline_for(ffn_gate_qw)
          ffn_up_pipe = gemv_pipeline_for(ffn_up_qw)
          ffn_down_pipe = gemv_pipeline_for(ffn_down_qw)
          return nil if q_pipe.nil? || k_pipe.nil? || v_pipe.nil? || out_pipe.nil? ||
                        ffn_gate_pipe.nil? || ffn_up_pipe.nil? || ffn_down_pipe.nil?
          return nil unless can_use_head_top1_fused?(output_qw)
          return nil unless n_tokens > 0

          ML::Metal::Device.init!

          hidden_dim = q_qw.in_dim
          q_dim = n_head * head_dim
          kv_dim = n_head_kv * head_dim
          ffn_dim = ffn_gate_qw.out_dim
          tile_count = (output_qw.out_dim + HEAD_TOP1_ROWS_PER_TG - 1) // HEAD_TOP1_ROWS_PER_TG
          raise "final full-attn top1 output norm size mismatch" unless output_norm.size == hidden_dim
          raise "final full-attn top1 head shape mismatch" unless output_qw.in_dim == hidden_dim
          hidden_bytes = (n_tokens * hidden_dim).to_i64 * sizeof(Float32)
          if ib = input_buf
            raise "final full-attn input buffer too small" if ib.size < hidden_bytes
          else
            raise "final full-attn input size mismatch" unless inp.size == n_tokens * hidden_dim
          end

          final_pos = start_pos + n_tokens - 1
          last_offset = (n_tokens - 1) * hidden_dim
          last_byte_offset = last_offset.to_i64 * sizeof(Float32)
          last_x = input_buf ? nil : inp[last_offset, hidden_dim]

          inp_buf = input_buf || Scratch.get(:full_last_inp, hidden_bytes)
          last_inp_buf = Scratch.get(:full_last_last_inp, hidden_dim.to_i64 * sizeof(Float32))
          norm_w_buf = Scratch.get(:full_last_norm_w, attn_norm.size.to_i64 * sizeof(Float32))
          cur_buf = Scratch.get(:full_last_cur, hidden_bytes)
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
          head_norm_w_buf = Scratch.get(:full_last_head_norm_w, output_norm.size.to_i64 * sizeof(Float32))
          head_normed_buf = Scratch.get(:full_last_head_normed, hidden_dim.to_i64 * sizeof(Float32))
          tile_values_buf = Scratch.get(:full_last_head_tile_values, tile_count.to_i64 * sizeof(Float32))
          tile_ids_buf = Scratch.get(:full_last_head_tile_ids, tile_count.to_i64 * sizeof(UInt32))
          top1_id_buf = Scratch.get(:full_last_head_top1_id, sizeof(UInt32).to_i64)
          top1_value_buf = Scratch.get(:full_last_head_top1_value, sizeof(Float32).to_i64)

          inp_buf.write(inp) unless input_buf
          last_inp_buf.write(last_x.not_nil!) unless input_buf
          norm_w_buf.write(attn_norm)
          qnorm_buf.write(q_norm)
          knorm_buf.write(k_norm)
          post_norm_buf.write(post_attention_norm)
          head_norm_w_buf.write(output_norm)

          q_w_buf, q_w_off = weight_slot(q_qw)
          k_w_buf, k_w_off = weight_slot(k_qw)
          v_w_buf, v_w_off = weight_slot(v_qw)
          out_w_buf, out_w_off = weight_slot(out_qw)
          ffn_gate_w_buf, ffn_gate_w_off = weight_slot(ffn_gate_qw)
          ffn_up_w_buf, ffn_up_w_off = weight_slot(ffn_up_qw)
          ffn_down_w_buf, ffn_down_w_off = weight_slot(ffn_down_qw)
          output_w_buf, output_w_off = weight_slot(output_qw)

          t0 = Time.instant if Profile.enabled?
          cmd = ML::Metal::CommandBuffer.new

          if input_buf
            blit = ML::Metal::BlitEncoder.new(cmd)
            blit.copy_buffer(inp_buf, last_byte_offset.to_i32, last_inp_buf, 0, (hidden_dim * sizeof(Float32)).to_i32)
            blit.end_encoding
          end

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

          head_norm_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_rmsnorm_vec(head_norm_enc, out_buf, head_norm_w_buf, head_normed_buf, hidden_dim, eps)
          head_norm_enc.end_encoding

          head_top1_enc = ML::Metal::ComputeEncoder.new(cmd)
          head_top1_enc.set_pipeline(output_qw.type.q8_0? ? mv8_top1_tiles_pipeline : mv6_top1_tiles_pipeline)
          head_top1_enc.set_buffer(output_w_buf, 0, ML::Metal::BufferAccess::Read, offset: output_w_off)
          head_top1_enc.set_buffer(head_normed_buf, 1)
          head_top1_enc.set_buffer(tile_values_buf, 2, ML::Metal::BufferAccess::Write)
          head_top1_enc.set_buffer(tile_ids_buf, 3, ML::Metal::BufferAccess::Write)
          head_top1_enc.set_value(output_qw.in_dim.to_u32, 4)
          head_top1_enc.set_value(output_qw.out_dim.to_u32, 5)
          head_top1_enc.dispatch_threadgroups({tile_count, 1, 1}, {output_qw.type.q8_0? ? MV_Q8_NSG * 32 : 64, 1, 1})
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
            Profile.bump_attn(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          {result[0].to_i32, result[1]}
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
                                               scale : Float32,
                                               read_output : Bool = true,
                                               output_buf : ML::MetalBuffer? = nil) : Array(Float32)?
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
          use_attn_sg4 = prefill_attn_rows_sg4_enabled? && n_tokens >= 4
          use_direct_gate = !prefill_attn_rows_sg4_pregate_enabled? && prefill_attn_rows_sg4_direct_gate_enabled?(n_tokens)
          attn_sg4_pipeline = use_direct_gate ? attn_rows_sg4_pipeline : attn_rows_sg4_pregate_pipeline
          attn_enc.set_pipeline(use_attn_sg4 ? attn_sg4_pipeline : attn_rows_pipeline)
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
          if use_attn_sg4
            attn_enc.dispatch_threadgroups({n_head, (n_tokens + 3) // 4, 1}, {128, 1, 1})
          else
            attn_enc.dispatch_threadgroups({n_head, n_tokens, 1}, {32, 1, 1})
          end
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

          if ob = output_buf
            blit = ML::Metal::BlitEncoder.new(cmd)
            blit.copy_buffer(out_buf, 0, ob, 0, ((n_tokens * hidden_dim).to_i64 * sizeof(Float32)).to_i32)
            blit.end_encoding
          end

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_output ? read_shared_f32(out_buf, n_tokens * hidden_dim) : [] of Float32
          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_attn(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              read_output ? (t_read - t_wait.not_nil!).total_nanoseconds.to_i64 : 0_i64,
            )
          end
          result
        end

        def self.rmsnorm_project_top1_rows_buffer(x_buf : ML::MetalBuffer,
                                                  rows : Int32,
                                                  norm_weight : Array(Float32),
                                                  out_qw : QuantWeight,
                                                  eps : Float32) : Array({Int32, Float32})?
          return nil unless head_top1_fused_enabled?
          return nil unless out_qw.type.q6_k?
          return nil unless out_qw.in_dim % QK_K == 0
          return nil unless rows > 0
          hidden_dim = out_qw.in_dim
          hidden_bytes = (rows * hidden_dim).to_i64 * sizeof(Float32)
          return nil if x_buf.size < hidden_bytes

          ML::Metal::Device.init!

          tile_count = (out_qw.out_dim + HEAD_TOP1_ROWS_PER_TG - 1) // HEAD_TOP1_ROWS_PER_TG
          norm_w_buf = Scratch.get(:head_top1_rows_buf_norm_w, norm_weight.size.to_i64 * sizeof(Float32))
          normed_buf = Scratch.get(:head_top1_rows_buf_normed, hidden_bytes)
          tile_values_buf = Scratch.get(:head_top1_rows_buf_tile_values, (rows * tile_count).to_i64 * sizeof(Float32))
          tile_ids_buf = Scratch.get(:head_top1_rows_buf_tile_ids, (rows * tile_count).to_i64 * sizeof(UInt32))
          top1_id_buf = Scratch.get(:head_top1_rows_buf_id, rows.to_i64 * sizeof(UInt32))
          top1_value_buf = Scratch.get(:head_top1_rows_buf_value, rows.to_i64 * sizeof(Float32))
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

        # Final full-attention body-only prefill specialization.
        #
        # When no logits or downstream hidden rows are requested, the last
        # decoder layer only has to populate its K/V cache for future decode.
        # Q/attention/FFN output is semantically dead and can be skipped.
        def self.full_attn_layer_chunk_kv_cache_only(inp : Array(Float32),
                                                     k_qw : QuantWeight,
                                                     v_qw : QuantWeight,
                                                     attn_norm : Array(Float32),
                                                     k_norm : Array(Float32),
                                                     k_cache_buf : ML::MetalBuffer,
                                                     v_cache_buf : ML::MetalBuffer,
                                                     start_pos : Int32,
                                                     n_tokens : Int32,
                                                     n_head_kv : Int32,
                                                     head_dim : Int32,
                                                     rope_dim_count : Int32,
                                                     rope_freq_base : Float32,
                                                     eps : Float32,
                                                     input_buf : ML::MetalBuffer? = nil,
                                                     append_command_buffer : ML::Metal::CommandBuffer? = nil) : Bool
          k_pipe = gemv_pipeline_for(k_qw)
          v_pipe = gemv_pipeline_for(v_qw)
          return false if k_pipe.nil? || v_pipe.nil?
          return false unless n_tokens > 0

          ML::Metal::Device.init!

          hidden_dim = k_qw.in_dim
          kv_dim = n_head_kv * head_dim
          hidden_elems = n_tokens * hidden_dim
          hidden_bytes = hidden_elems.to_i64 * sizeof(Float32)
          if ib = input_buf
            raise "full-attn kv-cache input buffer too small" if ib.size < hidden_bytes
          else
            raise "full-attn kv-cache input size mismatch" unless inp.size == hidden_elems
          end

          inp_buf = input_buf || Scratch.get(:full_kv_only_inp, hidden_bytes)
          norm_w_buf = Scratch.get(:full_kv_only_norm_w, attn_norm.size.to_i64 * sizeof(Float32))
          cur_buf = Scratch.get(:full_kv_only_cur, hidden_bytes)
          cur_h16_buf = Scratch.get(:full_kv_only_cur_h16, hidden_elems.to_i64 * 2_i64)
          k_buf = Scratch.get(:full_kv_only_k, (n_tokens * kv_dim).to_i64 * sizeof(Float32))
          v_buf = Scratch.get(:full_kv_only_v, (n_tokens * kv_dim).to_i64 * sizeof(Float32))
          knorm_buf = Scratch.get(:full_kv_only_knorm, k_norm.size.to_i64 * sizeof(Float32))

          unless input_buf
            inp_buf.write(inp)
            Profile.bump_group_transfer("full_kv_only.boundary", hidden_bytes, 0_i64) if Profile.enabled?
          end
          ConstCache.write_once("full_kv_only_norm_w_#{attn_norm.object_id}", norm_w_buf, attn_norm)
          ConstCache.write_once("full_kv_only_knorm_#{k_norm.object_id}", knorm_buf, k_norm)

          k_w_buf, k_w_off = weight_slot(k_qw)
          v_w_buf, v_w_off = weight_slot(v_qw)

          t0 = Time.instant if Profile.enabled?
          cmd = append_command_buffer || ML::Metal::CommandBuffer.new
          appended = !append_command_buffer.nil?

          norm_enc = ML::Metal::ComputeEncoder.new(cmd)
          norm_h16_proj = prefill_rmsnorm_h16_proj_enabled? && n_tokens > GEMM_BATCH_THRESHOLD
          if norm_h16_proj
            encode_rmsnorm_rows_f32_h16(norm_enc, inp_buf, norm_w_buf, cur_buf, cur_h16_buf, hidden_dim, n_tokens, eps)
          else
            encode_rmsnorm_rows(norm_enc, inp_buf, norm_w_buf, cur_buf, hidden_dim, n_tokens, eps)
          end
          norm_enc.end_encoding

          Profile.trace("prefill.full.kv_cache") do
            proj_enc = ML::Metal::ComputeEncoder.new(cmd)
            if norm_h16_proj && h16_batch_gemm_candidate?(k_qw, n_tokens)
              raise "unsupported h16 final k-cache route" unless encode_matmul_from_h16(proj_enc, k_qw, cur_h16_buf, k_buf, k_w_buf, k_w_off, k_qw.in_dim, k_qw.out_dim, n_tokens)
            else
              encode_matmul(proj_enc, k_pipe.not_nil!, k_qw, cur_buf, k_buf, k_w_buf, k_w_off, k_qw.in_dim, k_qw.out_dim, n_tokens)
            end
            if norm_h16_proj && h16_batch_gemm_candidate?(v_qw, n_tokens)
              raise "unsupported h16 final v-cache route" unless encode_matmul_from_h16(proj_enc, v_qw, cur_h16_buf, v_buf, v_w_buf, v_w_off, v_qw.in_dim, v_qw.out_dim, n_tokens)
            else
              encode_matmul(proj_enc, v_pipe.not_nil!, v_qw, cur_buf, v_buf, v_w_buf, v_w_off, v_qw.in_dim, v_qw.out_dim, n_tokens)
            end
            proj_enc.end_encoding
          end

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

          return true if appended

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          if Profile.enabled?
            Profile.bump_attn(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              0_i64,
            )
            Profile.bump_group("full_kv_only",
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              0_i64)
          end
          true
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
                                                             profile_label : String = "full_rec_chunk_many",
                                                             checkpoint_index : Int32? = nil,
                                                             checkpoint_conv_state_bufs : Array(ML::MetalBuffer)? = nil,
                                                             checkpoint_ssm_state_bufs : Array(ML::MetalBuffer)? = nil,
                                                             checkpoint_rollback_log : Bool = false,
                                                             input_buf : ML::MetalBuffer? = nil,
                                                             output_buf : ML::MetalBuffer? = nil,
                                                             read_output : Bool = true,
                                                             append_command_buffer : ML::Metal::CommandBuffer? = nil) : Array(Float32)?
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
          return nil if append_command_buffer && read_output
          return nil unless conv_state_bufs.size == rec_layers.size && ssm_state_bufs.size == rec_layers.size
          checkpoint_requested = !checkpoint_index.nil?
          if checkpoint_requested
            cp = checkpoint_index.not_nil!
            return nil unless cp >= 0 && cp < n_tokens
            return nil if checkpoint_rollback_log && cp + 1 >= n_tokens
            return nil unless conv_k > 1 && s == 128 && dn_chunk_rowwise_enabled?(s)
            conv_chk = checkpoint_conv_state_bufs
            ssm_chk = checkpoint_ssm_state_bufs
            return nil if conv_chk.nil? || conv_chk.size != rec_layers.size
            return nil if ssm_chk.nil? || ssm_chk.size != rec_layers.size
          end

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
          hidden_elems = n_tokens * hidden_dim
          hidden_bytes = hidden_elems.to_i64 * sizeof(Float32)
          if ib = input_buf
            raise "full+recurrent chunk input buffer too small" if ib.size < hidden_bytes
          else
            raise "full+recurrent chunk input size mismatch" unless inp.size == hidden_elems
          end
          if ob = output_buf
            raise "full+recurrent chunk output buffer too small" if ob.size < hidden_bytes
          end

          full_tag = "frec_full_#{q_qw.raw.to_unsafe.address}"
          inp_buf = input_buf || Scratch.get(:frec_inp, hidden_bytes)
          full_norm_w_buf = Scratch.get("#{full_tag}_norm_w", attn_norm.size.to_i64 * sizeof(Float32))
          full_cur_buf = Scratch.get(:frec_full_cur, hidden_bytes)
          full_cur_h16_buf = Scratch.get(:frec_full_cur_h16, hidden_elems.to_i64 * 2_i64)
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
          full_residual_buf = Scratch.get(:frec_full_residual, hidden_bytes)
          full_normed_buf = Scratch.get(:frec_full_normed, hidden_bytes)
          full_normed_h16_buf = Scratch.get(:frec_full_normed_h16, hidden_elems.to_i64 * 2_i64)
          full_ffn_gate_buf = Scratch.get(:frec_full_ffn_gate, (n_tokens * full_ffn_dim).to_i64 * sizeof(Float32))
          full_ffn_up_buf = Scratch.get(:frec_full_ffn_up, (n_tokens * full_ffn_dim).to_i64 * sizeof(Float32))
          full_ffn_comb_buf = Scratch.get(:frec_full_ffn_comb, (n_tokens * full_ffn_dim).to_i64 * sizeof(Float32))
          full_ffn_comb_h16_buf = Scratch.get(:frec_full_ffn_comb_h16, (n_tokens * full_ffn_dim).to_i64 * 2_i64)
          full_ffn_out_buf = Scratch.get(:frec_full_ffn_out, (n_tokens * ffn_down_qw.out_dim).to_i64 * sizeof(Float32))
          full_out_buf = Scratch.get(:frec_full_out, hidden_bytes)

          rec_dst_buf = Scratch.get(:frec_rec_hidden_b, hidden_bytes)
          rec_cur_buf = Scratch.get(:frec_rec_cur, hidden_bytes)
          rec_cur_h16_buf = Scratch.get(:frec_rec_cur_h16, hidden_elems.to_i64 * 2_i64)
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
          rec_attn_mid_h16_buf = Scratch.get(:frec_rec_mid_h16, (n_tokens * d_inner).to_i64 * 2_i64)
          rec_attn_out_buf = Scratch.get(:frec_rec_attn_out, hidden_bytes)
          rec_residual_buf = Scratch.get(:frec_rec_residual, hidden_bytes)
          rec_normed_buf = Scratch.get(:frec_rec_normed, hidden_bytes)
          rec_normed_h16_buf = Scratch.get(:frec_rec_normed_h16, hidden_elems.to_i64 * 2_i64)
          rec_ffn_gate_buf = Scratch.get(:frec_rec_ffn_gate, (n_tokens * rec_ffn_dim).to_i64 * sizeof(Float32))
          rec_ffn_up_buf = Scratch.get(:frec_rec_ffn_up, (n_tokens * rec_ffn_dim).to_i64 * sizeof(Float32))
          rec_ffn_comb_buf = Scratch.get(:frec_rec_ffn_comb, (n_tokens * rec_ffn_dim).to_i64 * sizeof(Float32))
          rec_ffn_comb_h16_buf = Scratch.get(:frec_rec_ffn_comb_h16, (n_tokens * rec_ffn_dim).to_i64 * 2_i64)
          rec_ffn_out_buf = Scratch.get(:frec_rec_ffn_out, hidden_bytes)

          unless input_buf
            t_upload0 = Time.instant if Profile.enabled?
            inp_buf.write(inp)
            if Profile.enabled?
              t_upload1 = Time.instant
              Profile.bump_group_transfer("#{profile_label}.boundary", hidden_bytes, 0_i64)
              Profile.bump_group("#{profile_label}.upload",
                (t_upload1 - t_upload0.not_nil!).total_nanoseconds.to_i64,
                0_i64,
                0_i64)
            end
          end
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
          cmd = append_command_buffer || ML::Metal::CommandBuffer.new
          appended = !append_command_buffer.nil?
          phase_profile = !appended && prefill_phase_profile_enabled? && Profile.enabled?
          full_detail_profile = prefill_full_detail_profile_enabled? && phase_profile
          phase_t0 = Time.instant

          norm_enc = ML::Metal::ComputeEncoder.new(cmd)
          full_norm_h16_proj = prefill_rmsnorm_h16_proj_enabled? && n_tokens > GEMM_BATCH_THRESHOLD
          if full_norm_h16_proj
            encode_rmsnorm_rows_f32_h16(norm_enc, inp_buf, full_norm_w_buf, full_cur_buf, full_cur_h16_buf, hidden_dim, n_tokens, eps)
          else
            encode_rmsnorm_rows(norm_enc, inp_buf, full_norm_w_buf, full_cur_buf, hidden_dim, n_tokens, eps)
          end
          norm_enc.end_encoding

          Profile.trace("prefill.full.qkv") do
            proj_enc = ML::Metal::ComputeEncoder.new(cmd)
            if full_norm_h16_proj && h16_batch_gemm_candidate?(q_qw, n_tokens)
              raise "unsupported h16 full q route" unless encode_matmul_from_h16(proj_enc, q_qw, full_cur_h16_buf, full_qfull_buf, q_w_buf, q_w_off, q_qw.in_dim, q_qw.out_dim, n_tokens)
            else
              encode_matmul(proj_enc, q_pipe.not_nil!, q_qw, full_cur_buf, full_qfull_buf, q_w_buf, q_w_off, q_qw.in_dim, q_qw.out_dim, n_tokens)
            end
            if full_norm_h16_proj && h16_batch_gemm_candidate?(k_qw, n_tokens)
              raise "unsupported h16 full k route" unless encode_matmul_from_h16(proj_enc, k_qw, full_cur_h16_buf, full_k_buf, k_w_buf, k_w_off, k_qw.in_dim, k_qw.out_dim, n_tokens)
            else
              encode_matmul(proj_enc, k_pipe.not_nil!, k_qw, full_cur_buf, full_k_buf, k_w_buf, k_w_off, k_qw.in_dim, k_qw.out_dim, n_tokens)
            end
            if full_norm_h16_proj && h16_batch_gemm_candidate?(v_qw, n_tokens)
              raise "unsupported h16 full v route" unless encode_matmul_from_h16(proj_enc, v_qw, full_cur_h16_buf, full_v_buf, v_w_buf, v_w_off, v_qw.in_dim, v_qw.out_dim, n_tokens)
            else
              encode_matmul(proj_enc, v_pipe.not_nil!, v_qw, full_cur_buf, full_v_buf, v_w_buf, v_w_off, v_qw.in_dim, v_qw.out_dim, n_tokens)
            end
            proj_enc.end_encoding
          end
          if full_detail_profile
            checked = prefill_phase_checkpoint(cmd, "#{profile_label}.full.qkv", phase_t0)
            cmd = checked[0]
            phase_t0 = checked[1]
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
          if full_detail_profile
            checked = prefill_phase_checkpoint(cmd, "#{profile_label}.full.split_qgate", phase_t0)
            cmd = checked[0]
            phase_t0 = checked[1]
          end

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
          if full_detail_profile
            checked = prefill_phase_checkpoint(cmd, "#{profile_label}.full.qknorm", phase_t0)
            cmd = checked[0]
            phase_t0 = checked[1]
          end

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
          if full_detail_profile
            checked = prefill_phase_checkpoint(cmd, "#{profile_label}.full.rope", phase_t0)
            cmd = checked[0]
            phase_t0 = checked[1]
          end

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
          if full_detail_profile
            checked = prefill_phase_checkpoint(cmd, "#{profile_label}.full.kvwrite", phase_t0)
            cmd = checked[0]
            phase_t0 = checked[1]
          end

          attn_enc = ML::Metal::ComputeEncoder.new(cmd)
          use_attn_sg4 = prefill_attn_rows_sg4_enabled? && n_tokens >= 4
          use_direct_gate = !prefill_attn_rows_sg4_pregate_enabled? && prefill_attn_rows_sg4_direct_gate_enabled?(n_tokens)
          attn_sg4_pipeline = use_direct_gate ? attn_rows_sg4_pipeline : attn_rows_sg4_pregate_pipeline
          attn_enc.set_pipeline(use_attn_sg4 ? attn_sg4_pipeline : attn_rows_pipeline)
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
          if use_attn_sg4
            attn_enc.dispatch_threadgroups({n_head, (n_tokens + 3) // 4, 1}, {128, 1, 1})
          else
            attn_enc.dispatch_threadgroups({n_head, n_tokens, 1}, {32, 1, 1})
          end
          attn_enc.end_encoding
          if full_detail_profile
            checked = prefill_phase_checkpoint(cmd, "#{profile_label}.full.attn_rows", phase_t0)
            cmd = checked[0]
            phase_t0 = checked[1]
          end

          Profile.trace("prefill.full.o_proj") do
            outproj_enc = ML::Metal::ComputeEncoder.new(cmd)
            encode_matmul(outproj_enc, out_pipe.not_nil!, out_qw, full_attn_buf, full_attn_out_buf, out_w_buf, out_w_off, out_qw.in_dim, out_qw.out_dim, n_tokens)
            outproj_enc.end_encoding
          end
          if full_detail_profile
            checked = prefill_phase_checkpoint(cmd, "#{profile_label}.full.o_proj", phase_t0)
            cmd = checked[0]
            phase_t0 = checked[1]
          end

          full_ffn_pair_h16 = prefill_addnorm_h16_ffn_enabled? && q4_pair_h16_gemm_candidate?(ffn_gate_qw, ffn_up_qw, n_tokens)
          addnorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          if full_ffn_pair_h16
            encode_add_rmsnorm_rows_h16(addnorm_enc, inp_buf, full_attn_out_buf, full_post_norm_buf, full_residual_buf, full_normed_h16_buf, hidden_dim, n_tokens, eps)
          else
            encode_add_rmsnorm_rows(addnorm_enc, inp_buf, full_attn_out_buf, full_post_norm_buf, full_residual_buf, full_normed_buf, hidden_dim, n_tokens, eps)
          end
          addnorm_enc.end_encoding

          full_ffn_act_buf = swiglu_inplace_enabled? ? full_ffn_up_buf : full_ffn_comb_buf
          full_ffn_down_h16 = prefill_swiglu_h16_down_candidate?(ffn_down_qw, n_tokens) ||
            q4_b64_up_swiglu_h16_down_candidate?(ffn_gate_qw, ffn_up_qw, ffn_down_qw, n_tokens)
          full_up_swiglu_fused = false

          Profile.trace("prefill.full.ffn_upgate") do
            full_ffn_proj_enc = ML::Metal::ComputeEncoder.new(cmd)
            pair_q4 = q4_pair_h16_gemm_candidate?(ffn_gate_qw, ffn_up_qw, n_tokens)
            if pair_q4
              Profile.bump_matmul_shape("q4_h16_gemm #{ffn_gate_qw.type.name} #{ffn_gate_qw.in_dim}x#{ffn_gate_qw.out_dim} b#{n_tokens}", ffn_gate_qw.raw.size.to_i64)
              Profile.bump_matmul_shape("q4_h16_gemm #{ffn_up_qw.type.name} #{ffn_up_qw.in_dim}x#{ffn_up_qw.out_dim} b#{n_tokens}", ffn_up_qw.raw.size.to_i64)
              if q4_b64_up_swiglu_h16_candidate?(ffn_gate_qw, ffn_up_qw, n_tokens, full_ffn_down_h16)
                full_up_swiglu_fused = true
                if full_ffn_pair_h16
                  encode_q4k_gemm_h16_from_h16(full_ffn_proj_enc, full_normed_h16_buf, full_ffn_gate_buf, full_ffn_gate_w_buf, full_ffn_gate_w_off, ffn_gate_qw.in_dim, ffn_gate_qw.out_dim, n_tokens)
                  encode_q4k_gemm_h16_b64_swiglu_h16_from_h16(full_ffn_proj_enc, full_normed_h16_buf, full_ffn_gate_buf, full_ffn_comb_h16_buf, full_ffn_up_w_buf, full_ffn_up_w_off, ffn_up_qw.in_dim, ffn_up_qw.out_dim, n_tokens)
                else
                  encode_q4k_gemm_h16_pair_b64_swiglu_h16(full_ffn_proj_enc, full_normed_buf, full_ffn_gate_buf, full_ffn_comb_h16_buf, full_ffn_gate_w_buf, full_ffn_gate_w_off, full_ffn_up_w_buf, full_ffn_up_w_off, ffn_gate_qw.in_dim, ffn_gate_qw.out_dim, n_tokens)
                end
              elsif q4_b64_up_swiglu_candidate?(ffn_gate_qw, ffn_up_qw, n_tokens, full_ffn_down_h16)
                full_up_swiglu_fused = true
                if full_ffn_pair_h16
                  encode_q4k_gemm_h16_from_h16(full_ffn_proj_enc, full_normed_h16_buf, full_ffn_gate_buf, full_ffn_gate_w_buf, full_ffn_gate_w_off, ffn_gate_qw.in_dim, ffn_gate_qw.out_dim, n_tokens)
                  encode_q4k_gemm_h16_b64_swiglu_from_h16(full_ffn_proj_enc, full_normed_h16_buf, full_ffn_gate_buf, full_ffn_act_buf, full_ffn_up_w_buf, full_ffn_up_w_off, ffn_up_qw.in_dim, ffn_up_qw.out_dim, n_tokens)
                else
                  encode_q4k_gemm_h16_pair_b64_swiglu(full_ffn_proj_enc, full_normed_buf, full_ffn_gate_buf, full_ffn_act_buf, full_ffn_gate_w_buf, full_ffn_gate_w_off, full_ffn_up_w_buf, full_ffn_up_w_off, ffn_gate_qw.in_dim, ffn_gate_qw.out_dim, n_tokens)
                end
              elsif full_ffn_pair_h16
                encode_q4k_gemm_h16_from_h16(full_ffn_proj_enc, full_normed_h16_buf, full_ffn_gate_buf, full_ffn_gate_w_buf, full_ffn_gate_w_off, ffn_gate_qw.in_dim, ffn_gate_qw.out_dim, n_tokens)
                encode_q4k_gemm_h16_from_h16(full_ffn_proj_enc, full_normed_h16_buf, full_ffn_up_buf, full_ffn_up_w_buf, full_ffn_up_w_off, ffn_up_qw.in_dim, ffn_up_qw.out_dim, n_tokens)
              else
                encode_q4k_gemm_h16_pair(full_ffn_proj_enc, full_normed_buf, full_ffn_gate_buf, full_ffn_up_buf, full_ffn_gate_w_buf, full_ffn_gate_w_off, full_ffn_up_w_buf, full_ffn_up_w_off, ffn_gate_qw.in_dim, ffn_gate_qw.out_dim, n_tokens)
              end
            else
              encode_matmul(full_ffn_proj_enc, full_ffn_gate_pipe.not_nil!, ffn_gate_qw, full_normed_buf, full_ffn_gate_buf, full_ffn_gate_w_buf, full_ffn_gate_w_off, ffn_gate_qw.in_dim, ffn_gate_qw.out_dim, n_tokens)
              encode_matmul(full_ffn_proj_enc, full_ffn_up_pipe.not_nil!, ffn_up_qw, full_normed_buf, full_ffn_up_buf, full_ffn_up_w_buf, full_ffn_up_w_off, ffn_up_qw.in_dim, ffn_up_qw.out_dim, n_tokens)
            end
            full_ffn_proj_enc.end_encoding
          end
          if full_detail_profile
            checked = prefill_phase_checkpoint(cmd, "#{profile_label}.full.ffn_upgate", phase_t0)
            cmd = checked[0]
            phase_t0 = checked[1]
          end

          unless full_up_swiglu_fused
            full_swiglu_enc = ML::Metal::ComputeEncoder.new(cmd)
            full_swiglu_enc.set_pipeline(full_ffn_down_h16 ? ffn_swiglu_h16_pipeline : ffn_swiglu_pipeline)
            full_swiglu_enc.set_buffer(full_ffn_gate_buf, 0)
            full_swiglu_enc.set_buffer(full_ffn_up_buf, 1)
            full_swiglu_enc.set_buffer(full_ffn_down_h16 ? full_ffn_comb_h16_buf : full_ffn_act_buf, 2, ML::Metal::BufferAccess::Write)
            full_swiglu_enc.set_value((n_tokens * full_ffn_dim).to_u32, 3)
            full_swiglu_enc.dispatch_1d(n_tokens * full_ffn_dim, 256)
            full_swiglu_enc.end_encoding
          end

          fused_down_add = false
          if prefill_ffn_down_add_fused_enabled?
            Profile.trace("prefill.full.ffn_down_add") do
              full_ffn_down_enc = ML::Metal::ComputeEncoder.new(cmd)
              fused_down_add = if full_ffn_down_h16
                                 encode_matmul_add_from_h16(full_ffn_down_enc, ffn_down_qw, full_ffn_comb_h16_buf, full_residual_buf, full_out_buf, full_ffn_down_w_buf, full_ffn_down_w_off, ffn_down_qw.in_dim, ffn_down_qw.out_dim, n_tokens)
                               else
                                 encode_matmul_add(full_ffn_down_enc, full_ffn_down_pipe.not_nil!, ffn_down_qw, full_ffn_act_buf, full_residual_buf, full_out_buf, full_ffn_down_w_buf, full_ffn_down_w_off, ffn_down_qw.in_dim, ffn_down_qw.out_dim, n_tokens)
                               end
              full_ffn_down_enc.end_encoding
            end
          end

          unless fused_down_add
            Profile.trace("prefill.full.ffn_down") do
              full_ffn_down_enc = ML::Metal::ComputeEncoder.new(cmd)
              if full_ffn_down_h16
                raise "unsupported h16 full FFN-down route" unless encode_matmul_from_h16(full_ffn_down_enc, ffn_down_qw, full_ffn_comb_h16_buf, full_ffn_out_buf, full_ffn_down_w_buf, full_ffn_down_w_off, ffn_down_qw.in_dim, ffn_down_qw.out_dim, n_tokens)
              else
                encode_matmul(full_ffn_down_enc, full_ffn_down_pipe.not_nil!, ffn_down_qw, full_ffn_act_buf, full_ffn_out_buf, full_ffn_down_w_buf, full_ffn_down_w_off, ffn_down_qw.in_dim, ffn_down_qw.out_dim, n_tokens)
              end
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
          if full_detail_profile
            checkpoint_name = fused_down_add ? "ffn_down_add" : "ffn_down"
            checked = prefill_phase_checkpoint(cmd, "#{profile_label}.full.#{checkpoint_name}", phase_t0)
            cmd = checked[0]
            phase_t0 = checked[1]
          end

          if phase_profile && !full_detail_profile
            phase_tenc = Time.instant
            cmd.commit
            cmd.wait
            phase_twait = Time.instant
            Profile.bump_group("#{profile_label}.full",
              (phase_tenc - phase_t0).total_nanoseconds.to_i64,
              (phase_twait - phase_tenc).total_nanoseconds.to_i64,
              0_i64)
            cmd = ML::Metal::CommandBuffer.new
            phase_t0 = Time.instant
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
            rec_norm_h16_proj = prefill_rmsnorm_h16_proj_enabled? && n_tokens > GEMM_BATCH_THRESHOLD
            if rec_norm_h16_proj
              encode_rmsnorm_rows_f32_h16(rec_norm_enc, src_buf, norm_w_buf, rec_cur_buf, rec_cur_h16_buf, hidden_dim, n_tokens, eps)
            else
              encode_rmsnorm_rows(rec_norm_enc, src_buf, norm_w_buf, rec_cur_buf, hidden_dim, n_tokens, eps)
            end
            rec_norm_enc.end_encoding

            Profile.trace("prefill.rec.proj") do
              rec_proj_enc = ML::Metal::ComputeEncoder.new(cmd)
              qkv_h16 = !checkpoint_requested && q5_qkv_h16_conv_enabled? && q56_batch_gemm_enabled? && lw.attn_qkv_qw.type.q5_k? && n_tokens > GEMM_BATCH_THRESHOLD
              shared_h16 = rec_proj_shared_h16_enabled? && qkv_h16 && q4_h16_gemm_enabled? &&
                           lw.attn_gate_qw.type.q4_k? && n_tokens > GEMM_BATCH_THRESHOLD
              if shared_h16 && rec_norm_h16_proj
                Profile.bump_matmul_shape("q5_h16_gemm #{lw.attn_qkv_qw.type.name} #{lw.attn_qkv_qw.in_dim}x#{lw.attn_qkv_qw.out_dim} b#{n_tokens}", lw.attn_qkv_qw.raw.size.to_i64)
                encode_q56k_gemm_h16_from_h16(rec_proj_enc, mm5_pipeline, rec_cur_h16_buf, rec_qkv_h16_buf, qkv_w_buf, qkv_w_off, lw.attn_qkv_qw.in_dim, lw.attn_qkv_qw.out_dim, n_tokens)
                Profile.bump_matmul_shape("q4_h16_gemm #{lw.attn_gate_qw.type.name} #{lw.attn_gate_qw.in_dim}x#{lw.attn_gate_qw.out_dim} b#{n_tokens}", lw.attn_gate_qw.raw.size.to_i64)
                encode_q4k_gemm_h16_from_h16(rec_proj_enc, rec_cur_h16_buf, rec_z_buf, gate_w_buf, gate_w_off, lw.attn_gate_qw.in_dim, lw.attn_gate_qw.out_dim, n_tokens)
              elsif shared_h16
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
              elsif qkv_h16 && rec_norm_h16_proj
                Profile.bump_matmul_shape("q5_h16_gemm #{lw.attn_qkv_qw.type.name} #{lw.attn_qkv_qw.in_dim}x#{lw.attn_qkv_qw.out_dim} b#{n_tokens}", lw.attn_qkv_qw.raw.size.to_i64)
                encode_q56k_gemm_h16_from_h16(rec_proj_enc, mm5_pipeline, rec_cur_h16_buf, rec_qkv_h16_buf, qkv_w_buf, qkv_w_off, lw.attn_qkv_qw.in_dim, lw.attn_qkv_qw.out_dim, n_tokens)
                encode_matmul(rec_proj_enc, gemv_pipeline_for(lw.attn_gate_qw).not_nil!, lw.attn_gate_qw, rec_cur_buf, rec_z_buf, gate_w_buf, gate_w_off, lw.attn_gate_qw.in_dim, lw.attn_gate_qw.out_dim, n_tokens)
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
            if full_detail_profile
              checked = prefill_phase_checkpoint(cmd, "#{profile_label}.rec#{local_i}.proj", phase_t0)
              cmd = checked[0]
              phase_t0 = checked[1]
            end

            conv_enc = ML::Metal::ComputeEncoder.new(cmd)
            qkv_h16 = !checkpoint_requested && q5_qkv_h16_conv_enabled? && q56_batch_gemm_enabled? && lw.attn_qkv_qw.type.q5_k? && n_tokens > GEMM_BATCH_THRESHOLD
            conv_enc.set_pipeline(checkpoint_requested ? recurrent_conv_shift_chunk_checkpoint_pipeline : (qkv_h16 ? recurrent_conv_shift_chunk_h16_pipeline : recurrent_conv_shift_chunk_pipeline))
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
            if checkpoint_requested
              conv_enc.set_buffer(checkpoint_conv_state_bufs.not_nil![local_i], 11, ML::Metal::BufferAccess::Write)
              conv_enc.set_value(checkpoint_index.not_nil!.to_u32, 12)
            end
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
            if full_detail_profile
              checked = prefill_phase_checkpoint(cmd, "#{profile_label}.rec#{local_i}.prep", phase_t0)
              cmd = checked[0]
              phase_t0 = checked[1]
            end

            dn_enc = ML::Metal::ComputeEncoder.new(cmd)
            use_dn_rowwise = dn_chunk_rowwise_enabled?(s)
            dn_enc.set_pipeline(checkpoint_requested ? (checkpoint_rollback_log ? dn128_chunk_rowwise_rollback_log_pipeline : dn128_chunk_rowwise_checkpoint_pipeline) : (use_dn_rowwise ? dn128_chunk_rowwise_pipeline : dn128_chunk_fused_pipeline))
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
            if checkpoint_requested
              dn_enc.set_buffer(checkpoint_ssm_state_bufs.not_nil![local_i], 12, ML::Metal::BufferAccess::Write)
              log_index = checkpoint_rollback_log ? checkpoint_index.not_nil! + 1 : checkpoint_index.not_nil!
              dn_enc.set_value(log_index.to_u32, 13)
            end
            if use_dn_rowwise
              dn_enc.dispatch_threadgroups({(s + 3) // 4, h_v, 1}, {32, 4, 1})
            else
              dn_enc.dispatch_threadgroups({h_v, 1, 1}, {128, 1, 1})
            end
            dn_enc.end_encoding
            if full_detail_profile
              checked = prefill_phase_checkpoint(cmd, "#{profile_label}.rec#{local_i}.dn", phase_t0)
              cmd = checked[0]
              phase_t0 = checked[1]
            end

            post_enc = ML::Metal::ComputeEncoder.new(cmd)
            rec_o_proj_h16 = prefill_dn_post_h16_oproj_enabled? && h16_batch_gemm_candidate?(lw.ssm_out_qw, n_tokens)
            post_enc.set_pipeline(rec_o_proj_h16 ? dn_post_chunk_h16_pipeline : dn_post_chunk_pipeline)
            post_enc.set_buffer(rec_attn_mid_buf, 0, ML::Metal::BufferAccess::ReadWrite)
            post_enc.set_buffer(rec_z_buf, 1)
            post_enc.set_buffer(ssm_norm_buf, 2)
            if rec_o_proj_h16
              post_enc.set_buffer(rec_attn_mid_h16_buf, 3, ML::Metal::BufferAccess::Write)
              post_enc.set_value(h_v.to_u32, 4)
              post_enc.set_value(s.to_u32, 5)
              post_enc.set_value(eps, 6)
              post_enc.set_value(n_tokens.to_u32, 7)
            else
              post_enc.set_value(h_v.to_u32, 3)
              post_enc.set_value(s.to_u32, 4)
              post_enc.set_value(eps, 5)
              post_enc.set_value(n_tokens.to_u32, 6)
            end
            post_enc.dispatch_threadgroups({h_v, n_tokens, 1}, {32, 1, 1})
            post_enc.end_encoding

            Profile.trace("prefill.rec.o_proj") do
              rec_out_enc = ML::Metal::ComputeEncoder.new(cmd)
              if rec_o_proj_h16
                raise "unsupported h16 recurrent o_proj route" unless encode_matmul_from_h16(rec_out_enc, lw.ssm_out_qw, rec_attn_mid_h16_buf, rec_attn_out_buf, rec_out_w_buf, rec_out_w_off, lw.ssm_out_qw.in_dim, lw.ssm_out_qw.out_dim, n_tokens)
              else
                encode_matmul(rec_out_enc, gemv_pipeline_for(lw.ssm_out_qw).not_nil!, lw.ssm_out_qw, rec_attn_mid_buf, rec_attn_out_buf, rec_out_w_buf, rec_out_w_off, lw.ssm_out_qw.in_dim, lw.ssm_out_qw.out_dim, n_tokens)
              end
              rec_out_enc.end_encoding
            end
            if full_detail_profile
              checked = prefill_phase_checkpoint(cmd, "#{profile_label}.rec#{local_i}.post_oproj", phase_t0)
              cmd = checked[0]
              phase_t0 = checked[1]
            end

            rec_ffn_pair_h16 = prefill_addnorm_h16_ffn_enabled? && q4_pair_h16_gemm_candidate?(lw.ffn_gate_qw, lw.ffn_up_qw, n_tokens)
            rec_addnorm_enc = ML::Metal::ComputeEncoder.new(cmd)
            if rec_ffn_pair_h16
              encode_add_rmsnorm_rows_h16(rec_addnorm_enc, src_buf, rec_attn_out_buf, post_w_buf, rec_residual_buf, rec_normed_h16_buf, hidden_dim, n_tokens, eps)
            else
              encode_add_rmsnorm_rows(rec_addnorm_enc, src_buf, rec_attn_out_buf, post_w_buf, rec_residual_buf, rec_normed_buf, hidden_dim, n_tokens, eps)
            end
            rec_addnorm_enc.end_encoding

            rec_ffn_act_buf = swiglu_inplace_enabled? ? rec_ffn_up_buf : rec_ffn_comb_buf
            rec_ffn_down_h16 = prefill_swiglu_h16_down_candidate?(lw.ffn_down_qw, n_tokens) ||
              q4_b64_up_swiglu_h16_down_candidate?(lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw, n_tokens)
            rec_up_swiglu_fused = false

            Profile.trace("prefill.rec.ffn_upgate") do
              rec_ffn_proj_enc = ML::Metal::ComputeEncoder.new(cmd)
              pair_q4 = q4_pair_h16_gemm_candidate?(lw.ffn_gate_qw, lw.ffn_up_qw, n_tokens)
              if pair_q4
                Profile.bump_matmul_shape("q4_h16_gemm #{lw.ffn_gate_qw.type.name} #{lw.ffn_gate_qw.in_dim}x#{lw.ffn_gate_qw.out_dim} b#{n_tokens}", lw.ffn_gate_qw.raw.size.to_i64)
                Profile.bump_matmul_shape("q4_h16_gemm #{lw.ffn_up_qw.type.name} #{lw.ffn_up_qw.in_dim}x#{lw.ffn_up_qw.out_dim} b#{n_tokens}", lw.ffn_up_qw.raw.size.to_i64)
                if q4_b64_up_swiglu_h16_candidate?(lw.ffn_gate_qw, lw.ffn_up_qw, n_tokens, rec_ffn_down_h16)
                  rec_up_swiglu_fused = true
                  if rec_ffn_pair_h16
                    encode_q4k_gemm_h16_from_h16(rec_ffn_proj_enc, rec_normed_h16_buf, rec_ffn_gate_buf, rec_ffn_gate_w_buf, rec_ffn_gate_w_off, lw.ffn_gate_qw.in_dim, lw.ffn_gate_qw.out_dim, n_tokens)
                    encode_q4k_gemm_h16_b64_swiglu_h16_from_h16(rec_ffn_proj_enc, rec_normed_h16_buf, rec_ffn_gate_buf, rec_ffn_comb_h16_buf, rec_ffn_up_w_buf, rec_ffn_up_w_off, lw.ffn_up_qw.in_dim, lw.ffn_up_qw.out_dim, n_tokens)
                  else
                    encode_q4k_gemm_h16_pair_b64_swiglu_h16(rec_ffn_proj_enc, rec_normed_buf, rec_ffn_gate_buf, rec_ffn_comb_h16_buf, rec_ffn_gate_w_buf, rec_ffn_gate_w_off, rec_ffn_up_w_buf, rec_ffn_up_w_off, lw.ffn_gate_qw.in_dim, lw.ffn_gate_qw.out_dim, n_tokens)
                  end
                elsif q4_b64_up_swiglu_candidate?(lw.ffn_gate_qw, lw.ffn_up_qw, n_tokens, rec_ffn_down_h16)
                  rec_up_swiglu_fused = true
                  if rec_ffn_pair_h16
                    encode_q4k_gemm_h16_from_h16(rec_ffn_proj_enc, rec_normed_h16_buf, rec_ffn_gate_buf, rec_ffn_gate_w_buf, rec_ffn_gate_w_off, lw.ffn_gate_qw.in_dim, lw.ffn_gate_qw.out_dim, n_tokens)
                    encode_q4k_gemm_h16_b64_swiglu_from_h16(rec_ffn_proj_enc, rec_normed_h16_buf, rec_ffn_gate_buf, rec_ffn_act_buf, rec_ffn_up_w_buf, rec_ffn_up_w_off, lw.ffn_up_qw.in_dim, lw.ffn_up_qw.out_dim, n_tokens)
                  else
                    encode_q4k_gemm_h16_pair_b64_swiglu(rec_ffn_proj_enc, rec_normed_buf, rec_ffn_gate_buf, rec_ffn_act_buf, rec_ffn_gate_w_buf, rec_ffn_gate_w_off, rec_ffn_up_w_buf, rec_ffn_up_w_off, lw.ffn_gate_qw.in_dim, lw.ffn_gate_qw.out_dim, n_tokens)
                  end
                elsif rec_ffn_pair_h16
                  encode_q4k_gemm_h16_from_h16(rec_ffn_proj_enc, rec_normed_h16_buf, rec_ffn_gate_buf, rec_ffn_gate_w_buf, rec_ffn_gate_w_off, lw.ffn_gate_qw.in_dim, lw.ffn_gate_qw.out_dim, n_tokens)
                  encode_q4k_gemm_h16_from_h16(rec_ffn_proj_enc, rec_normed_h16_buf, rec_ffn_up_buf, rec_ffn_up_w_buf, rec_ffn_up_w_off, lw.ffn_up_qw.in_dim, lw.ffn_up_qw.out_dim, n_tokens)
                else
                  encode_q4k_gemm_h16_pair(rec_ffn_proj_enc, rec_normed_buf, rec_ffn_gate_buf, rec_ffn_up_buf, rec_ffn_gate_w_buf, rec_ffn_gate_w_off, rec_ffn_up_w_buf, rec_ffn_up_w_off, lw.ffn_gate_qw.in_dim, lw.ffn_gate_qw.out_dim, n_tokens)
                end
              else
                encode_matmul(rec_ffn_proj_enc, gemv_pipeline_for(lw.ffn_gate_qw).not_nil!, lw.ffn_gate_qw, rec_normed_buf, rec_ffn_gate_buf, rec_ffn_gate_w_buf, rec_ffn_gate_w_off, lw.ffn_gate_qw.in_dim, lw.ffn_gate_qw.out_dim, n_tokens)
                encode_matmul(rec_ffn_proj_enc, gemv_pipeline_for(lw.ffn_up_qw).not_nil!, lw.ffn_up_qw, rec_normed_buf, rec_ffn_up_buf, rec_ffn_up_w_buf, rec_ffn_up_w_off, lw.ffn_up_qw.in_dim, lw.ffn_up_qw.out_dim, n_tokens)
              end
              rec_ffn_proj_enc.end_encoding
            end
            if full_detail_profile
              checked = prefill_phase_checkpoint(cmd, "#{profile_label}.rec#{local_i}.ffn_upgate", phase_t0)
              cmd = checked[0]
              phase_t0 = checked[1]
            end

            unless rec_up_swiglu_fused
              rec_swiglu_enc = ML::Metal::ComputeEncoder.new(cmd)
              rec_swiglu_enc.set_pipeline(rec_ffn_down_h16 ? ffn_swiglu_h16_pipeline : ffn_swiglu_pipeline)
              rec_swiglu_enc.set_buffer(rec_ffn_gate_buf, 0)
              rec_swiglu_enc.set_buffer(rec_ffn_up_buf, 1)
              rec_swiglu_enc.set_buffer(rec_ffn_down_h16 ? rec_ffn_comb_h16_buf : rec_ffn_act_buf, 2, ML::Metal::BufferAccess::Write)
              rec_swiglu_enc.set_value((n_tokens * rec_ffn_dim).to_u32, 3)
              rec_swiglu_enc.dispatch_1d(n_tokens * rec_ffn_dim, 256)
              rec_swiglu_enc.end_encoding
            end

            fused_down_add = false
            if prefill_ffn_down_add_fused_enabled?
              Profile.trace("prefill.rec.ffn_down_add") do
                rec_ffn_down_enc = ML::Metal::ComputeEncoder.new(cmd)
                fused_down_add = if rec_ffn_down_h16
                                   encode_matmul_add_from_h16(rec_ffn_down_enc, lw.ffn_down_qw, rec_ffn_comb_h16_buf, rec_residual_buf, dst_buf, rec_ffn_down_w_buf, rec_ffn_down_w_off, lw.ffn_down_qw.in_dim, lw.ffn_down_qw.out_dim, n_tokens)
                                 else
                                   encode_matmul_add(rec_ffn_down_enc, gemv_pipeline_for(lw.ffn_down_qw).not_nil!, lw.ffn_down_qw, rec_ffn_act_buf, rec_residual_buf, dst_buf, rec_ffn_down_w_buf, rec_ffn_down_w_off, lw.ffn_down_qw.in_dim, lw.ffn_down_qw.out_dim, n_tokens)
                                 end
                rec_ffn_down_enc.end_encoding
              end
            end

            unless fused_down_add
              Profile.trace("prefill.rec.ffn_down") do
                rec_ffn_down_enc = ML::Metal::ComputeEncoder.new(cmd)
                if rec_ffn_down_h16
                  raise "unsupported h16 recurrent FFN-down route" unless encode_matmul_from_h16(rec_ffn_down_enc, lw.ffn_down_qw, rec_ffn_comb_h16_buf, rec_ffn_out_buf, rec_ffn_down_w_buf, rec_ffn_down_w_off, lw.ffn_down_qw.in_dim, lw.ffn_down_qw.out_dim, n_tokens)
                else
                  encode_matmul(rec_ffn_down_enc, gemv_pipeline_for(lw.ffn_down_qw).not_nil!, lw.ffn_down_qw, rec_ffn_act_buf, rec_ffn_out_buf, rec_ffn_down_w_buf, rec_ffn_down_w_off, lw.ffn_down_qw.in_dim, lw.ffn_down_qw.out_dim, n_tokens)
                end
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
            if full_detail_profile
              checkpoint_name = fused_down_add ? "ffn_down_add" : "ffn_down"
              checked = prefill_phase_checkpoint(cmd, "#{profile_label}.rec#{local_i}.#{checkpoint_name}", phase_t0)
              cmd = checked[0]
              phase_t0 = checked[1]
            end

            src_buf, dst_buf = dst_buf, src_buf

            if phase_profile && !full_detail_profile
              phase_tenc = Time.instant
              cmd.commit
              cmd.wait
              phase_twait = Time.instant
              Profile.bump_group("#{profile_label}.rec#{local_i}",
                (phase_tenc - phase_t0).total_nanoseconds.to_i64,
                (phase_twait - phase_tenc).total_nanoseconds.to_i64,
                0_i64)
              cmd = ML::Metal::CommandBuffer.new
              phase_t0 = Time.instant
            end
          end

          if phase_profile
            if ob = output_buf
              ob.copy_from(src_buf, hidden_bytes)
            end
            t_read0 = Time.instant
            result = read_output ? read_shared_f32(src_buf, hidden_elems) : [] of Float32
            t_read = Time.instant
            Profile.bump_group_transfer("#{profile_label}.boundary", 0_i64, hidden_bytes) if read_output
            Profile.bump_group("#{profile_label}.read", 0_i64, 0_i64, (t_read - t_read0).total_nanoseconds.to_i64)
            return result
          end

          if ob = output_buf
            blit = ML::Metal::BlitEncoder.new(cmd)
            blit.copy_buffer(src_buf, 0, ob, 0, hidden_bytes.to_i32)
            blit.end_encoding
          end
          return [] of Float32 if appended
          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_output ? read_shared_f32(src_buf, hidden_elems) : [] of Float32
          if Profile.enabled?
            t_read = Time.instant
            encode_ns = (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64
            wait_ns = (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64
            read_ns = read_output ? (t_read - t_wait.not_nil!).total_nanoseconds.to_i64 : 0_i64
            Profile.bump_group_transfer("#{profile_label}.boundary", 0_i64, hidden_bytes) if read_output
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
          profile_bump_head_top1_shape("head_top1", out_qw)
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

        def self.rmsnorm_project_top1_allowed_ids(x : Array(Float32),
                                                  norm_weight : Array(Float32),
                                                  out_qw : QuantWeight,
                                                  eps : Float32,
                                                  allowed_ids : Array(Int32)) : Array(Float32)?
          return nil unless head_top1_fused_enabled? && out_qw.type.q6_k? && out_qw.in_dim % QK_K == 0
          return nil if allowed_ids.empty?
          raise "rmsnorm_project_top1_allowed_ids input mismatch: expected #{out_qw.in_dim}, got #{x.size}" unless x.size == out_qw.in_dim
          raise "rmsnorm_project_top1_allowed_ids norm mismatch: expected #{out_qw.in_dim}, got #{norm_weight.size}" unless norm_weight.size == out_qw.in_dim
          allowed_ids.each do |id|
            raise "allowed token id #{id} out of range 0...#{out_qw.out_dim}" if id < 0 || id >= out_qw.out_dim
          end

          ML::Metal::Device.init!

          hidden_dim = x.size
          allowed_n = allowed_ids.size
          tile_count = (allowed_n + HEAD_TOP1_ROWS_PER_TG - 1) // HEAD_TOP1_ROWS_PER_TG
          x_buf = Scratch.get(:head_top1_allowed_x, hidden_dim.to_i64 * sizeof(Float32))
          norm_w_buf = Scratch.get(:head_top1_allowed_norm_w, norm_weight.size.to_i64 * sizeof(Float32))
          normed_buf = Scratch.get(:head_top1_allowed_normed, hidden_dim.to_i64 * sizeof(Float32))
          allowed_ids_buf = Scratch.get(:head_top1_allowed_ids, allowed_n.to_i64 * sizeof(UInt32))
          tile_values_buf = Scratch.get(:head_top1_allowed_tile_values, tile_count.to_i64 * sizeof(Float32))
          tile_ids_buf = Scratch.get(:head_top1_allowed_tile_ids, tile_count.to_i64 * sizeof(UInt32))
          top1_id_buf = Scratch.get(:head_top1_allowed_id, sizeof(UInt32).to_i64)
          top1_value_buf = Scratch.get(:head_top1_allowed_value, sizeof(Float32).to_i64)
          x_buf.write(x)
          norm_w_buf.write(norm_weight)
          allowed_ptr = allowed_ids_buf.contents.as(Pointer(UInt32))
          allowed_ids.each_with_index { |id, i| allowed_ptr[i] = id.to_u32 }

          out_w_buf, out_w_off = weight_slot(out_qw)

          t0 = Time.instant if Profile.enabled?
          cmd = ML::Metal::CommandBuffer.new

          norm_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_rmsnorm_vec(norm_enc, x_buf, norm_w_buf, normed_buf, hidden_dim, eps)
          norm_enc.end_encoding

          head_top1_enc = ML::Metal::ComputeEncoder.new(cmd)
          head_top1_enc.set_pipeline(mv6_top1_allowed_tiles_pipeline)
          profile_bump_head_top1_shape("head_top1_allowed#{allowed_n}", out_qw, rows: allowed_n)
          head_top1_enc.set_buffer(out_w_buf, 0, ML::Metal::BufferAccess::Read, offset: out_w_off)
          head_top1_enc.set_buffer(normed_buf, 1)
          head_top1_enc.set_buffer(allowed_ids_buf, 2)
          head_top1_enc.set_buffer(tile_values_buf, 3, ML::Metal::BufferAccess::Write)
          head_top1_enc.set_buffer(tile_ids_buf, 4, ML::Metal::BufferAccess::Write)
          head_top1_enc.set_value(out_qw.in_dim.to_u32, 5)
          head_top1_enc.set_value(out_qw.out_dim.to_u32, 6)
          head_top1_enc.set_value(allowed_n.to_u32, 7)
          head_top1_enc.dispatch_threadgroups({tile_count, 1, 1}, {64, 1, 1})
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

        def self.project_top1_no_norm(out_qw : QuantWeight,
                                      x : Array(Float32)) : Array(Float32)?
          return nil unless can_use_head_top1_fused?(out_qw)
          raise "project_top1_no_norm input mismatch: expected #{out_qw.in_dim}, got #{x.size}" unless x.size == out_qw.in_dim

          ML::Metal::Device.init!

          tile_count = (out_qw.out_dim + HEAD_TOP1_ROWS_PER_TG - 1) // HEAD_TOP1_ROWS_PER_TG
          x_buf = Scratch.get(:head_top1_nonorm_x, out_qw.in_dim.to_i64 * sizeof(Float32))
          tile_values_buf = Scratch.get(:head_top1_nonorm_tile_values, tile_count.to_i64 * sizeof(Float32))
          tile_ids_buf = Scratch.get(:head_top1_nonorm_tile_ids, tile_count.to_i64 * sizeof(UInt32))
          top1_id_buf = Scratch.get(:head_top1_nonorm_id, sizeof(UInt32).to_i64)
          top1_value_buf = Scratch.get(:head_top1_nonorm_value, sizeof(Float32).to_i64)
          x_buf.write(x)

          out_w_buf, out_w_off = weight_slot(out_qw)

          t0 = Time.instant if Profile.enabled?
          cmd = ML::Metal::CommandBuffer.new

          head_top1_enc = ML::Metal::ComputeEncoder.new(cmd)
          head_top1_enc.set_pipeline(out_qw.type.q8_0? ? mv8_top1_tiles_pipeline : mv6_top1_tiles_pipeline)
          profile_bump_head_top1_shape("head_top1_no_norm", out_qw)
          head_top1_enc.set_buffer(out_w_buf, 0, ML::Metal::BufferAccess::Read, offset: out_w_off)
          head_top1_enc.set_buffer(x_buf, 1)
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

        def self.project_top2_no_norm(out_qw : QuantWeight,
                                      x : Array(Float32)) : Array(Float32)?
          return nil unless can_use_head_top1_fused?(out_qw)
          raise "project_top2_no_norm input mismatch: expected #{out_qw.in_dim}, got #{x.size}" unless x.size == out_qw.in_dim

          ML::Metal::Device.init!

          tile_count = (out_qw.out_dim + HEAD_TOP1_ROWS_PER_TG - 1) // HEAD_TOP1_ROWS_PER_TG
          x_buf = Scratch.get(:head_top2_nonorm_x, out_qw.in_dim.to_i64 * sizeof(Float32))
          tile_values_buf = Scratch.get(:head_top2_nonorm_tile_values, tile_count.to_i64 * sizeof(Float32))
          tile_ids_buf = Scratch.get(:head_top2_nonorm_tile_ids, tile_count.to_i64 * sizeof(UInt32))
          second_tile_values_buf = Scratch.get(:head_top2_nonorm_second_tile_values, tile_count.to_i64 * sizeof(Float32))
          second_tile_ids_buf = Scratch.get(:head_top2_nonorm_second_tile_ids, tile_count.to_i64 * sizeof(UInt32))
          top1_id_buf = Scratch.get(:head_top2_nonorm_id, sizeof(UInt32).to_i64)
          top1_value_buf = Scratch.get(:head_top2_nonorm_value, sizeof(Float32).to_i64)
          second_id_buf = Scratch.get(:head_top2_nonorm_second_id, sizeof(UInt32).to_i64)
          second_value_buf = Scratch.get(:head_top2_nonorm_second_value, sizeof(Float32).to_i64)
          x_buf.write(x)

          out_w_buf, out_w_off = weight_slot(out_qw)

          t0 = Time.instant if Profile.enabled?
          cmd = ML::Metal::CommandBuffer.new

          head_top2_enc = ML::Metal::ComputeEncoder.new(cmd)
          head_top2_enc.set_pipeline(out_qw.type.q8_0? ? mv8_top2_tiles_pipeline : mv6_top2_tiles_pipeline)
          head_top2_enc.set_buffer(out_w_buf, 0, ML::Metal::BufferAccess::Read, offset: out_w_off)
          head_top2_enc.set_buffer(x_buf, 1)
          head_top2_enc.set_buffer(tile_values_buf, 2, ML::Metal::BufferAccess::Write)
          head_top2_enc.set_buffer(tile_ids_buf, 3, ML::Metal::BufferAccess::Write)
          head_top2_enc.set_buffer(second_tile_values_buf, 4, ML::Metal::BufferAccess::Write)
          head_top2_enc.set_buffer(second_tile_ids_buf, 5, ML::Metal::BufferAccess::Write)
          head_top2_enc.set_value(out_qw.in_dim.to_u32, 6)
          head_top2_enc.set_value(out_qw.out_dim.to_u32, 7)
          head_top2_enc.dispatch_threadgroups({tile_count, 1, 1}, {out_qw.type.q8_0? ? MV_Q8_NSG * 32 : 64, 1, 1})
          head_top2_enc.end_encoding

          reduce_top2_enc = ML::Metal::ComputeEncoder.new(cmd)
          reduce_top2_enc.set_pipeline(top2_reduce_tiles_pipeline)
          reduce_top2_enc.set_buffer(tile_values_buf, 0)
          reduce_top2_enc.set_buffer(tile_ids_buf, 1)
          reduce_top2_enc.set_buffer(second_tile_values_buf, 2)
          reduce_top2_enc.set_buffer(second_tile_ids_buf, 3)
          reduce_top2_enc.set_buffer(top1_id_buf, 4, ML::Metal::BufferAccess::Write)
          reduce_top2_enc.set_buffer(top1_value_buf, 5, ML::Metal::BufferAccess::Write)
          reduce_top2_enc.set_buffer(second_id_buf, 6, ML::Metal::BufferAccess::Write)
          reduce_top2_enc.set_buffer(second_value_buf, 7, ML::Metal::BufferAccess::Write)
          reduce_top2_enc.set_value(tile_count.to_u32, 8)
          reduce_top2_enc.dispatch_threadgroups({1, 1, 1}, {256, 1, 1})
          reduce_top2_enc.end_encoding

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_top2(top1_id_buf, top1_value_buf, second_id_buf, second_value_buf)
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
                                     emit_head : Bool = true,
                                     top1_allowed_ids : Array(Int32)? = nil,
                                     lowrank_layer_indices : Set(Int32)? = nil,
                                     lowrank_state_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                     lowrank_basis_bufs : Hash(Int32, ML::MetalBuffer)? = nil,
                                     lowrank_rank : Int32 = 0) : Array(Float32)?
          if submission = forward_decode_wave_async(
               emb, layers,
               k_cache_bufs, v_cache_bufs, conv_state_bufs, ssm_state_bufs,
               output_norm, output_qw, hp, pos, top1: top1, emit_head: emit_head,
               top1_allowed_ids: top1_allowed_ids,
               lowrank_layer_indices: lowrank_layer_indices,
               lowrank_state_bufs: lowrank_state_bufs,
               lowrank_basis_bufs: lowrank_basis_bufs,
               lowrank_rank: lowrank_rank)
            wait_forward_decode_wave(submission)
          end
        end

        def self.forward_decode_wave_async(emb : Array(Float32)?,
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
                                           top2 : Bool = false,
                                           emit_head : Bool = true,
                                           top1_allowed_ids : Array(Int32)? = nil,
                                           fresh_scratch : Bool = false,
                                           scratch_namespace : String? = nil,
                                           retained_scratch : Array(ML::MetalBuffer)? = nil,
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
                                           token_embd_qw : QuantWeight? = nil,
                                           token_ids_buf : ML::MetalBuffer? = nil,
                                           token_index : Int32 = 0,
                                           top1_store_token_ids_buf : ML::MetalBuffer? = nil,
                                           top1_store_index : Int32 = -1,
                                           command_queue_name : String? = nil,
                                           append_command_buffer : ML::Metal::CommandBuffer? = nil) : DecodeWaveSubmission?
          # Two-lane callers can request fresh scratch so multiple submitted waves
          # do not race through the pooled temporary buffers before wait/readback.
          if fresh_scratch
            return Scratch.with_fresh do |retained|
              forward_decode_wave_async(
                emb, layers,
                k_cache_bufs, v_cache_bufs, conv_state_bufs, ssm_state_bufs,
                output_norm, output_qw, hp, pos,
                top1: top1, top2: top2, emit_head: emit_head, top1_allowed_ids: top1_allowed_ids,
                fresh_scratch: false, retained_scratch: retained,
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
                token_embd_qw: token_embd_qw,
                token_ids_buf: token_ids_buf,
                token_index: token_index,
                top1_store_token_ids_buf: top1_store_token_ids_buf,
                top1_store_index: top1_store_index,
                command_queue_name: command_queue_name,
                append_command_buffer: append_command_buffer)
            end
          end
          if namespace = scratch_namespace
            return Scratch.with_namespace(namespace) do
              forward_decode_wave_async(
                emb, layers,
                k_cache_bufs, v_cache_bufs, conv_state_bufs, ssm_state_bufs,
                output_norm, output_qw, hp, pos,
                top1: top1, top2: top2, emit_head: emit_head, top1_allowed_ids: top1_allowed_ids,
                scratch_namespace: nil, retained_scratch: retained_scratch,
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
                token_embd_qw: token_embd_qw,
                token_ids_buf: token_ids_buf,
                token_index: token_index,
                top1_store_token_ids_buf: top1_store_token_ids_buf,
                top1_store_index: top1_store_index,
                command_queue_name: command_queue_name,
                append_command_buffer: append_command_buffer)
            end
          end

          top1 = true if top2
          out_pipe = gemv_pipeline_for(output_qw)
          return nil if emit_head && out_pipe.nil?
          return nil if top1_store_token_ids_buf && (!emit_head || !top1)
          return nil if top1_store_token_ids_buf && top1_store_index < 0
          use_allowed_top1 = emit_head && top1 && !top2 && top1_allowed_ids && !top1_allowed_ids.not_nil!.empty? && output_qw.type.q6_k?
          allowed_ids = use_allowed_top1 ? top1_allowed_ids.not_nil! : nil
          if ids = allowed_ids
            ids.each do |id|
              return nil if id < 0 || id >= output_qw.out_dim
            end
          end

          ML::Metal::Device.init!

          hidden_dim = emb ? emb.not_nil!.size : hp.n_embd
          use_token_embedding = emb.nil?
          if use_token_embedding
            return nil unless token_embd = token_embd_qw
            return nil unless token_ids = token_ids_buf
            return nil unless token_embd.type.q4_k?
            return nil unless token_embd.in_dim == hidden_dim
          end
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
          allowed_n = allowed_ids.try(&.size) || 0
          tile_rows = use_allowed_top1 ? allowed_n : output_qw.out_dim
          tile_count = emit_head ? ((tile_rows + HEAD_TOP1_ROWS_PER_TG - 1) // HEAD_TOP1_ROWS_PER_TG) : 0
          allowed_ids_buf = use_allowed_top1 ? Scratch.get(:wave_top1_allowed_ids, allowed_n.to_i64 * sizeof(UInt32)) : nil
          top1_tile_values_buf = emit_head ? Scratch.get(:wave_top1_tile_values, tile_count.to_i64 * sizeof(Float32)) : nil
          top1_tile_ids_buf = emit_head ? Scratch.get(:wave_top1_tile_ids, tile_count.to_i64 * sizeof(UInt32)) : nil
          second_tile_values_buf = (emit_head && top2) ? Scratch.get(:wave_top2_tile_values, tile_count.to_i64 * sizeof(Float32)) : nil
          second_tile_ids_buf = (emit_head && top2) ? Scratch.get(:wave_top2_tile_ids, tile_count.to_i64 * sizeof(UInt32)) : nil
          top1_id_buf = emit_head ? Scratch.get(:wave_top1_id, sizeof(UInt32).to_i64) : nil
          top1_value_buf = emit_head ? Scratch.get(:wave_top1_value, sizeof(Float32).to_i64) : nil
          second_id_buf = (emit_head && top2) ? Scratch.get(:wave_second_id, sizeof(UInt32).to_i64) : nil
          second_value_buf = (emit_head && top2) ? Scratch.get(:wave_second_value, sizeof(Float32).to_i64) : nil

          # Full-attention scratch.
          qfull_buf = Scratch.get(:wave_qfull, (2 * q_dim).to_i64 * sizeof(Float32))
          q_buf = Scratch.get(:wave_q, q_dim.to_i64 * sizeof(Float32))
          gate_buf = Scratch.get(:wave_gate, q_dim.to_i64 * sizeof(Float32))
          k_buf = Scratch.get(:wave_k, kv_dim.to_i64 * sizeof(Float32))
          v_buf = Scratch.get(:wave_v, kv_dim.to_i64 * sizeof(Float32))
          attn_buf = Scratch.get(:wave_attn, q_dim.to_i64 * sizeof(Float32))
          attn_out_buf = Scratch.get(:wave_attn_out, hidden_dim.to_i64 * sizeof(Float32))
          splitk_chunk = attn_splitk_chunk_size
          splitk_blocks = ((pos + 1) + splitk_chunk - 1) // splitk_chunk
          splitk_partial_o_buf = Scratch.get(:wave_attn_splitk_o, (hp.n_head * splitk_blocks * hp.head_dim).to_i64 * sizeof(Float32))
          splitk_partial_m_buf = Scratch.get(:wave_attn_splitk_m, (hp.n_head * splitk_blocks).to_i64 * sizeof(Float32))
          splitk_partial_l_buf = Scratch.get(:wave_attn_splitk_l, (hp.n_head * splitk_blocks).to_i64 * sizeof(Float32))

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
          zero_hidden_buf = (lowrank_skip_ffn || skip_recurrent_ffn || !lowrank_skip_ffn_layer_indices.nil?) ? Scratch.get(:wave_zero_hidden, hidden_dim.to_i64 * sizeof(Float32)) : nil

          src_buf.write(emb.not_nil!) if emb
          if zh = zero_hidden_buf
            ConstCache.write_zero_f32_once("wave_zero_hidden", zh, hidden_dim)
          end
          if emit_head
            ConstCache.write_once("wave_output_norm", output_norm_buf.not_nil!, output_norm)
            if ids = allowed_ids
              ptr = allowed_ids_buf.not_nil!.contents.as(Pointer(UInt32))
              ids.each_with_index { |id, i| ptr[i] = id.to_u32 }
            end
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
          chunk_layers = append_command_buffer ? 0 : wave_chunk_layers
          pending_cmds = [] of ML::Metal::CommandBuffer

          lr_set = lowrank_layer_indices
          lr_states = lowrank_state_bufs
          lr_bases = lowrank_basis_bufs
          lr_active = !lr_set.nil? && !lr_set.empty? && lowrank_rank > 0 && !lr_states.nil? && !lr_bases.nil?
          lr_updown_active = lowrank_updown_rank > 0
          if lr_updown_active
            raise "decode wave lowrank updown requires active lowrank layers" unless lr_active
            raise "decode wave lowrank updown rank too large" if lowrank_updown_rank > 64
            raise "decode wave lowrank updown requires x_mean buffers" if lowrank_updown_x_mean_bufs.nil?
            raise "decode wave lowrank updown requires c_mean buffers" if lowrank_updown_c_mean_bufs.nil?
            q8_updown = !lowrank_updown_coeff_q8_bufs.nil? &&
                         !lowrank_updown_coeff_q8_scale_bufs.nil? &&
                         !lowrank_updown_down_q8_bufs.nil? &&
                         !lowrank_updown_down_q8_scale_bufs.nil?
            f32_updown = !lowrank_updown_coeff_w_bufs.nil? && !lowrank_updown_down_bufs.nil?
            raise "decode wave lowrank updown requires either f32 or q8 coeff/down buffers" unless q8_updown || f32_updown
          end
          lr_c_buf = lr_active ? Scratch.get(:wave_lr_c, (hp.ssm_group_count * lowrank_rank).to_i64 * sizeof(Float32)) : nil
          lr_qbar_buf = lr_active ? Scratch.get(:wave_lr_qbar, (hp.ssm_group_count * lowrank_rank).to_i64 * sizeof(Float32)) : nil

          cmd_queue = command_queue_name ? lane_command_queue(command_queue_name.not_nil!) : nil
          t0 = Time.instant if Profile.enabled?
          cmd = append_command_buffer || ML::Metal::CommandBuffer.new(queue: cmd_queue, fast: wave_fast_command_buffer_enabled?)

          if use_token_embedding
            emb_qw = token_embd_qw.not_nil!
            emb_w_buf, emb_w_off = weight_slot(emb_qw)
            emb_enc = ML::Metal::ComputeEncoder.new(cmd)
            encode_embedding_q4k_from_token_id(emb_enc, emb_w_buf, emb_w_off, token_ids_buf.not_nil!, src_buf,
              hidden_dim, emb_qw.out_dim, token_index)
            emb_enc.end_encoding
          end

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

                use_splitk_attn = ENV["QWEN35_ATTN_SPLITK_OFF"]? != "1" &&
                                  pos + 1 >= attn_splitk_min_context &&
                                  hp.head_dim <= 256
                if use_splitk_attn
                  split1_enc = ML::Metal::ComputeEncoder.new(cmd)
                  split1_enc.set_pipeline(attn_splitk_stage1_pipeline)
                  split1_enc.set_buffer(q_buf, 0)
                  split1_enc.set_buffer(k_cache_buf, 1)
                  split1_enc.set_buffer(v_cache_buf, 2)
                  split1_enc.set_buffer(splitk_partial_o_buf, 3, ML::Metal::BufferAccess::Write)
                  split1_enc.set_buffer(splitk_partial_m_buf, 4, ML::Metal::BufferAccess::Write)
                  split1_enc.set_buffer(splitk_partial_l_buf, 5, ML::Metal::BufferAccess::Write)
                  split1_enc.set_value((pos + 1).to_u32, 6)
                  split1_enc.set_value(hp.n_head.to_u32, 7)
                  split1_enc.set_value(hp.n_head_kv.to_u32, 8)
                  split1_enc.set_value(hp.head_dim.to_u32, 9)
                  split1_enc.set_value((hp.n_head // hp.n_head_kv).to_u32, 10)
                  split1_enc.set_value(attn_scale, 11)
                  split1_enc.set_value(splitk_chunk.to_u32, 12)
                  split1_enc.set_value(splitk_blocks.to_u32, 13)
                  split1_enc.dispatch_threadgroups({hp.n_head, splitk_blocks, 1}, {32, 1, 1})
                  split1_enc.end_encoding

                  split2_enc = ML::Metal::ComputeEncoder.new(cmd)
                  split2_enc.set_pipeline(attn_splitk_stage2_pipeline)
                  split2_enc.set_buffer(gate_buf, 0)
                  split2_enc.set_buffer(splitk_partial_o_buf, 1)
                  split2_enc.set_buffer(splitk_partial_m_buf, 2)
                  split2_enc.set_buffer(splitk_partial_l_buf, 3)
                  split2_enc.set_buffer(attn_buf, 4, ML::Metal::BufferAccess::Write)
                  split2_enc.set_value(hp.n_head.to_u32, 5)
                  split2_enc.set_value(hp.head_dim.to_u32, 6)
                  split2_enc.set_value(splitk_blocks.to_u32, 7)
                  split2_enc.dispatch_threadgroups({hp.n_head, 1, 1}, {32, 1, 1})
                  split2_enc.end_encoding
                else
                  attn_enc = ML::Metal::ComputeEncoder.new(cmd)
                  use_gqa4_attn = hp.n_head // hp.n_head_kv == 4 && hp.head_dim <= 128 && attn_gqa4_enabled?
                  attn_enc.set_pipeline(use_gqa4_attn ? attn_gqa4_pipeline : attn_pipeline)
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
                  if use_gqa4_attn
                    attn_enc.dispatch_threadgroups({hp.n_head_kv, 1, 1}, {128, 1, 1})
                  else
                    attn_enc.dispatch_threadgroups({hp.n_head, 1, 1}, {32, 1, 1})
                  end
                  attn_enc.end_encoding
                end
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

                use_lr = lr_active && lr_set.try(&.includes?(il)) == true

                Profile.trace("rec.dn") do
                  if use_lr
                    lr_state_buf = lr_states.not_nil![il]
                    lr_basis_buf = lr_bases.not_nil![il]

                    proj_enc = ML::Metal::ComputeEncoder.new(cmd)
                    proj_enc.set_pipeline(lowrank_project_coeffs_pipeline)
                    proj_enc.set_buffer(rec_q_buf, 0)
                    proj_enc.set_buffer(rec_k_buf, 1)
                    proj_enc.set_buffer(lr_basis_buf, 2)
                    proj_enc.set_buffer(lr_c_buf.not_nil!, 3, ML::Metal::BufferAccess::Write)
                    proj_enc.set_buffer(lr_qbar_buf.not_nil!, 4, ML::Metal::BufferAccess::Write)
                    proj_enc.set_value(hp.ssm_group_count.to_u32, 5)
                    proj_enc.set_value(hp.ssm_state_size.to_u32, 6)
                    proj_enc.set_value(lowrank_rank.to_u32, 7)
                    proj_enc.dispatch_threadgroups({(lowrank_rank + 7) // 8, hp.ssm_group_count, 2}, {8, 1, 1})
                    proj_enc.end_encoding

                    step_enc = ML::Metal::ComputeEncoder.new(cmd)
                    step_enc.set_pipeline(lowrank_delta_pipeline)
                    step_enc.set_buffer(lr_state_buf, 0, ML::Metal::BufferAccess::ReadWrite)
                    step_enc.set_buffer(lr_c_buf.not_nil!, 1)
                    step_enc.set_buffer(lr_qbar_buf.not_nil!, 2)
                    step_enc.set_buffer(rec_v_buf, 3)
                    step_enc.set_buffer(g_buf, 4)
                    step_enc.set_buffer(beta_buf, 5)
                    step_enc.set_buffer(rec_mid_buf, 6, ML::Metal::BufferAccess::Write)
                    step_enc.set_value(hp.ssm_group_count.to_u32, 7)
                    step_enc.set_value(hp.ssm_time_step_rank.to_u32, 8)
                    step_enc.set_value(hp.ssm_state_size.to_u32, 9)
                    step_enc.set_value(lowrank_rank.to_u32, 10)
                    step_enc.set_value(rec_scale, 11)
                    step_enc.dispatch_threadgroups({hp.ssm_state_size, hp.ssm_time_step_rank, 1}, {1, 1, 1})
                    step_enc.end_encoding

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
                  else
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

                use_layer_skip_ffn = use_lr && (lowrank_skip_ffn || lowrank_skip_ffn_layer_indices.try(&.includes?(il)) == true)
                use_layer_updown = use_lr && lr_updown_active && (lowrank_updown_layer_indices.nil? || lowrank_updown_layer_indices.not_nil!.includes?(il))

                if skip_recurrent_ffn || use_layer_skip_ffn
                  Profile.trace("rec.ffn_skip") do
                    copy_enc = ML::Metal::ComputeEncoder.new(cmd)
                    copy_enc.set_pipeline(add_vec_pipeline)
                    copy_enc.set_buffer(residual_buf, 0)
                    copy_enc.set_buffer(zero_hidden_buf.not_nil!, 1)
                    copy_enc.set_buffer(dst_buf, 2, ML::Metal::BufferAccess::Write)
                    copy_enc.set_value(hidden_dim.to_u32, 3)
                    copy_enc.dispatch_1d(hidden_dim, 256)
                    copy_enc.end_encoding
                  end
                elsif use_layer_updown
                  x_mean_buf = lowrank_updown_x_mean_bufs.not_nil![il]? || raise "decode wave lowrank updown missing x_mean for layer #{il}"
                  c_mean_buf = lowrank_updown_c_mean_bufs.not_nil![il]? || raise "decode wave lowrank updown missing c_mean for layer #{il}"
                  use_q8_updown = !lowrank_updown_coeff_q8_bufs.nil? &&
                                   !lowrank_updown_coeff_q8_scale_bufs.nil? &&
                                   !lowrank_updown_down_q8_bufs.nil? &&
                                   !lowrank_updown_down_q8_scale_bufs.nil? &&
                                   lowrank_updown_coeff_q8_bufs.not_nil!.has_key?(il) &&
                                   lowrank_updown_down_q8_bufs.not_nil!.has_key?(il)

                  Profile.trace("rec.ffn_pca_updown") do
                    updown_enc = ML::Metal::ComputeEncoder.new(cmd)
                    if use_q8_updown
                      coeff_q8_buf = lowrank_updown_coeff_q8_bufs.not_nil![il]? || raise "decode wave lowrank updown missing q8 coeff for layer #{il}"
                      coeff_q8_scale_buf = lowrank_updown_coeff_q8_scale_bufs.not_nil![il]? || raise "decode wave lowrank updown missing q8 coeff scales for layer #{il}"
                      down_q8_buf = lowrank_updown_down_q8_bufs.not_nil![il]? || raise "decode wave lowrank updown missing q8 down for layer #{il}"
                      down_q8_scale_buf = lowrank_updown_down_q8_scale_bufs.not_nil![il]? || raise "decode wave lowrank updown missing q8 down scales for layer #{il}"
                      updown_enc.set_pipeline(ffn_pca_updown_fused_rows_q8_pipeline)
                      updown_enc.set_buffer(pre_norm_buf, 0)
                      updown_enc.set_buffer(x_mean_buf, 1)
                      updown_enc.set_buffer(c_mean_buf, 2)
                      updown_enc.set_buffer(coeff_q8_buf, 3)
                      updown_enc.set_buffer(coeff_q8_scale_buf, 4)
                      updown_enc.set_buffer(down_q8_buf, 5)
                      updown_enc.set_buffer(down_q8_scale_buf, 6)
                      updown_enc.set_buffer(ffn_out_buf, 7, ML::Metal::BufferAccess::Write)
                      updown_enc.set_value(hidden_dim.to_u32, 8)
                      updown_enc.set_value(lowrank_updown_rank.to_u32, 9)
                      updown_enc.set_value(1_u32, 10)
                    else
                      coeff_w_buf = lowrank_updown_coeff_w_bufs.not_nil![il]? || raise "decode wave lowrank updown missing coeff_w for layer #{il}"
                      down_buf = lowrank_updown_down_bufs.not_nil![il]? || raise "decode wave lowrank updown missing down for layer #{il}"
                      updown_enc.set_pipeline(ffn_pca_updown_fused_rows_pipeline)
                      updown_enc.set_buffer(pre_norm_buf, 0)
                      updown_enc.set_buffer(x_mean_buf, 1)
                      updown_enc.set_buffer(c_mean_buf, 2)
                      updown_enc.set_buffer(coeff_w_buf, 3)
                      updown_enc.set_buffer(down_buf, 4)
                      updown_enc.set_buffer(ffn_out_buf, 5, ML::Metal::BufferAccess::Write)
                      updown_enc.set_value(hidden_dim.to_u32, 6)
                      updown_enc.set_value(lowrank_updown_rank.to_u32, 7)
                      updown_enc.set_value(1_u32, 8)
                    end
                    updown_enc.dispatch_threadgroups({1, 1, 1}, {256, 1, 1})
                    updown_enc.end_encoding
                  end

                  Profile.trace("rec.ffn_pca_updown_add") do
                    add_enc = ML::Metal::ComputeEncoder.new(cmd)
                    add_enc.set_pipeline(add_vec_pipeline)
                    add_enc.set_buffer(residual_buf, 0)
                    add_enc.set_buffer(ffn_out_buf, 1)
                    add_enc.set_buffer(dst_buf, 2, ML::Metal::BufferAccess::Write)
                    add_enc.set_value(hidden_dim.to_u32, 3)
                    add_enc.dispatch_1d(hidden_dim, 256)
                    add_enc.end_encoding
                  end
                else
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
            end

            src_buf, dst_buf = dst_buf, src_buf
            if chunk_layers > 0 && il + 1 < layers.size && (il + 1) % chunk_layers == 0
              cmd.commit
              pending_cmds << cmd
              cmd = ML::Metal::CommandBuffer.new(queue: cmd_queue, fast: wave_fast_command_buffer_enabled?)
            end
          end

          use_head_top1 = false
          use_head_top2 = false
          if emit_head
            Profile.trace("head") do
              Profile.trace("head.norm") do
                head_norm_enc = ML::Metal::ComputeEncoder.new(cmd)
                encode_rmsnorm_vec(head_norm_enc, src_buf, output_norm_buf.not_nil!, pre_norm_buf, hidden_dim, hp.rms_eps)
                head_norm_enc.end_encoding
              end

              use_head_top1 = top1 && can_use_head_top1_fused?(output_qw)
              use_head_top2 = top2 && can_use_head_top1_fused?(output_qw)
              if use_head_top1
                Profile.trace(use_allowed_top1 ? "head.top1_allowed" : (use_head_top2 ? "head.top2" : "head.top1")) do
                  head_top1_enc = ML::Metal::ComputeEncoder.new(cmd)
                  head_top1_enc.set_pipeline(
                    if use_allowed_top1
                      mv6_top1_allowed_tiles_pipeline
                    elsif use_head_top2
                      output_qw.type.q8_0? ? mv8_top2_tiles_pipeline : mv6_top2_tiles_pipeline
                    else
                      output_qw.type.q8_0? ? mv8_top1_tiles_pipeline : mv6_top1_tiles_pipeline
                    end
                  )
                  if use_allowed_top1
                    profile_bump_head_top1_shape("head_top1_allowed#{allowed_n}", output_qw, rows: allowed_n)
                  else
                    profile_bump_head_top1_shape(use_head_top2 ? "head_top2" : "head_top1", output_qw)
                  end
                  head_top1_enc.set_buffer(out_w_buf.not_nil!, 0, ML::Metal::BufferAccess::Read, offset: out_w_off)
                  head_top1_enc.set_buffer(pre_norm_buf, 1)
                  if use_allowed_top1
                    head_top1_enc.set_buffer(allowed_ids_buf.not_nil!, 2)
                    head_top1_enc.set_buffer(top1_tile_values_buf.not_nil!, 3, ML::Metal::BufferAccess::Write)
                    head_top1_enc.set_buffer(top1_tile_ids_buf.not_nil!, 4, ML::Metal::BufferAccess::Write)
                    head_top1_enc.set_value(output_qw.in_dim.to_u32, 5)
                    head_top1_enc.set_value(output_qw.out_dim.to_u32, 6)
                    head_top1_enc.set_value(allowed_n.to_u32, 7)
                  else
                    head_top1_enc.set_buffer(top1_tile_values_buf.not_nil!, 2, ML::Metal::BufferAccess::Write)
                    head_top1_enc.set_buffer(top1_tile_ids_buf.not_nil!, 3, ML::Metal::BufferAccess::Write)
                  end
                  if use_head_top2
                    head_top1_enc.set_buffer(second_tile_values_buf.not_nil!, 4, ML::Metal::BufferAccess::Write)
                    head_top1_enc.set_buffer(second_tile_ids_buf.not_nil!, 5, ML::Metal::BufferAccess::Write)
                    head_top1_enc.set_value(output_qw.in_dim.to_u32, 6)
                    head_top1_enc.set_value(output_qw.out_dim.to_u32, 7)
                  elsif !use_allowed_top1
                    head_top1_enc.set_value(output_qw.in_dim.to_u32, 4)
                    head_top1_enc.set_value(output_qw.out_dim.to_u32, 5)
                  end
                  head_top1_enc.dispatch_threadgroups({tile_count, 1, 1}, {output_qw.type.q8_0? ? MV_Q8_NSG * 32 : 64, 1, 1})
                  head_top1_enc.end_encoding

                  reduce_top1_enc = ML::Metal::ComputeEncoder.new(cmd)
                  reduce_top1_enc.set_pipeline(use_head_top2 ? top2_reduce_tiles_pipeline : top1_reduce_tiles_pipeline)
                  reduce_top1_enc.set_buffer(top1_tile_values_buf.not_nil!, 0)
                  reduce_top1_enc.set_buffer(top1_tile_ids_buf.not_nil!, 1)
                  if use_head_top2
                    reduce_top1_enc.set_buffer(second_tile_values_buf.not_nil!, 2)
                    reduce_top1_enc.set_buffer(second_tile_ids_buf.not_nil!, 3)
                    reduce_top1_enc.set_buffer(top1_id_buf.not_nil!, 4, ML::Metal::BufferAccess::Write)
                    reduce_top1_enc.set_buffer(top1_value_buf.not_nil!, 5, ML::Metal::BufferAccess::Write)
                    reduce_top1_enc.set_buffer(second_id_buf.not_nil!, 6, ML::Metal::BufferAccess::Write)
                    reduce_top1_enc.set_buffer(second_value_buf.not_nil!, 7, ML::Metal::BufferAccess::Write)
                    reduce_top1_enc.set_value(tile_count.to_u32, 8)
                  else
                    reduce_top1_enc.set_buffer(top1_id_buf.not_nil!, 2, ML::Metal::BufferAccess::Write)
                    reduce_top1_enc.set_buffer(top1_value_buf.not_nil!, 3, ML::Metal::BufferAccess::Write)
                    reduce_top1_enc.set_value(tile_count.to_u32, 4)
                  end
                  reduce_top1_enc.dispatch_threadgroups({1, 1, 1}, {256, 1, 1})
                  reduce_top1_enc.end_encoding
                end

                if store_buf = top1_store_token_ids_buf
                  Profile.trace("head.store_top1_token") do
                    store_enc = ML::Metal::ComputeEncoder.new(cmd)
                    store_enc.set_pipeline(store_top1_token_id_pipeline)
                    store_enc.set_buffer(top1_id_buf.not_nil!, 0)
                    store_enc.set_buffer(store_buf, 1, ML::Metal::BufferAccess::Write)
                    store_enc.set_value(top1_store_index.to_u32, 2)
                    store_enc.dispatch_threadgroups({1, 1, 1}, {1, 1, 1})
                    store_enc.end_encoding
                  end
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
          cmd.commit unless append_command_buffer
          DecodeWaveSubmission.new(
            cmd, pending_cmds, emit_head, use_head_top1, use_head_top2,
            logits_buf, top1_id_buf, top1_value_buf, second_id_buf, second_value_buf, output_qw.out_dim,
            retained_scratch || [] of ML::MetalBuffer, t0, t_enc)
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

        # BF16 MTP sidecar tensors live in a separate safetensors mmap, outside
        # the registered GGUF mmap. Cache one MetalBuffer per raw slice so the
        # correctness probe does not re-upload 100-170 MiB weights per matvec.
        private def self.bf16_weight_slot(w_raw : Bytes) : {ML::MetalBuffer, Int64}
          if slot = mmap_slot_for(w_raw)
            return slot
          end

          key = "#{w_raw.to_unsafe.address}:#{w_raw.size}"
          @@bf16_weight_mutex.synchronize do
            if buf = @@bf16_weight_buffers[key]?
              {buf, 0_i64}
            else
              buf = upload_weights(w_raw)
              @@bf16_weight_buffers[key] = buf
              {buf, 0_i64}
            end
          end
        end

        def self.clear_bf16_weight_cache : Nil
          @@bf16_weight_mutex.synchronize do
            @@bf16_weight_buffers.clear
          end
        end

        private def self.encode_bf16_gemv(enc : ML::Metal::ComputeEncoder,
                                          w_buf : ML::MetalBuffer,
                                          w_off : Int64,
                                          x_buf : ML::MetalBuffer,
                                          out_buf : ML::MetalBuffer,
                                          in_dim : Int32,
                                          out_dim : Int32) : Nil
          enc.set_pipeline(bf16_gemv_pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_off)
          enc.set_buffer(x_buf, 1)
          enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32, 3)
          enc.set_value(out_dim.to_u32, 4)
          rows_per_tg = 2
          enc.dispatch_threadgroups({(out_dim + rows_per_tg - 1) // rows_per_tg, 1, 1}, {64, 1, 1})
        end

        private def self.encode_bf16_q_gate_gemv(enc : ML::Metal::ComputeEncoder,
                                                 w_buf : ML::MetalBuffer,
                                                 w_off : Int64,
                                                 x_buf : ML::MetalBuffer,
                                                 out_buf : ML::MetalBuffer,
                                                 in_dim : Int32,
                                                 q_dim : Int32,
                                                 head_dim : Int32) : Nil
          enc.set_pipeline(bf16_q_gate_gemv_pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_off)
          enc.set_buffer(x_buf, 1)
          enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32, 3)
          enc.set_value(q_dim.to_u32, 4)
          enc.set_value(head_dim.to_u32, 5)
          rows_per_tg = 2
          enc.dispatch_threadgroups({(q_dim + rows_per_tg - 1) // rows_per_tg, 1, 1}, {64, 1, 1})
        end

        def self.bf16_gemv(w_raw : Bytes,
                           in_dim : Int32,
                           out_dim : Int32,
                           x : Array(Float32)) : Array(Float32)
          raise "bf16_gemv input mismatch: expected #{in_dim}, got #{x.size}" unless x.size == in_dim
          expected_w = out_dim.to_i64 * in_dim.to_i64 * 2_i64
          raise "bf16_gemv weight size mismatch: expected #{expected_w}, got #{w_raw.size}" unless w_raw.size.to_i64 == expected_w

          ML::Metal::Device.init!
          w_buf, w_off = bf16_weight_slot(w_raw)
          x_buf = Scratch.get(:mtp_bf16_x, in_dim.to_i64 * sizeof(Float32))
          out_buf = Scratch.get("mtp_bf16_out_#{out_dim}", out_dim.to_i64 * sizeof(Float32))
          x_buf.write(x)

          Profile.bump_matmul_shape("gemv BF16 #{in_dim}x#{out_dim} b1", expected_w)
          t0 = Time.instant if Profile.enabled?
          cmd = ML::Metal::CommandBuffer.new
          enc = ML::Metal::ComputeEncoder.new(cmd)
          enc.set_pipeline(bf16_gemv_pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_off)
          enc.set_buffer(x_buf, 1)
          enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32, 3)
          enc.set_value(out_dim.to_u32, 4)
          rows_per_tg = 2
          enc.dispatch_threadgroups({(out_dim + rows_per_tg - 1) // rows_per_tg, 1, 1}, {64, 1, 1})
          enc.end_encoding
          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_f32(out_buf, out_dim)
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

        def self.bf16_q_gate_gemv(w_raw : Bytes,
                                  in_dim : Int32,
                                  q_dim : Int32,
                                  head_dim : Int32,
                                  x : Array(Float32)) : Array(Float32)
          raise "bf16_q_gate_gemv input mismatch: expected #{in_dim}, got #{x.size}" unless x.size == in_dim
          raise "bf16_q_gate_gemv q_dim must be divisible by head_dim" unless head_dim > 0 && q_dim % head_dim == 0
          expected_w = q_dim.to_i64 * 2_i64 * in_dim.to_i64 * 2_i64
          raise "bf16_q_gate_gemv weight size mismatch: expected #{expected_w}, got #{w_raw.size}" unless w_raw.size.to_i64 == expected_w

          ML::Metal::Device.init!
          w_buf, w_off = bf16_weight_slot(w_raw)
          x_buf = Scratch.get(:mtp_bf16_x, in_dim.to_i64 * sizeof(Float32))
          out_buf = Scratch.get("mtp_bf16_q_gate_out_#{q_dim}", q_dim.to_i64 * sizeof(Float32))
          x_buf.write(x)

          Profile.bump_matmul_shape("gemv BF16 q_gate #{in_dim}x#{q_dim} b1", expected_w // 2_i64)
          t0 = Time.instant if Profile.enabled?
          cmd = ML::Metal::CommandBuffer.new
          enc = ML::Metal::ComputeEncoder.new(cmd)
          enc.set_pipeline(bf16_q_gate_gemv_pipeline)
          enc.set_buffer(w_buf, 0, ML::Metal::BufferAccess::Read, offset: w_off)
          enc.set_buffer(x_buf, 1)
          enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_value(in_dim.to_u32, 3)
          enc.set_value(q_dim.to_u32, 4)
          enc.set_value(head_dim.to_u32, 5)
          rows_per_tg = 2
          enc.dispatch_threadgroups({(q_dim + rows_per_tg - 1) // rows_per_tg, 1, 1}, {64, 1, 1})
          enc.end_encoding
          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_f32(out_buf, q_dim)
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

        def self.mtp_one_token_hidden_from_fc_in(fc_in : Array(Float32),
                                                 fc_raw : Bytes,
                                                 v_raw : Bytes,
                                                 q_raw : Bytes,
                                                 o_raw : Bytes,
                                                 ffn_gate_raw : Bytes,
                                                 ffn_up_raw : Bytes,
                                                 ffn_down_raw : Bytes,
                                                 input_norm : Array(Float32),
                                                 post_norm : Array(Float32),
                                                 final_norm : Array(Float32),
                                                 hidden_dim : Int32,
                                                 q_dim : Int32,
                                                 kv_dim : Int32,
                                                 ffn_dim : Int32,
                                                 n_head : Int32,
                                                 n_head_kv : Int32,
                                                 head_dim : Int32,
                                                 eps : Float32) : Array(Float32)?
          raise "mtp_one_token_hidden fc_in mismatch" unless fc_in.size == hidden_dim * 2
          raise "mtp_one_token_hidden norm size mismatch" unless input_norm.size == hidden_dim && post_norm.size == hidden_dim && final_norm.size == hidden_dim
          raise "mtp_one_token_hidden q_dim mismatch" unless q_dim == n_head * head_dim
          raise "mtp_one_token_hidden kv_dim mismatch" unless kv_dim == n_head_kv * head_dim
          raise "mtp_one_token_hidden n_head must be divisible by n_head_kv" unless n_head_kv > 0 && n_head % n_head_kv == 0

          expected_fc = hidden_dim.to_i64 * (hidden_dim * 2).to_i64 * 2_i64
          expected_q = (q_dim * 2).to_i64 * hidden_dim.to_i64 * 2_i64
          expected_v = kv_dim.to_i64 * hidden_dim.to_i64 * 2_i64
          expected_o = hidden_dim.to_i64 * q_dim.to_i64 * 2_i64
          expected_ffn = ffn_dim.to_i64 * hidden_dim.to_i64 * 2_i64
          expected_down = hidden_dim.to_i64 * ffn_dim.to_i64 * 2_i64
          raise "mtp fc weight size mismatch" unless fc_raw.size.to_i64 == expected_fc
          raise "mtp q weight size mismatch" unless q_raw.size.to_i64 == expected_q
          raise "mtp v weight size mismatch" unless v_raw.size.to_i64 == expected_v
          raise "mtp o weight size mismatch" unless o_raw.size.to_i64 == expected_o
          raise "mtp ffn gate weight size mismatch" unless ffn_gate_raw.size.to_i64 == expected_ffn
          raise "mtp ffn up weight size mismatch" unless ffn_up_raw.size.to_i64 == expected_ffn
          raise "mtp ffn down weight size mismatch" unless ffn_down_raw.size.to_i64 == expected_down

          ML::Metal::Device.init!

          fc_w, fc_off = bf16_weight_slot(fc_raw)
          v_w, v_off = bf16_weight_slot(v_raw)
          q_w, q_off = bf16_weight_slot(q_raw)
          o_w, o_off = bf16_weight_slot(o_raw)
          ffn_gate_w, ffn_gate_off = bf16_weight_slot(ffn_gate_raw)
          ffn_up_w, ffn_up_off = bf16_weight_slot(ffn_up_raw)
          ffn_down_w, ffn_down_off = bf16_weight_slot(ffn_down_raw)

          hidden_bytes = hidden_dim.to_i64 * sizeof(Float32)
          q_bytes = q_dim.to_i64 * sizeof(Float32)
          kv_bytes = kv_dim.to_i64 * sizeof(Float32)
          ffn_bytes = ffn_dim.to_i64 * sizeof(Float32)

          fc_in_buf = Scratch.get(:mtp_body_fc_in, fc_in.size.to_i64 * sizeof(Float32))
          residual_buf = Scratch.get(:mtp_body_residual, hidden_bytes)
          input_norm_buf = Scratch.get(:mtp_body_input_norm_w, hidden_bytes)
          post_norm_buf = Scratch.get(:mtp_body_post_norm_w, hidden_bytes)
          final_norm_buf = Scratch.get(:mtp_body_final_norm_w, hidden_bytes)
          cur_buf = Scratch.get(:mtp_body_cur, hidden_bytes)
          v_buf = Scratch.get(:mtp_body_v, kv_bytes)
          gate_buf = Scratch.get(:mtp_body_gate, q_bytes)
          attn_o_buf = Scratch.get(:mtp_body_attn_o, q_bytes)
          attn_out_buf = Scratch.get(:mtp_body_attn_out, hidden_bytes)
          after_attn_buf = Scratch.get(:mtp_body_after_attn, hidden_bytes)
          cur2_buf = Scratch.get(:mtp_body_cur2, hidden_bytes)
          ffn_gate_buf = Scratch.get(:mtp_body_ffn_gate, ffn_bytes)
          ffn_up_buf = Scratch.get(:mtp_body_ffn_up, ffn_bytes)
          ffn_act_buf = Scratch.get(:mtp_body_ffn_act, ffn_bytes)
          ffn_out_buf = Scratch.get(:mtp_body_ffn_out, hidden_bytes)
          after_ffn_buf = Scratch.get(:mtp_body_after_ffn, hidden_bytes)
          final_buf = Scratch.get(:mtp_body_final, hidden_bytes)

          fc_in_buf.write(fc_in)
          input_norm_buf.write(input_norm)
          post_norm_buf.write(post_norm)
          final_norm_buf.write(final_norm)

          Profile.bump_matmul_shape("mtp_body gemv BF16 fc #{hidden_dim * 2}x#{hidden_dim} b1", expected_fc)
          Profile.bump_matmul_shape("mtp_body gemv BF16 v #{hidden_dim}x#{kv_dim} b1", expected_v)
          Profile.bump_matmul_shape("mtp_body gemv BF16 q_gate #{hidden_dim}x#{q_dim} b1", expected_q // 2_i64)
          Profile.bump_matmul_shape("mtp_body gemv BF16 o #{q_dim}x#{hidden_dim} b1", expected_o)
          Profile.bump_matmul_shape("mtp_body gemv BF16 ffn_gate #{hidden_dim}x#{ffn_dim} b1", expected_ffn)
          Profile.bump_matmul_shape("mtp_body gemv BF16 ffn_up #{hidden_dim}x#{ffn_dim} b1", expected_ffn)
          Profile.bump_matmul_shape("mtp_body gemv BF16 ffn_down #{ffn_dim}x#{hidden_dim} b1", expected_down)

          t0 = Time.instant if Profile.enabled?
          cmd = ML::Metal::CommandBuffer.new

          fc_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_bf16_gemv(fc_enc, fc_w, fc_off, fc_in_buf, residual_buf, hidden_dim * 2, hidden_dim)
          fc_enc.end_encoding

          norm_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_rmsnorm_vec(norm_enc, residual_buf, input_norm_buf, cur_buf, hidden_dim, eps)
          norm_enc.end_encoding

          v_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_bf16_gemv(v_enc, v_w, v_off, cur_buf, v_buf, hidden_dim, kv_dim)
          v_enc.end_encoding

          gate_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_bf16_q_gate_gemv(gate_enc, q_w, q_off, cur_buf, gate_buf, hidden_dim, q_dim, head_dim)
          gate_enc.end_encoding

          attn_gate_enc = ML::Metal::ComputeEncoder.new(cmd)
          attn_gate_enc.set_pipeline(mtp_attn_gate_pipeline)
          attn_gate_enc.set_buffer(v_buf, 0)
          attn_gate_enc.set_buffer(gate_buf, 1)
          attn_gate_enc.set_buffer(attn_o_buf, 2, ML::Metal::BufferAccess::Write)
          attn_gate_enc.set_value(head_dim.to_u32, 3)
          attn_gate_enc.set_value(n_head.to_u32, 4)
          attn_gate_enc.set_value(n_head_kv.to_u32, 5)
          attn_gate_enc.dispatch_1d(q_dim, 256)
          attn_gate_enc.end_encoding

          o_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_bf16_gemv(o_enc, o_w, o_off, attn_o_buf, attn_out_buf, q_dim, hidden_dim)
          o_enc.end_encoding

          post_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_add_rmsnorm(post_enc, residual_buf, attn_out_buf, post_norm_buf, after_attn_buf, cur2_buf, hidden_dim, eps)
          post_enc.end_encoding

          ffn_gate_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_bf16_gemv(ffn_gate_enc, ffn_gate_w, ffn_gate_off, cur2_buf, ffn_gate_buf, hidden_dim, ffn_dim)
          ffn_gate_enc.end_encoding

          ffn_up_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_bf16_gemv(ffn_up_enc, ffn_up_w, ffn_up_off, cur2_buf, ffn_up_buf, hidden_dim, ffn_dim)
          ffn_up_enc.end_encoding

          swiglu_enc = ML::Metal::ComputeEncoder.new(cmd)
          swiglu_enc.set_pipeline(ffn_swiglu_pipeline)
          swiglu_enc.set_buffer(ffn_gate_buf, 0)
          swiglu_enc.set_buffer(ffn_up_buf, 1)
          swiglu_enc.set_buffer(ffn_act_buf, 2, ML::Metal::BufferAccess::Write)
          swiglu_enc.set_value(ffn_dim.to_u32, 3)
          swiglu_enc.dispatch_1d(ffn_dim, 256)
          swiglu_enc.end_encoding

          ffn_down_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_bf16_gemv(ffn_down_enc, ffn_down_w, ffn_down_off, ffn_act_buf, ffn_out_buf, ffn_dim, hidden_dim)
          ffn_down_enc.end_encoding

          final_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_add_rmsnorm(final_enc, after_attn_buf, ffn_out_buf, final_norm_buf, after_ffn_buf, final_buf, hidden_dim, eps)
          final_enc.end_encoding

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_f32(final_buf, hidden_dim)
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

        def self.matmul_iq4_nl(x : Array(Float32),
                               w_raw : Bytes,
                               in_dim : Int32,
                               out_dim : Int32,
                               batch : Int32) : Array(Float32)
          raise "in_dim must be multiple of #{IQ4_NL_QK}: got #{in_dim}" unless in_dim % IQ4_NL_QK == 0
          raise "x size mismatch: expected #{batch * in_dim}, got #{x.size}" unless x.size == batch * in_dim
          expected_w = (in_dim // IQ4_NL_QK) * IQ4_NL_BLOCK_BYTES * out_dim
          raise "w_raw size mismatch: expected #{expected_w}, got #{w_raw.size}" unless w_raw.size == expected_w
          ML::Metal::Device.init!
          buf, off = weight_slot(w_raw)
          matmul_gemv_buf(mv_iq4_nl_pipeline, x, buf, off, in_dim, out_dim, batch)
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
                       when .iq4_nl? then mv_iq4_nl_pipeline
                       when .f32? then mv_f32_pipeline
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
            enc.dispatch_threadgroups(grid, {gemv_threads_per_tg_for(pipeline), 1, 1})
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
        def self.bench_q4_h16_pair_wait_ms(gate_qw : QuantWeight,
                                           up_qw : QuantWeight,
                                           x : Array(Float32),
                                           batch : Int32,
                                           validate : Bool = false) : Float64
          raise ArgumentError.new("Q4 H16 pair bench requires Q4_K weights") unless gate_qw.type.q4_k? && up_qw.type.q4_k?
          raise ArgumentError.new("Q4 H16 pair bench requires matching dimensions") unless gate_qw.in_dim == up_qw.in_dim && gate_qw.out_dim == up_qw.out_dim
          raise ArgumentError.new("Q4 H16 pair bench x size mismatch") unless x.size == gate_qw.in_dim * batch

          ML::Metal::Device.init!
          gate_buf, gate_off = if slot = mmap_slot_for(gate_qw.raw)
                                 slot
                               else
                                 {gate_qw.fallback_metal_buffer, 0_i64}
                               end
          up_buf, up_off = if slot = mmap_slot_for(up_qw.raw)
                             slot
                           else
                             {up_qw.fallback_metal_buffer, 0_i64}
                           end

          x_buf = Scratch.get(:bench_q4_pair_x, x.size.to_i64 * sizeof(Float32))
          x_buf.write(x)
          gate_out = Scratch.get(:bench_q4_pair_gate, (batch * gate_qw.out_dim).to_i64 * sizeof(Float32))
          up_out = Scratch.get(:bench_q4_pair_up, (batch * up_qw.out_dim).to_i64 * sizeof(Float32))
          if validate
            gate_out.contents.as(Pointer(Float32))[0] = Float32::NAN
            up_out.contents.as(Pointer(Float32))[0] = Float32::NAN
          end

          cmd = ML::Metal::CommandBuffer.new
          enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_q4k_gemm_h16_pair(enc, x_buf, gate_out, up_out,
            gate_buf, gate_off, up_buf, up_off,
            gate_qw.in_dim, gate_qw.out_dim, batch)
          enc.end_encoding

          t0 = Time.instant
          cmd.commit
          cmd.wait
          elapsed = (Time.instant - t0).total_milliseconds
          if validate
            gate0 = gate_out.contents.as(Pointer(Float32))[0]
            up0 = up_out.contents.as(Pointer(Float32))[0]
            unless gate0.finite? && up0.finite?
              raise "Q4_H16 pair bench validation failed: non-finite sample gate0=#{gate0} up0=#{up0}"
            end
          end
          elapsed
        end

        # Gemma-style FFN producer-consumer fusion: gate is materialized as F32,
        # up is consumed tile-local, and the output is GELU(gate) * up.
        def self.encode_q4k_gemm_h16_pair_b64_gelu_mul(enc : ML::Metal::ComputeEncoder,
                                                        gate_qw : QuantWeight,
                                                        up_qw : QuantWeight,
                                                        x_buf : ML::MetalBuffer,
                                                        gate_buf : ML::MetalBuffer,
                                                        act_buf : ML::MetalBuffer,
                                                        batch : Int32) : Bool
          return false unless q4_pair_h16_gemm_candidate?(gate_qw, up_qw, batch)
          return false unless q4_h16_b64_gemm_enabled?
          return false unless q4_h16_b64_swiglu_batch_candidate?(batch)
          return false unless (up_qw.out_dim % MM_NR0) == 0
          return false if x_buf.size < batch.to_i64 * gate_qw.in_dim * sizeof(Float32)
          return false if gate_buf.size < batch.to_i64 * gate_qw.out_dim * sizeof(Float32)
          return false if act_buf.size < batch.to_i64 * gate_qw.out_dim * sizeof(Float32)

          gate_w_buf, gate_w_off = weight_slot(gate_qw)
          up_w_buf, up_w_off = weight_slot(up_qw)
          x16_buf = Scratch.get(:mm4_pair_gelu_x16, (batch * gate_qw.in_dim).to_i64 * 2_i64)

          Profile.bump_conversion("f32_to_f16 q4_pair_gelu_input #{gate_qw.in_dim} b#{batch}", (batch * gate_qw.in_dim).to_i64 * 6_i64)
          enc.set_pipeline(f32_to_f16_pipeline)
          enc.set_buffer(x_buf, 0)
          enc.set_buffer(x16_buf, 1, ML::Metal::BufferAccess::Write)
          enc.set_value((batch * gate_qw.in_dim).to_u32, 2)
          enc.dispatch_1d(batch * gate_qw.in_dim, 256)

          encode_q4k_gemm_h16_from_h16(enc, x16_buf, gate_buf, gate_w_buf, gate_w_off, gate_qw.in_dim, gate_qw.out_dim, batch)
          encode_q4k_gemm_h16_b64_gelu_mul_from_h16(enc, x16_buf, gate_buf, act_buf, up_w_buf, up_w_off, up_qw.in_dim, up_qw.out_dim, batch)
          true
        end

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
          when .iq4_nl?
            if batch > GEMM_BATCH_THRESHOLD
              nil
            else
              matmul_gemv_buf(mv_iq4_nl_pipeline, x, buf, off, qw.in_dim, qw.out_dim, batch)
            end
          when .f32?
            if batch > GEMM_BATCH_THRESHOLD
              nil
            else
              matmul_gemv_buf(mv_f32_pipeline, x, buf, off, qw.in_dim, qw.out_dim, batch)
            end
          else
            nil
          end
        end

        # Resident-buffer matmul transport for callers that already keep the
        # input vector on Metal. This intentionally reuses the same encoder and
        # routing as `matmul` so new resident paths cannot silently diverge from
        # the established Q4/Q5/Q6/Q8/IQ4/F32 kernels.
        def self.matmul_to_buffer(qw : QuantWeight,
                                  x_buf : ML::MetalBuffer,
                                  out_buf : ML::MetalBuffer,
                                  batch : Int32 = 1) : Bool
          return false if batch <= 0
          return false if x_buf.size < batch.to_i64 * qw.in_dim * sizeof(Float32)
          return false if out_buf.size < batch.to_i64 * qw.out_dim * sizeof(Float32)

          ML::Metal::Device.init!
          pipeline = gemv_pipeline_for(qw)
          return false if pipeline.nil?
          w_buf, w_off = weight_slot(qw)

          cmd = ML::Metal::CommandBuffer.new
          enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_matmul(enc, pipeline, qw, x_buf, out_buf, w_buf, w_off, qw.in_dim, qw.out_dim, batch)
          enc.end_encoding
          cmd.commit
          cmd.wait
          true
        end

        def self.encode_matmul_to_buffer(enc : ML::Metal::ComputeEncoder,
                                         qw : QuantWeight,
                                         x_buf : ML::MetalBuffer,
                                         out_buf : ML::MetalBuffer,
                                         batch : Int32 = 1) : Bool
          return false if batch <= 0
          return false if x_buf.size < batch.to_i64 * qw.in_dim * sizeof(Float32)
          return false if out_buf.size < batch.to_i64 * qw.out_dim * sizeof(Float32)

          ML::Metal::Device.init!
          pipeline = gemv_pipeline_for(qw)
          return false if pipeline.nil?
          w_buf, w_off = weight_slot(qw)

          encode_matmul(enc, pipeline, qw, x_buf, out_buf, w_buf, w_off, qw.in_dim, qw.out_dim, batch)
          true
        end

        def self.matmul_many_to_buffers(qws : Array(QuantWeight),
                                        x_buf : ML::MetalBuffer,
                                        out_bufs : Array(ML::MetalBuffer),
                                        batch : Int32 = 1) : Bool
          return false if batch <= 0
          return false if qws.empty? || qws.size != out_bufs.size
          in_dim = qws[0].in_dim
          return false if x_buf.size < batch.to_i64 * in_dim * sizeof(Float32)

          slots = [] of {ML::Metal::ComputePipeline, ML::MetalBuffer, Int64, QuantWeight, ML::MetalBuffer}
          qws.each_with_index do |qw, i|
            return false unless qw.in_dim == in_dim
            out_buf = out_bufs[i]
            return false if out_buf.size < batch.to_i64 * qw.out_dim * sizeof(Float32)
            pipeline = gemv_pipeline_for(qw)
            return false if pipeline.nil?
            w_buf, w_off = weight_slot(qw)
            slots << {pipeline, w_buf, w_off, qw, out_buf}
          end

          ML::Metal::Device.init!
          cmd = ML::Metal::CommandBuffer.new
          enc = ML::Metal::ComputeEncoder.new(cmd)
          slots.each do |pipeline, w_buf, w_off, qw, out_buf|
            encode_matmul(enc, pipeline, qw, x_buf, out_buf, w_buf, w_off, qw.in_dim, qw.out_dim, batch)
          end
          enc.end_encoding
          cmd.commit
          cmd.wait
          true
        end

        def self.encode_matmul_many_to_buffers(enc : ML::Metal::ComputeEncoder,
                                               qws : Array(QuantWeight),
                                               x_buf : ML::MetalBuffer,
                                               out_bufs : Array(ML::MetalBuffer),
                                               batch : Int32 = 1) : Bool
          return false if batch <= 0
          return false if qws.empty? || qws.size != out_bufs.size
          in_dim = qws[0].in_dim
          return false if x_buf.size < batch.to_i64 * in_dim * sizeof(Float32)

          slots = [] of {ML::Metal::ComputePipeline, ML::MetalBuffer, Int64, QuantWeight, ML::MetalBuffer}
          qws.each_with_index do |qw, i|
            return false unless qw.in_dim == in_dim
            out_buf = out_bufs[i]
            return false if out_buf.size < batch.to_i64 * qw.out_dim * sizeof(Float32)
            pipeline = gemv_pipeline_for(qw)
            return false if pipeline.nil?
            w_buf, w_off = weight_slot(qw)
            slots << {pipeline, w_buf, w_off, qw, out_buf}
          end

          ML::Metal::Device.init!
          slots.each do |pipeline, w_buf, w_off, qw, out_buf|
            encode_matmul(enc, pipeline, qw, x_buf, out_buf, w_buf, w_off, qw.in_dim, qw.out_dim, batch)
          end
          true
        end
      {% end %}
    end
  end
end
