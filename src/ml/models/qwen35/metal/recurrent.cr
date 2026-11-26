# Recurrent convolution, DeltaNet state, and layer dispatch for the Qwen35 Metal engine.

module ML
  module GGUF
    module Qwen35Metal
      {% unless flag?(:cpu_only) %}
        DELTA_NET_SOURCE = {{ read_file("#{__DIR__}/../../../metal/kernels/qwen35/delta_net.metal") }}
        FFN_UPDOWN_Q8_SOURCE = {{ read_file("#{__DIR__}/../../../metal/kernels/qwen35/ffn_updown_q8.metal") }}

        RECURRENT_SOURCE = {{ read_file("#{__DIR__}/../../../metal/kernels/qwen35/recurrent_qwen35.metal") }}

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

        # The experimental Q4 Tensor route is not parity-certified for recurrent
        # projections. Preserve their established F32-input dispatcher when it is
        # explicitly enabled instead of widening that experiment through H16 reuse.
        private def self.recurrent_h16_projection_candidate?(qw : QuantWeight) : Bool
          !qw.type.q4_k? || !q4_tensor_mm_enabled?
        end

        private def self.q5_qkv_h16_conv_enabled? : Bool
          ENV["QWEN35_Q5_QKV_H16_CONV_OFF"]? != "1"
        end

        private def self.rec_proj_shared_h16_enabled? : Bool
          ENV["QWEN35_REC_PROJ_SHARED_H16_OFF"]? != "1"
        end

        private def self.prefill_dn_post_h16_oproj_enabled? : Bool
          ENV["QWEN35_DN_POST_H16_OPROJ"]? == "1"
        end

        private def self.q8_alpha_beta_dual_gemv_enabled? : Bool
          q8_dual_gemv_enabled? && ENV["QWEN35_Q8_ALPHA_BETA_DUAL_GEMV_OFF"]? != "1"
        end

        private def self.q8_alpha_beta_dual_gemv_candidate?(alpha_qw : QuantWeight,
                                                            beta_qw : QuantWeight,
                                                            batch : Int32 = 1) : Bool
          q8_alpha_beta_dual_gemv_enabled? && batch == 1 &&
            alpha_qw.type.q8_0? && beta_qw.type.q8_0? &&
            alpha_qw.in_dim == beta_qw.in_dim &&
            alpha_qw.out_dim == beta_qw.out_dim
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

        private def self.recurrent_conv_shift_fused_enabled? : Bool
          ENV["QWEN35_REC_CONVSHIFT_FUSED"]? == "1"
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
          ffn_act_buf = prefill_ffn_activation_buffer(ffn_up_buf) do
            Scratch.get(:rec_chunk_many_ffn_comb, (n_tokens * ffn_dim).to_i64 * sizeof(Float32))
          end
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
              rec_projection_qws = [lw.attn_qkv_qw, lw.attn_gate_qw]
              if !qkv_h16 && !norm_h16_proj && prefill_shared_projection_h16_candidate?(rec_projection_qws, n_tokens, checkpoint_requested)
                encode_shared_projection_h16(proj_enc, rec_projection_qws, cur_buf,
                  [qkv_buf, z_buf], n_tokens, "rec_chunk_projection_x16")
              elsif shared_h16 && norm_h16_proj
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
                unless !checkpoint_requested && norm_h16_proj && recurrent_h16_projection_candidate?(lw.attn_qkv_qw) && encode_matmul_from_h16(proj_enc, lw.attn_qkv_qw, cur_h16_buf, qkv_buf, qkv_w_buf, qkv_w_off, lw.attn_qkv_qw.in_dim, lw.attn_qkv_qw.out_dim, n_tokens)
                  encode_matmul(proj_enc, gemv_pipeline_for(lw.attn_qkv_qw).not_nil!, lw.attn_qkv_qw, cur_buf, qkv_buf, qkv_w_buf, qkv_w_off, lw.attn_qkv_qw.in_dim, lw.attn_qkv_qw.out_dim, n_tokens)
                end
                unless !checkpoint_requested && norm_h16_proj && recurrent_h16_projection_candidate?(lw.attn_gate_qw) && encode_matmul_from_h16(proj_enc, lw.attn_gate_qw, cur_h16_buf, z_buf, gate_w_buf, gate_w_off, lw.attn_gate_qw.in_dim, lw.attn_gate_qw.out_dim, n_tokens)
                  encode_matmul(proj_enc, gemv_pipeline_for(lw.attn_gate_qw).not_nil!, lw.attn_gate_qw, cur_buf, z_buf, gate_w_buf, gate_w_off, lw.attn_gate_qw.in_dim, lw.attn_gate_qw.out_dim, n_tokens)
                end
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
              checked = prefill_phase_checkpoint(cmd, "#{profile_label}.rec#{local_i}.#{checkpoint_name}", phase_t0,
                next_command: local_i + 1 < layers.size)
              cmd = checked[0]
              phase_t0 = checked[1]
            end

            src_buf, dst_buf = dst_buf, src_buf
          end

          if full_detail_profile
            # The terminal checkpoint completed all work without a successor.
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
      {% end %}
    end
  end
end
