# Post-attention residual and FFN tail execution for Gemma4 Metal.

module ML::GGUF
  module Gemma4Metal
    {% unless flag?(:cpu_only) %}
      private def q4_gelu_fuse_enabled? : Bool
        ENV["GEMMA4_ROW_PREFILL_Q4_GELU_FUSE"]? == "1"
      end

      private def q4_pair_ffn_enabled? : Bool
        ENV["GEMMA4_ROW_PREFILL_Q4_PAIR_FFN"]? == "1"
      end

      private def q4_decode_dual_ffn_enabled? : Bool
        ENV["GEMMA4_DECODE_Q4_DUAL_GEMV"]? == "1" &&
          ENV["GEMMA4_DECODE_Q4_DUAL_GEMV_OFF"]? != "1"
      end

      private def residual_norm_fuse_enabled? : Bool
        ENV["GEMMA4_RESIDUAL_NORM_FUSE"]? == "1" &&
          ENV["GEMMA4_RESIDUAL_NORM_FUSE_OFF"]? != "1"
      end

      private def ffn_in_residual_norm_fuse_enabled?(batch : Int32) : Bool
        return false if ENV["GEMMA4_FFN_IN_RESIDUAL_NORM_FUSE_OFF"]? == "1"
        return true if ENV["GEMMA4_FFN_IN_RESIDUAL_NORM_FUSE"]? == "1"

        min_batch = (ENV["GEMMA4_FFN_IN_RESIDUAL_NORM_FUSE_MIN_BATCH"]? || "128").to_i32
        residual_norm_fuse_enabled? || batch >= min_batch
      end

      private def ffn_out_residual_norm_fuse_enabled? : Bool
        return false if ENV["GEMMA4_FFN_OUT_RESIDUAL_NORM_FUSE_OFF"]? == "1"

        residual_norm_fuse_enabled? || ENV["GEMMA4_FFN_OUT_RESIDUAL_NORM_FUSE"]? == "1"
      end

      class LayerTailGraphPlan
        getter out_buf : ML::MetalBuffer
        getter graph : ML::Metal::ComputeGraph

        def initialize(x_buf : ML::MetalBuffer,
                       attn_projected_buf : ML::MetalBuffer,
                       lw : Gemma4LayerWeights,
                       hp : Gemma4Hparams,
                       scratch : ResidentScratch? = nil)
          raise "Metal not available" unless Gemma4Metal.available?

          hidden_dim = hp.n_embd
          @out_buf = ML::MetalBuffer.new(hidden_dim.to_i64 * sizeof(Float32))
          @graph = ML::Metal::ComputeGraph.new
          enc = ML::Metal::GraphEncoder.new(@graph)
          ok = Gemma4Metal.encode_layer_tail_resident_buffer_inputs_graph(enc, x_buf, attn_projected_buf, @out_buf, lw, hp, scratch)
          raise "failed to encode Gemma4 layer-tail graph" unless ok
          @graph.compile!
        end

        def run! : ML::MetalBuffer
          cmd = ML::Metal::CommandBuffer.new
          @graph.encode(cmd)
          cmd.commit_and_wait
          @out_buf
        end

        def read(count : Int32) : Array(Float32)
          @out_buf.read(count)
        end
      end

      def layer_tail(x : Array(Float32),
                     attn_projected : Array(Float32),
                     lw : Gemma4LayerWeights,
                     hp : Gemma4Hparams) : Array(Float32)?
        return nil unless available?

        hidden_dim = hp.n_embd
        raise ArgumentError.new("layer_tail x size mismatch") unless x.size == hidden_dim
        raise ArgumentError.new("layer_tail attn_projected size mismatch") unless attn_projected.size == hidden_dim
        raise ArgumentError.new("layer_tail post_attention_norm size mismatch") unless lw.post_attention_norm.size == hidden_dim
        raise ArgumentError.new("layer_tail ffn_norm size mismatch") unless lw.ffn_norm.size == hidden_dim
        raise ArgumentError.new("layer_tail post_ffw_norm size mismatch") unless lw.post_ffw_norm.size == hidden_dim
        raise ArgumentError.new("layer_tail ffn gate/up mismatch") unless lw.ffn_gate_qw.in_dim == hidden_dim && lw.ffn_up_qw.in_dim == hidden_dim && lw.ffn_gate_qw.out_dim == lw.ffn_up_qw.out_dim
        raise ArgumentError.new("layer_tail ffn down mismatch") unless lw.ffn_down_qw.in_dim == lw.ffn_gate_qw.out_dim && lw.ffn_down_qw.out_dim == hidden_dim

        attn_normed = rms_norm(attn_projected, lw.post_attention_norm, hp.rms_eps).not_nil!
        attn_out = add_vec(x, attn_normed).not_nil!
        ffn_in = rms_norm(attn_out, lw.ffn_norm, hp.rms_eps).not_nil!
        gate_up = Qwen35Metal.matmul_many([lw.ffn_gate_qw, lw.ffn_up_qw], ffn_in)
        return nil unless gate_up
        combined = gelu_mul(gate_up[0], gate_up[1]).not_nil!
        ffn = Qwen35Metal.matmul(lw.ffn_down_qw, combined, 1)
        return nil unless ffn
        ffn_normed = rms_norm(ffn, lw.post_ffw_norm, hp.rms_eps).not_nil!
        scale = lw.layer_output_scale.first? || 1.0_f32
        add_scaled_vec(attn_out, ffn_normed, scale)
      end

      def layer_tail_resident_buffers(x : Array(Float32),
                                      attn_projected : Array(Float32),
                                      lw : Gemma4LayerWeights,
                                      hp : Gemma4Hparams) : Array(Float32)?
        return nil unless available?

        hidden_dim = hp.n_embd
        raise ArgumentError.new("layer_tail x size mismatch") unless x.size == hidden_dim
        raise ArgumentError.new("layer_tail attn_projected size mismatch") unless attn_projected.size == hidden_dim
        raise ArgumentError.new("layer_tail post_attention_norm size mismatch") unless lw.post_attention_norm.size == hidden_dim
        raise ArgumentError.new("layer_tail ffn_norm size mismatch") unless lw.ffn_norm.size == hidden_dim
        raise ArgumentError.new("layer_tail post_ffw_norm size mismatch") unless lw.post_ffw_norm.size == hidden_dim
        raise ArgumentError.new("layer_tail ffn gate/up mismatch") unless lw.ffn_gate_qw.in_dim == hidden_dim && lw.ffn_up_qw.in_dim == hidden_dim && lw.ffn_gate_qw.out_dim == lw.ffn_up_qw.out_dim
        raise ArgumentError.new("layer_tail ffn down mismatch") unless lw.ffn_down_qw.in_dim == lw.ffn_gate_qw.out_dim && lw.ffn_down_qw.out_dim == hidden_dim

        x_buf = ML::MetalBuffer.from_array(x)
        attn_buf = ML::MetalBuffer.from_array(attn_projected)
        post_attn_w = ML::MetalBuffer.from_array(lw.post_attention_norm)
        ffn_w = ML::MetalBuffer.from_array(lw.ffn_norm)
        post_ffw_w = ML::MetalBuffer.from_array(lw.post_ffw_norm)

        attn_normed_buf = ML::MetalBuffer.new(hidden_dim.to_i64 * sizeof(Float32))
        attn_out_buf = ML::MetalBuffer.new(hidden_dim.to_i64 * sizeof(Float32))
        ffn_in_buf = ML::MetalBuffer.new(hidden_dim.to_i64 * sizeof(Float32))
        gate_buf = ML::MetalBuffer.new(lw.ffn_gate_qw.out_dim.to_i64 * sizeof(Float32))
        up_buf = ML::MetalBuffer.new(lw.ffn_up_qw.out_dim.to_i64 * sizeof(Float32))
        combined_buf = ML::MetalBuffer.new(lw.ffn_down_qw.in_dim.to_i64 * sizeof(Float32))
        ffn_buf = ML::MetalBuffer.new(hidden_dim.to_i64 * sizeof(Float32))
        ffn_normed_buf = ML::MetalBuffer.new(hidden_dim.to_i64 * sizeof(Float32))
        out_buf = ML::MetalBuffer.new(hidden_dim.to_i64 * sizeof(Float32))

        cmd = ML::Metal::CommandBuffer.new
        enc = ML::Metal::ComputeEncoder.new(cmd)
        encode_rmsnorm_weighted_out(enc, attn_buf, post_attn_w, attn_normed_buf, hidden_dim, hp.rms_eps)
        encode_add_vec(enc, x_buf, attn_normed_buf, attn_out_buf, hidden_dim)
        encode_rmsnorm_weighted_out(enc, attn_out_buf, ffn_w, ffn_in_buf, hidden_dim, hp.rms_eps)
        enc.end_encoding
        cmd.commit
        cmd.wait

        return nil unless Qwen35Metal.matmul_many_to_buffers([lw.ffn_gate_qw, lw.ffn_up_qw], ffn_in_buf, [gate_buf, up_buf], 1)

        cmd2 = ML::Metal::CommandBuffer.new
        enc2 = ML::Metal::ComputeEncoder.new(cmd2)
        encode_gelu_mul(enc2, gate_buf, up_buf, combined_buf, lw.ffn_down_qw.in_dim)
        enc2.end_encoding
        cmd2.commit
        cmd2.wait

        return nil unless Qwen35Metal.matmul_to_buffer(lw.ffn_down_qw, combined_buf, ffn_buf, 1)

        scale = lw.layer_output_scale.first? || 1.0_f32
        cmd3 = ML::Metal::CommandBuffer.new
        enc3 = ML::Metal::ComputeEncoder.new(cmd3)
        encode_rmsnorm_weighted_out(enc3, ffn_buf, post_ffw_w, ffn_normed_buf, hidden_dim, hp.rms_eps)
        encode_add_scaled_vec(enc3, attn_out_buf, ffn_normed_buf, out_buf, hidden_dim, scale)
        enc3.end_encoding
        cmd3.commit
        cmd3.wait
        out_buf.read(hidden_dim)
      end

      def encode_layer_tail_resident_buffer_inputs_graph(enc : ML::Metal::GraphEncoder,
                                                         x_buf : ML::MetalBuffer,
                                                         attn_projected_buf : ML::MetalBuffer,
                                                         out_buf : ML::MetalBuffer,
                                                         lw : Gemma4LayerWeights,
                                                         hp : Gemma4Hparams,
                                                         scratch : ResidentScratch? = nil) : Bool
        hidden_dim = hp.n_embd
        raise ArgumentError.new("graph layer_tail x buffer too small") if x_buf.size < hidden_dim.to_i64 * sizeof(Float32)
        raise ArgumentError.new("graph layer_tail attn_projected buffer too small") if attn_projected_buf.size < hidden_dim.to_i64 * sizeof(Float32)
        raise ArgumentError.new("graph layer_tail out buffer too small") if out_buf.size < hidden_dim.to_i64 * sizeof(Float32)
        raise ArgumentError.new("graph layer_tail post_attention_norm size mismatch") unless lw.post_attention_norm.size == hidden_dim
        raise ArgumentError.new("graph layer_tail ffn_norm size mismatch") unless lw.ffn_norm.size == hidden_dim
        raise ArgumentError.new("graph layer_tail post_ffw_norm size mismatch") unless lw.post_ffw_norm.size == hidden_dim
        raise ArgumentError.new("graph layer_tail ffn gate/up mismatch") unless lw.ffn_gate_qw.in_dim == hidden_dim && lw.ffn_up_qw.in_dim == hidden_dim && lw.ffn_gate_qw.out_dim == lw.ffn_up_qw.out_dim
        raise ArgumentError.new("graph layer_tail ffn down mismatch") unless lw.ffn_down_qw.in_dim == lw.ffn_gate_qw.out_dim && lw.ffn_down_qw.out_dim == hidden_dim

        post_attn_w = ML::MetalBuffer.from_array(lw.post_attention_norm)
        ffn_w = ML::MetalBuffer.from_array(lw.ffn_norm)
        post_ffw_w = ML::MetalBuffer.from_array(lw.post_ffw_norm)

        attn_normed_buf = scratch ? scratch.not_nil!.get("tail.graph.attn_normed", hidden_dim.to_i64 * sizeof(Float32)) : ML::MetalBuffer.new(hidden_dim.to_i64 * sizeof(Float32))
        attn_out_buf = scratch ? scratch.not_nil!.get("tail.graph.attn_out", hidden_dim.to_i64 * sizeof(Float32)) : ML::MetalBuffer.new(hidden_dim.to_i64 * sizeof(Float32))
        ffn_in_buf = scratch ? scratch.not_nil!.get("tail.graph.ffn_in", hidden_dim.to_i64 * sizeof(Float32)) : ML::MetalBuffer.new(hidden_dim.to_i64 * sizeof(Float32))
        gate_buf = scratch ? scratch.not_nil!.get("tail.graph.gate", lw.ffn_gate_qw.out_dim.to_i64 * sizeof(Float32)) : ML::MetalBuffer.new(lw.ffn_gate_qw.out_dim.to_i64 * sizeof(Float32))
        up_buf = scratch ? scratch.not_nil!.get("tail.graph.up", lw.ffn_up_qw.out_dim.to_i64 * sizeof(Float32)) : ML::MetalBuffer.new(lw.ffn_up_qw.out_dim.to_i64 * sizeof(Float32))
        combined_buf = scratch ? scratch.not_nil!.get("tail.graph.combined", lw.ffn_down_qw.in_dim.to_i64 * sizeof(Float32)) : ML::MetalBuffer.new(lw.ffn_down_qw.in_dim.to_i64 * sizeof(Float32))
        ffn_buf = scratch ? scratch.not_nil!.get("tail.graph.ffn", hidden_dim.to_i64 * sizeof(Float32)) : ML::MetalBuffer.new(hidden_dim.to_i64 * sizeof(Float32))
        ffn_normed_buf = scratch ? scratch.not_nil!.get("tail.graph.ffn_normed", hidden_dim.to_i64 * sizeof(Float32)) : ML::MetalBuffer.new(hidden_dim.to_i64 * sizeof(Float32))

        encode_rmsnorm_weighted_out(enc, attn_projected_buf, post_attn_w, attn_normed_buf, hidden_dim, hp.rms_eps)
        encode_add_vec(enc, x_buf, attn_normed_buf, attn_out_buf, hidden_dim)
        encode_rmsnorm_weighted_out(enc, attn_out_buf, ffn_w, ffn_in_buf, hidden_dim, hp.rms_eps)
        return false unless Qwen35Metal.encode_matmul_many_to_buffers(enc, [lw.ffn_gate_qw, lw.ffn_up_qw], ffn_in_buf, [gate_buf, up_buf], 1)
        encode_gelu_mul(enc, gate_buf, up_buf, combined_buf, lw.ffn_down_qw.in_dim)
        return false unless Qwen35Metal.encode_matmul_to_buffer(enc, lw.ffn_down_qw, combined_buf, ffn_buf, 1)
        scale = lw.layer_output_scale.first? || 1.0_f32
        encode_rmsnorm_weighted_out(enc, ffn_buf, post_ffw_w, ffn_normed_buf, hidden_dim, hp.rms_eps)
        encode_add_scaled_vec(enc, attn_out_buf, ffn_normed_buf, out_buf, hidden_dim, scale)
        true
      end

      def encode_layer_tail_resident_buffer_inputs(enc : ML::Metal::ComputeEncoder,
                                                   x_buf : ML::MetalBuffer,
                                                   attn_projected_buf : ML::MetalBuffer,
                                                   out_buf : ML::MetalBuffer,
                                                   lw : Gemma4LayerWeights,
                                                   hp : Gemma4Hparams,
                                                   scratch : ResidentScratch? = nil) : Bool
        return false unless available?

        hidden_dim = hp.n_embd
        raise ArgumentError.new("layer_tail x buffer too small") if x_buf.size < hidden_dim.to_i64 * sizeof(Float32)
        raise ArgumentError.new("layer_tail attn_projected buffer too small") if attn_projected_buf.size < hidden_dim.to_i64 * sizeof(Float32)
        raise ArgumentError.new("layer_tail out buffer too small") if out_buf.size < hidden_dim.to_i64 * sizeof(Float32)
        raise ArgumentError.new("layer_tail post_attention_norm size mismatch") unless lw.post_attention_norm.size == hidden_dim
        raise ArgumentError.new("layer_tail ffn_norm size mismatch") unless lw.ffn_norm.size == hidden_dim
        raise ArgumentError.new("layer_tail post_ffw_norm size mismatch") unless lw.post_ffw_norm.size == hidden_dim
        raise ArgumentError.new("layer_tail ffn gate/up mismatch") unless lw.ffn_gate_qw.in_dim == hidden_dim && lw.ffn_up_qw.in_dim == hidden_dim && lw.ffn_gate_qw.out_dim == lw.ffn_up_qw.out_dim
        raise ArgumentError.new("layer_tail ffn down mismatch") unless lw.ffn_down_qw.in_dim == lw.ffn_gate_qw.out_dim && lw.ffn_down_qw.out_dim == hidden_dim

        post_attn_w = ML::MetalBuffer.from_array(lw.post_attention_norm)
        ffn_w = ML::MetalBuffer.from_array(lw.ffn_norm)
        post_ffw_w = ML::MetalBuffer.from_array(lw.post_ffw_norm)

        attn_normed_buf = scratch ? scratch.not_nil!.get("tail.attn_normed", hidden_dim.to_i64 * sizeof(Float32)) : ML::MetalBuffer.new(hidden_dim.to_i64 * sizeof(Float32))
        attn_out_buf = scratch ? scratch.not_nil!.get("tail.attn_out", hidden_dim.to_i64 * sizeof(Float32)) : ML::MetalBuffer.new(hidden_dim.to_i64 * sizeof(Float32))
        ffn_in_buf = scratch ? scratch.not_nil!.get("tail.ffn_in", hidden_dim.to_i64 * sizeof(Float32)) : ML::MetalBuffer.new(hidden_dim.to_i64 * sizeof(Float32))
        gate_buf = scratch ? scratch.not_nil!.get("tail.gate", lw.ffn_gate_qw.out_dim.to_i64 * sizeof(Float32)) : ML::MetalBuffer.new(lw.ffn_gate_qw.out_dim.to_i64 * sizeof(Float32))
        up_buf = scratch ? scratch.not_nil!.get("tail.up", lw.ffn_up_qw.out_dim.to_i64 * sizeof(Float32)) : ML::MetalBuffer.new(lw.ffn_up_qw.out_dim.to_i64 * sizeof(Float32))
        combined_buf = scratch ? scratch.not_nil!.get("tail.combined", lw.ffn_down_qw.in_dim.to_i64 * sizeof(Float32)) : ML::MetalBuffer.new(lw.ffn_down_qw.in_dim.to_i64 * sizeof(Float32))
        ffn_buf = scratch ? scratch.not_nil!.get("tail.ffn", hidden_dim.to_i64 * sizeof(Float32)) : ML::MetalBuffer.new(hidden_dim.to_i64 * sizeof(Float32))
        ffn_normed_buf = scratch ? scratch.not_nil!.get("tail.ffn_normed", hidden_dim.to_i64 * sizeof(Float32)) : ML::MetalBuffer.new(hidden_dim.to_i64 * sizeof(Float32))

        encode_rmsnorm_weighted_out(enc, attn_projected_buf, post_attn_w, attn_normed_buf, hidden_dim, hp.rms_eps)
        encode_add_vec(enc, x_buf, attn_normed_buf, attn_out_buf, hidden_dim)
        encode_rmsnorm_weighted_out(enc, attn_out_buf, ffn_w, ffn_in_buf, hidden_dim, hp.rms_eps)
        projected = q4_decode_dual_ffn_enabled? &&
          Qwen35Metal.encode_q4k_gemv_dual_to_buffers(enc, lw.ffn_gate_qw, lw.ffn_up_qw, ffn_in_buf, gate_buf, up_buf, 1)
        return false unless projected || Qwen35Metal.encode_matmul_many_to_buffers(enc, [lw.ffn_gate_qw, lw.ffn_up_qw], ffn_in_buf, [gate_buf, up_buf], 1)
        encode_gelu_mul(enc, gate_buf, up_buf, combined_buf, lw.ffn_down_qw.in_dim)
        return false unless Qwen35Metal.encode_matmul_to_buffer(enc, lw.ffn_down_qw, combined_buf, ffn_buf, 1)
        scale = lw.layer_output_scale.first? || 1.0_f32
        encode_rmsnorm_weighted_out(enc, ffn_buf, post_ffw_w, ffn_normed_buf, hidden_dim, hp.rms_eps)
        encode_add_scaled_vec(enc, attn_out_buf, ffn_normed_buf, out_buf, hidden_dim, scale)
        true
      end

      def layer_tail_resident_buffer_inputs(x_buf : ML::MetalBuffer,
                                            attn_projected_buf : ML::MetalBuffer,
                                            lw : Gemma4LayerWeights,
                                            hp : Gemma4Hparams,
                                            scratch : ResidentScratch? = nil) : ML::MetalBuffer?
        return nil unless available?

        hidden_dim = hp.n_embd
        out_buf = ML::MetalBuffer.new(hidden_dim.to_i64 * sizeof(Float32))

        cmd = ML::Metal::CommandBuffer.new
        enc = ML::Metal::ComputeEncoder.new(cmd)
        ok = encode_layer_tail_resident_buffer_inputs(enc, x_buf, attn_projected_buf, out_buf, lw, hp, scratch)
        enc.end_encoding
        return nil unless ok
        cmd.commit_and_wait
        out_buf
      end

      def layer_tail_resident_buffer_inputs_graph(x_buf : ML::MetalBuffer,
                                                  attn_projected_buf : ML::MetalBuffer,
                                                  lw : Gemma4LayerWeights,
                                                  hp : Gemma4Hparams,
                                                  scratch : ResidentScratch? = nil) : ML::MetalBuffer?
        return nil unless available?

        hidden_dim = hp.n_embd
        out_buf = ML::MetalBuffer.new(hidden_dim.to_i64 * sizeof(Float32))
        graph = ML::Metal::ComputeGraph.new
        enc = ML::Metal::GraphEncoder.new(graph)
        ok = encode_layer_tail_resident_buffer_inputs_graph(enc, x_buf, attn_projected_buf, out_buf, lw, hp, scratch)
        return nil unless ok
        graph.compile!

        cmd = ML::Metal::CommandBuffer.new
        graph.encode(cmd)
        cmd.commit_and_wait
        out_buf
      end

      def layer_tail_batch(x_rows : Array(Float32),
                           attn_projected_rows : Array(Float32),
                           lw : Gemma4LayerWeights,
                           hp : Gemma4Hparams,
                           batch : Int32) : Array(Float32)?
        return nil unless available?

        hidden_dim = hp.n_embd
        ffn_dim = lw.ffn_gate_qw.out_dim
        raise ArgumentError.new("batch must be positive") unless batch > 0
        raise ArgumentError.new("layer_tail_batch x size mismatch") unless x_rows.size == batch * hidden_dim
        raise ArgumentError.new("layer_tail_batch attn size mismatch") unless attn_projected_rows.size == batch * hidden_dim
        raise ArgumentError.new("layer_tail_batch post_attention_norm size mismatch") unless lw.post_attention_norm.size == hidden_dim
        raise ArgumentError.new("layer_tail_batch ffn_norm size mismatch") unless lw.ffn_norm.size == hidden_dim
        raise ArgumentError.new("layer_tail_batch post_ffw_norm size mismatch") unless lw.post_ffw_norm.size == hidden_dim

        x_buf = ML::MetalBuffer.from_array(x_rows)
        attn_buf = ML::MetalBuffer.from_array(attn_projected_rows)
        post_attn_w = ML::MetalBuffer.from_array(lw.post_attention_norm)
        ffn_w = ML::MetalBuffer.from_array(lw.ffn_norm)
        post_ffw_w = ML::MetalBuffer.from_array(lw.post_ffw_norm)

        attn_normed_buf = ML::MetalBuffer.new(batch.to_i64 * hidden_dim * sizeof(Float32))
        attn_out_buf = ML::MetalBuffer.new(batch.to_i64 * hidden_dim * sizeof(Float32))
        ffn_in_buf = ML::MetalBuffer.new(batch.to_i64 * hidden_dim * sizeof(Float32))
        gate_buf = ML::MetalBuffer.new(batch.to_i64 * ffn_dim * sizeof(Float32))
        up_buf = ML::MetalBuffer.new(batch.to_i64 * ffn_dim * sizeof(Float32))
        combined_buf = ML::MetalBuffer.new(batch.to_i64 * ffn_dim * sizeof(Float32))
        ffn_buf = ML::MetalBuffer.new(batch.to_i64 * hidden_dim * sizeof(Float32))
        ffn_normed_buf = ML::MetalBuffer.new(batch.to_i64 * hidden_dim * sizeof(Float32))
        out_buf = ML::MetalBuffer.new(batch.to_i64 * hidden_dim * sizeof(Float32))

        cmd = ML::Metal::CommandBuffer.new
        enc = ML::Metal::ComputeEncoder.new(cmd)
        if ffn_in_residual_norm_fuse_enabled?(batch)
          encode_rmsnorm_add_rows(enc, attn_buf, post_attn_w, x_buf, attn_out_buf, hidden_dim, batch, hp.rms_eps)
        else
          encode_rmsnorm_rows_weighted_out(enc, attn_buf, post_attn_w, attn_normed_buf, hidden_dim, batch, hp.rms_eps)
          encode_add_vec(enc, x_buf, attn_normed_buf, attn_out_buf, batch * hidden_dim)
        end
        encode_rmsnorm_rows_weighted_out(enc, attn_out_buf, ffn_w, ffn_in_buf, hidden_dim, batch, hp.rms_eps)
        fused_gelu = q4_gelu_fuse_enabled? &&
          Qwen35Metal.encode_q4k_gemm_h16_pair_b64_gelu_mul(enc, lw.ffn_gate_qw, lw.ffn_up_qw, ffn_in_buf, gate_buf, combined_buf, batch)
        unless fused_gelu
          dual_q4 = q4_decode_dual_ffn_enabled? &&
            Qwen35Metal.encode_q4k_gemv_dual_to_buffers(enc, lw.ffn_gate_qw, lw.ffn_up_qw, ffn_in_buf, gate_buf, up_buf, batch)
          pair_q4 = !dual_q4 && q4_pair_ffn_enabled? &&
            Qwen35Metal.encode_q4k_gemm_h16_pair_to_buffers(enc, lw.ffn_gate_qw, lw.ffn_up_qw, ffn_in_buf, gate_buf, up_buf, batch)
          unless dual_q4 || pair_q4 || Qwen35Metal.encode_matmul_many_to_buffers(enc, [lw.ffn_gate_qw, lw.ffn_up_qw], ffn_in_buf, [gate_buf, up_buf], batch)
            enc.end_encoding
            return nil
          end
          encode_gelu_mul(enc, gate_buf, up_buf, combined_buf, batch * ffn_dim)
        end
        unless Qwen35Metal.encode_matmul_to_buffer(enc, lw.ffn_down_qw, combined_buf, ffn_buf, batch)
          enc.end_encoding
          return nil
        end
        scale = lw.layer_output_scale.first? || 1.0_f32
        if ffn_out_residual_norm_fuse_enabled?
          encode_rmsnorm_add_scaled_rows(enc, ffn_buf, post_ffw_w, attn_out_buf, out_buf, hidden_dim, batch, hp.rms_eps, scale)
        else
          encode_rmsnorm_rows_weighted_out(enc, ffn_buf, post_ffw_w, ffn_normed_buf, hidden_dim, batch, hp.rms_eps)
          encode_add_scaled_vec(enc, attn_out_buf, ffn_normed_buf, out_buf, batch * hidden_dim, scale)
        end
        enc.end_encoding
        cmd.commit
        cmd.wait
        out_buf.read(batch * hidden_dim)
      end


    {% end %}
  end
end
