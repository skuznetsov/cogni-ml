require "./gemma4_cpu"
require "./qwen35_metal"

{% unless flag?(:cpu_only) %}
  require "../metal/device"
  require "../metal/dispatch"
  require "../core/buffer"
{% end %}

module ML::GGUF
  module Gemma4Metal
    extend self

    class LayerState
      property k_cache : Array(Float32)
      property v_cache : Array(Float32)

      def initialize(kv_dim : Int32, max_seq : Int32)
        @k_cache = Array(Float32).new(max_seq * kv_dim, 0.0_f32)
        @v_cache = Array(Float32).new(max_seq * kv_dim, 0.0_f32)
      end
    end

    class State
      getter layers : Array(LayerState)
      getter max_seq : Int32

      def initialize(hp : Gemma4Hparams, @max_seq : Int32 = 1024)
        @layers = Array(LayerState).new(hp.n_layer) do |il|
          kv_dim = hp.n_head_kv(il) * hp.head_dim_for_layer(il)
          LayerState.new(kv_dim, @max_seq)
        end
      end
    end

    {% if flag?(:cpu_only) %}
      def available? : Bool
        false
      end

      def normalize_and_rope_projection(proj : Gemma4CPU::AttentionProjection,
                                        lw : Gemma4LayerWeights,
                                        hp : Gemma4Hparams,
                                        il : Int32,
                                        pos : Int32,
                                        rope_freqs : Array(Float32)? = nil) : Gemma4CPU::AttentionProjection?
        nil
      end

      def attention_context_from_projection(proj : Gemma4CPU::AttentionProjection,
                                            hp : Gemma4Hparams,
                                            il : Int32,
                                            pos : Int32,
                                            max_seq : Int32,
                                            k_cache : Array(Float32),
                                            v_cache : Array(Float32)) : NamedTuple(context: Array(Float32), k_cache: Array(Float32), v_cache: Array(Float32))?
        nil
      end

      def layer_tail(x : Array(Float32),
                     attn_projected : Array(Float32),
                     lw : Gemma4LayerWeights,
                     hp : Gemma4Hparams) : Array(Float32)?
        nil
      end

      def forward_layer_resident_cache_rows(weights : Gemma4Weights,
                                            il : Int32,
                                            x_rows : Array(Float32),
                                            base_pos : Int32,
                                            batch : Int32,
                                            state : ResidentState) : Array(Float32)?
        nil
      end

      def prefill_tokens_last_hidden_resident_rows(weights : Gemma4Weights,
                                                   token_ids : Array(Int32),
                                                   start_pos : Int32,
                                                   state : ResidentState,
                                                   chunk_size : Int32 = 8,
                                                   stop_layer : Int32? = nil) : Array(Float32)?
        nil
      end

      def forward_layer(weights : Gemma4Weights,
                        il : Int32,
                        x : Array(Float32),
                        pos : Int32,
                        max_seq : Int32,
                        k_cache : Array(Float32),
                        v_cache : Array(Float32)) : NamedTuple(out: Array(Float32), k_cache: Array(Float32), v_cache: Array(Float32))?
        nil
      end

      def forward_logits_from_hidden(weights : Gemma4Weights,
                                     hidden : Array(Float32)) : Array(Float32)?
        nil
      end

      def forward_hidden(weights : Gemma4Weights,
                         token_id : Int32,
                         pos : Int32,
                         state : State,
                         stop_layer : Int32? = nil) : Array(Float32)?
        nil
      end
    {% else %}
      GEMMA4_SOURCE = {{ read_file("#{__DIR__}/kernels/gemma4.metal") }}

      private def q4_gelu_fuse_enabled? : Bool
        ENV["GEMMA4_ROW_PREFILL_Q4_GELU_FUSE"]? == "1"
      end

      @@rmsnorm_weighted_pipeline : ML::Metal::ComputePipeline?
      @@rmsnorm_vec_weighted_pipeline : ML::Metal::ComputePipeline?
      @@rmsnorm_rows_weighted_pipeline : ML::Metal::ComputePipeline?
      @@rmsnorm_plain_pipeline : ML::Metal::ComputePipeline?
      @@rmsnorm_heads_weighted_rows_pipeline : ML::Metal::ComputePipeline?
      @@rmsnorm_heads_plain_rows_pipeline : ML::Metal::ComputePipeline?
      @@rope_pipeline : ML::Metal::ComputePipeline?
      @@rope_rows_pipeline : ML::Metal::ComputePipeline?
      @@kv_write_pipeline : ML::Metal::ComputePipeline?
      @@kv_write_rows_pipeline : ML::Metal::ComputePipeline?
      @@attn_context_pipeline : ML::Metal::ComputePipeline?
      @@attn_context_rows_pipeline : ML::Metal::ComputePipeline?
      @@add_vec_pipeline : ML::Metal::ComputePipeline?
      @@add_scaled_vec_pipeline : ML::Metal::ComputePipeline?
      @@gelu_mul_pipeline : ML::Metal::ComputePipeline?
      @@softcap_pipeline : ML::Metal::ComputePipeline?

      class ResidentLayerState
        getter k_cache_buf : ML::MetalBuffer
        getter v_cache_buf : ML::MetalBuffer
        getter kv_dim : Int32
        getter max_seq : Int32

        def initialize(@kv_dim : Int32, @max_seq : Int32)
          @k_cache_buf = ML::MetalBuffer.from_array(Array(Float32).new(max_seq * kv_dim, 0.0_f32))
          @v_cache_buf = ML::MetalBuffer.from_array(Array(Float32).new(max_seq * kv_dim, 0.0_f32))
        end

        def k_cache : Array(Float32)
          @k_cache_buf.read(@max_seq * @kv_dim)
        end

        def v_cache : Array(Float32)
          @v_cache_buf.read(@max_seq * @kv_dim)
        end
      end

      class ResidentScratch
        @buffers = {} of String => ML::MetalBuffer

        def get(name : String, byte_size : Int64) : ML::MetalBuffer
          if buf = @buffers[name]?
            return buf if buf.size >= byte_size
          end
          buf = ML::MetalBuffer.new(byte_size)
          @buffers[name] = buf
          buf
        end
      end

      class ResidentState
        getter layers : Array(ResidentLayerState)
        getter max_seq : Int32
        getter scratch : ResidentScratch

        def initialize(hp : Gemma4Hparams, @max_seq : Int32 = 1024)
          @layers = Array(ResidentLayerState).new(hp.n_layer) do |il|
            kv_dim = hp.n_head_kv(il) * hp.head_dim_for_layer(il)
            ResidentLayerState.new(kv_dim, @max_seq)
          end
          @scratch = ResidentScratch.new
        end
      end

      def available? : Bool
        ML::Metal::Device.init!
      end

      def normalize_and_rope_projection(proj : Gemma4CPU::AttentionProjection,
                                        lw : Gemma4LayerWeights,
                                        hp : Gemma4Hparams,
                                        il : Int32,
                                        pos : Int32,
                                        rope_freqs : Array(Float32)? = nil) : Gemma4CPU::AttentionProjection?
        return nil unless available?

        head_dim = hp.head_dim_for_layer(il)
        rope_dim = hp.rope_dim_for_layer(il)
        n_head = hp.n_head
        n_head_kv = hp.n_head_kv(il)
        raise ArgumentError.new("q projection size mismatch at layer #{il}") unless proj.q.size == n_head * head_dim
        raise ArgumentError.new("k projection size mismatch at layer #{il}") unless proj.k.size == n_head_kv * head_dim
        raise ArgumentError.new("v projection size mismatch at layer #{il}") unless proj.v.size == n_head_kv * head_dim
        raise ArgumentError.new("rope_dim #{rope_dim} must be even") unless rope_dim.even?

        q_buf = ML::MetalBuffer.from_array(proj.q)
        k_buf = ML::MetalBuffer.from_array(proj.k)
        v_buf = ML::MetalBuffer.from_array(proj.v)
        q_weight = ML::MetalBuffer.from_array(lw.attn_q_norm)
        k_weight = ML::MetalBuffer.from_array(lw.attn_k_norm)
        factors = (hp.full_attention?(il) && rope_freqs) ? rope_freqs.not_nil! : [1.0_f32]
        factor_buf = ML::MetalBuffer.from_array(factors)
        use_factors = (hp.full_attention?(il) && rope_freqs) ? 1_u32 : 0_u32
        base = hp.rope_freq_base_for_layer(il)

        cmd = ML::Metal::CommandBuffer.new
        enc = ML::Metal::ComputeEncoder.new(cmd)
        encode_rmsnorm_weighted(enc, q_buf, q_weight, head_dim, hp.rms_eps, n_head)
        encode_rmsnorm_weighted(enc, k_buf, k_weight, head_dim, hp.rms_eps, n_head_kv)
        encode_rmsnorm_plain(enc, v_buf, head_dim, hp.rms_eps, n_head_kv)
        encode_rope(enc, q_buf, factor_buf, head_dim, rope_dim, pos, base, use_factors, n_head)
        encode_rope(enc, k_buf, factor_buf, head_dim, rope_dim, pos, base, use_factors, n_head_kv)
        enc.end_encoding
        cmd.commit
        cmd.wait

        Gemma4CPU::AttentionProjection.new(
          q_buf.read(proj.q.size),
          k_buf.read(proj.k.size),
          v_buf.read(proj.v.size),
          proj.reused_k_as_v,
        )
      end

      def normalize_and_rope_projection_buffers(q_buf : ML::MetalBuffer,
                                                k_buf : ML::MetalBuffer,
                                                v_buf : ML::MetalBuffer,
                                                lw : Gemma4LayerWeights,
                                                hp : Gemma4Hparams,
                                                il : Int32,
                                                pos : Int32,
                                                rope_freqs : Array(Float32)? = nil) : Bool
        return false unless available?

        head_dim = hp.head_dim_for_layer(il)
        rope_dim = hp.rope_dim_for_layer(il)
        n_head = hp.n_head
        n_head_kv = hp.n_head_kv(il)
        return false if q_buf.size < n_head.to_i64 * head_dim * sizeof(Float32)
        return false if k_buf.size < n_head_kv.to_i64 * head_dim * sizeof(Float32)
        return false if v_buf.size < n_head_kv.to_i64 * head_dim * sizeof(Float32)
        raise ArgumentError.new("rope_dim #{rope_dim} must be even") unless rope_dim.even?

        q_weight = ML::MetalBuffer.from_array(lw.attn_q_norm)
        k_weight = ML::MetalBuffer.from_array(lw.attn_k_norm)
        factors = (hp.full_attention?(il) && rope_freqs) ? rope_freqs.not_nil! : [1.0_f32]
        factor_buf = ML::MetalBuffer.from_array(factors)
        use_factors = (hp.full_attention?(il) && rope_freqs) ? 1_u32 : 0_u32
        base = hp.rope_freq_base_for_layer(il)

        cmd = ML::Metal::CommandBuffer.new
        enc = ML::Metal::ComputeEncoder.new(cmd)
        encode_rmsnorm_weighted(enc, q_buf, q_weight, head_dim, hp.rms_eps, n_head)
        encode_rmsnorm_weighted(enc, k_buf, k_weight, head_dim, hp.rms_eps, n_head_kv)
        encode_rmsnorm_plain(enc, v_buf, head_dim, hp.rms_eps, n_head_kv)
        encode_rope(enc, q_buf, factor_buf, head_dim, rope_dim, pos, base, use_factors, n_head)
        encode_rope(enc, k_buf, factor_buf, head_dim, rope_dim, pos, base, use_factors, n_head_kv)
        enc.end_encoding
        cmd.commit
        cmd.wait
        true
      end

      def attention_context_from_projection(proj : Gemma4CPU::AttentionProjection,
                                            hp : Gemma4Hparams,
                                            il : Int32,
                                            pos : Int32,
                                            max_seq : Int32,
                                            k_cache : Array(Float32),
                                            v_cache : Array(Float32)) : NamedTuple(context: Array(Float32), k_cache: Array(Float32), v_cache: Array(Float32))?
        return nil unless available?

        n_head = hp.n_head
        n_head_kv = hp.n_head_kv(il)
        head_dim = hp.head_dim_for_layer(il)
        kv_dim = n_head_kv * head_dim
        q_dim = n_head * head_dim
        heads_per_group = n_head // n_head_kv
        raise ArgumentError.new("position #{pos} exceeds max_seq #{max_seq}") if pos < 0 || pos >= max_seq
        raise ArgumentError.new("q projection size mismatch at layer #{il}") unless proj.q.size == q_dim
        raise ArgumentError.new("k projection size mismatch at layer #{il}") unless proj.k.size == kv_dim
        raise ArgumentError.new("v projection size mismatch at layer #{il}") unless proj.v.size == kv_dim
        raise ArgumentError.new("k_cache size mismatch") unless k_cache.size == max_seq * kv_dim
        raise ArgumentError.new("v_cache size mismatch") unless v_cache.size == max_seq * kv_dim
        raise ArgumentError.new("unsupported head_dim #{head_dim}; max 512") if head_dim > 512

        start_pos = hp.attention_start_pos(il, pos)
        len = pos - start_pos + 1
        q_buf = ML::MetalBuffer.from_array(proj.q)
        k_buf = ML::MetalBuffer.from_array(proj.k)
        v_buf = ML::MetalBuffer.from_array(proj.v)
        k_cache_buf = ML::MetalBuffer.from_array(k_cache)
        v_cache_buf = ML::MetalBuffer.from_array(v_cache)
        out_buf = ML::MetalBuffer.new(q_dim.to_i64 * sizeof(Float32))

        cmd = ML::Metal::CommandBuffer.new
        enc = ML::Metal::ComputeEncoder.new(cmd)
        encode_kv_write_one(enc, k_buf, v_buf, k_cache_buf, v_cache_buf, pos, kv_dim)
        encode_attention_context_one(enc, q_buf, k_cache_buf, v_cache_buf, out_buf,
          start_pos, len, n_head, n_head_kv, head_dim, heads_per_group)
        enc.end_encoding
        cmd.commit
        cmd.wait

        {
          context: out_buf.read(q_dim),
          k_cache: k_cache_buf.read(k_cache.size),
          v_cache: v_cache_buf.read(v_cache.size),
        }
      end

      def attention_context_from_projection_resident(proj : Gemma4CPU::AttentionProjection,
                                                     hp : Gemma4Hparams,
                                                     il : Int32,
                                                     pos : Int32,
                                                     lstate : ResidentLayerState) : Array(Float32)?
        return nil unless available?

        n_head = hp.n_head
        n_head_kv = hp.n_head_kv(il)
        head_dim = hp.head_dim_for_layer(il)
        kv_dim = n_head_kv * head_dim
        q_dim = n_head * head_dim
        heads_per_group = n_head // n_head_kv
        raise ArgumentError.new("position #{pos} exceeds max_seq #{lstate.max_seq}") if pos < 0 || pos >= lstate.max_seq
        raise ArgumentError.new("q projection size mismatch at layer #{il}") unless proj.q.size == q_dim
        raise ArgumentError.new("k projection size mismatch at layer #{il}") unless proj.k.size == kv_dim
        raise ArgumentError.new("v projection size mismatch at layer #{il}") unless proj.v.size == kv_dim
        raise ArgumentError.new("resident kv_dim mismatch") unless lstate.kv_dim == kv_dim
        raise ArgumentError.new("unsupported head_dim #{head_dim}; max 512") if head_dim > 512

        start_pos = hp.attention_start_pos(il, pos)
        len = pos - start_pos + 1
        q_buf = ML::MetalBuffer.from_array(proj.q)
        k_buf = ML::MetalBuffer.from_array(proj.k)
        v_buf = ML::MetalBuffer.from_array(proj.v)
        out_buf = ML::MetalBuffer.new(q_dim.to_i64 * sizeof(Float32))

        cmd = ML::Metal::CommandBuffer.new
        enc = ML::Metal::ComputeEncoder.new(cmd)
        encode_kv_write_one(enc, k_buf, v_buf, lstate.k_cache_buf, lstate.v_cache_buf, pos, kv_dim)
        encode_attention_context_one(enc, q_buf, lstate.k_cache_buf, lstate.v_cache_buf, out_buf,
          start_pos, len, n_head, n_head_kv, head_dim, heads_per_group)
        enc.end_encoding
        cmd.commit
        cmd.wait

        out_buf.read(q_dim)
      end

      def attention_context_from_projection_resident_buffers(q_buf : ML::MetalBuffer,
                                                             k_buf : ML::MetalBuffer,
                                                             v_buf : ML::MetalBuffer,
                                                             hp : Gemma4Hparams,
                                                             il : Int32,
                                                             pos : Int32,
                                                             lstate : ResidentLayerState,
                                                             scratch : ResidentScratch? = nil) : ML::MetalBuffer?
        return nil unless available?

        n_head = hp.n_head
        n_head_kv = hp.n_head_kv(il)
        head_dim = hp.head_dim_for_layer(il)
        kv_dim = n_head_kv * head_dim
        q_dim = n_head * head_dim
        heads_per_group = n_head // n_head_kv
        raise ArgumentError.new("position #{pos} exceeds max_seq #{lstate.max_seq}") if pos < 0 || pos >= lstate.max_seq
        raise ArgumentError.new("resident kv_dim mismatch") unless lstate.kv_dim == kv_dim
        raise ArgumentError.new("q projection buffer too small at layer #{il}") if q_buf.size < q_dim.to_i64 * sizeof(Float32)
        raise ArgumentError.new("k projection buffer too small at layer #{il}") if k_buf.size < kv_dim.to_i64 * sizeof(Float32)
        raise ArgumentError.new("v projection buffer too small at layer #{il}") if v_buf.size < kv_dim.to_i64 * sizeof(Float32)
        raise ArgumentError.new("unsupported head_dim #{head_dim}; max 512") if head_dim > 512

        start_pos = hp.attention_start_pos(il, pos)
        len = pos - start_pos + 1
        out_buf = if scratch
                    scratch.not_nil!.get("attn.ctx", q_dim.to_i64 * sizeof(Float32))
                  else
                    ML::MetalBuffer.new(q_dim.to_i64 * sizeof(Float32))
                  end

        cmd = ML::Metal::CommandBuffer.new
        enc = ML::Metal::ComputeEncoder.new(cmd)
        encode_kv_write_one(enc, k_buf, v_buf, lstate.k_cache_buf, lstate.v_cache_buf, pos, kv_dim)
        encode_attention_context_one(enc, q_buf, lstate.k_cache_buf, lstate.v_cache_buf, out_buf,
          start_pos, len, n_head, n_head_kv, head_dim, heads_per_group)
        enc.end_encoding
        cmd.commit
        cmd.wait
        out_buf
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
        return false unless Qwen35Metal.encode_matmul_many_to_buffers(enc, [lw.ffn_gate_qw, lw.ffn_up_qw], ffn_in_buf, [gate_buf, up_buf], 1)
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
        cmd.commit
        cmd.wait
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
        encode_rmsnorm_rows_weighted_out(enc, attn_buf, post_attn_w, attn_normed_buf, hidden_dim, batch, hp.rms_eps)
        encode_add_vec(enc, x_buf, attn_normed_buf, attn_out_buf, batch * hidden_dim)
        encode_rmsnorm_rows_weighted_out(enc, attn_out_buf, ffn_w, ffn_in_buf, hidden_dim, batch, hp.rms_eps)
        fused_gelu = q4_gelu_fuse_enabled? &&
          Qwen35Metal.encode_q4k_gemm_h16_pair_b64_gelu_mul(enc, lw.ffn_gate_qw, lw.ffn_up_qw, ffn_in_buf, gate_buf, combined_buf, batch)
        unless fused_gelu
          unless Qwen35Metal.encode_matmul_many_to_buffers(enc, [lw.ffn_gate_qw, lw.ffn_up_qw], ffn_in_buf, [gate_buf, up_buf], batch)
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
        encode_rmsnorm_rows_weighted_out(enc, ffn_buf, post_ffw_w, ffn_normed_buf, hidden_dim, batch, hp.rms_eps)
        encode_add_scaled_vec(enc, attn_out_buf, ffn_normed_buf, out_buf, batch * hidden_dim, scale)
        enc.end_encoding
        cmd.commit
        cmd.wait
        out_buf.read(batch * hidden_dim)
      end

      def forward_layer_resident_cache_rows(weights : Gemma4Weights,
                                            il : Int32,
                                            x_rows : Array(Float32),
                                            base_pos : Int32,
                                            batch : Int32,
                                            state : ResidentState) : Array(Float32)?
        return nil unless available?

        hp = weights.hparams
        lw = weights.layers[il]
        hidden_dim = hp.n_embd
        head_dim = hp.head_dim_for_layer(il)
        rope_dim = hp.rope_dim_for_layer(il)
        q_dim = hp.n_head * head_dim
        kv_dim = hp.n_head_kv(il) * head_dim
        ffn_dim = lw.ffn_gate_qw.out_dim
        raise ArgumentError.new("batch must be positive") unless batch > 0
        raise ArgumentError.new("forward_layer_resident_cache_rows input size mismatch") unless x_rows.size == batch * hidden_dim
        raise ArgumentError.new("unsupported head_dim #{head_dim}; max 512") if head_dim > 512
        raise ArgumentError.new("rope_dim #{rope_dim} must be even") unless rope_dim.even?
        raise ArgumentError.new("base_pos #{base_pos} exceeds max_seq #{state.layers[il].max_seq}") if base_pos < 0 || base_pos + batch > state.layers[il].max_seq

        lstate = state.layers[il]
        raise ArgumentError.new("resident kv_dim mismatch") unless lstate.kv_dim == kv_dim

        x_buf = ML::MetalBuffer.from_array(x_rows)
        attn_norm_w = ML::MetalBuffer.from_array(lw.attn_norm)
        post_attn_w = ML::MetalBuffer.from_array(lw.post_attention_norm)
        ffn_w = ML::MetalBuffer.from_array(lw.ffn_norm)
        post_ffw_w = ML::MetalBuffer.from_array(lw.post_ffw_norm)
        q_weight = ML::MetalBuffer.from_array(lw.attn_q_norm)
        k_weight = ML::MetalBuffer.from_array(lw.attn_k_norm)
        rope_freqs = hp.full_attention?(il) ? weights.rope_freqs : nil
        factors = (hp.full_attention?(il) && rope_freqs) ? rope_freqs.not_nil! : [1.0_f32]
        factor_buf = ML::MetalBuffer.from_array(factors)
        use_factors = (hp.full_attention?(il) && rope_freqs) ? 1_u32 : 0_u32
        base = hp.rope_freq_base_for_layer(il)
        heads_per_group = hp.n_head // hp.n_head_kv(il)
        sliding_window = hp.sliding_window?(il) ? hp.sliding_window : 0

        x_norm_buf = ML::MetalBuffer.new(batch.to_i64 * hidden_dim * sizeof(Float32))
        q_buf = ML::MetalBuffer.new(batch.to_i64 * q_dim * sizeof(Float32))
        k_buf = ML::MetalBuffer.new(batch.to_i64 * kv_dim * sizeof(Float32))
        v_buf = ML::MetalBuffer.new(batch.to_i64 * kv_dim * sizeof(Float32))
        ctx_buf = ML::MetalBuffer.new(batch.to_i64 * q_dim * sizeof(Float32))
        attn_projected_buf = ML::MetalBuffer.new(batch.to_i64 * hidden_dim * sizeof(Float32))
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
        encode_rmsnorm_rows_weighted_out(enc, x_buf, attn_norm_w, x_norm_buf, hidden_dim, batch, hp.rms_eps)
        projected = if v_qw = lw.attn_v_qw
                      Qwen35Metal.encode_matmul_many_to_buffers(enc, [lw.attn_q_qw, lw.attn_k_qw, v_qw], x_norm_buf, [q_buf, k_buf, v_buf], batch)
                    else
                      Qwen35Metal.encode_matmul_many_to_buffers(enc, [lw.attn_q_qw, lw.attn_k_qw, lw.attn_k_qw], x_norm_buf, [q_buf, k_buf, v_buf], batch)
                    end
        unless projected
          enc.end_encoding
          return nil
        end

        encode_rmsnorm_weighted_rows(enc, q_buf, q_weight, head_dim, hp.rms_eps, hp.n_head, batch)
        encode_rmsnorm_weighted_rows(enc, k_buf, k_weight, head_dim, hp.rms_eps, hp.n_head_kv(il), batch)
        encode_rmsnorm_plain_rows(enc, v_buf, head_dim, hp.rms_eps, hp.n_head_kv(il), batch)
        encode_rope_rows(enc, q_buf, factor_buf, head_dim, rope_dim, base_pos, base, use_factors, hp.n_head, batch)
        encode_rope_rows(enc, k_buf, factor_buf, head_dim, rope_dim, base_pos, base, use_factors, hp.n_head_kv(il), batch)
        encode_kv_write_rows(enc, k_buf, v_buf, lstate.k_cache_buf, lstate.v_cache_buf, base_pos, kv_dim, batch)
        encode_attention_context_rows(enc, q_buf, lstate.k_cache_buf, lstate.v_cache_buf, ctx_buf,
          base_pos, batch, hp.n_head, hp.n_head_kv(il), head_dim, heads_per_group, sliding_window)
        unless Qwen35Metal.encode_matmul_to_buffer(enc, lw.attn_output_qw, ctx_buf, attn_projected_buf, batch)
          enc.end_encoding
          return nil
        end

        encode_rmsnorm_rows_weighted_out(enc, attn_projected_buf, post_attn_w, attn_normed_buf, hidden_dim, batch, hp.rms_eps)
        encode_add_vec(enc, x_buf, attn_normed_buf, attn_out_buf, batch * hidden_dim)
        encode_rmsnorm_rows_weighted_out(enc, attn_out_buf, ffn_w, ffn_in_buf, hidden_dim, batch, hp.rms_eps)
        fused_gelu = q4_gelu_fuse_enabled? &&
          Qwen35Metal.encode_q4k_gemm_h16_pair_b64_gelu_mul(enc, lw.ffn_gate_qw, lw.ffn_up_qw, ffn_in_buf, gate_buf, combined_buf, batch)
        unless fused_gelu
          unless Qwen35Metal.encode_matmul_many_to_buffers(enc, [lw.ffn_gate_qw, lw.ffn_up_qw], ffn_in_buf, [gate_buf, up_buf], batch)
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
        encode_rmsnorm_rows_weighted_out(enc, ffn_buf, post_ffw_w, ffn_normed_buf, hidden_dim, batch, hp.rms_eps)
        encode_add_scaled_vec(enc, attn_out_buf, ffn_normed_buf, out_buf, batch * hidden_dim, scale)
        enc.end_encoding
        cmd.commit
        cmd.wait
        out_buf.read(batch * hidden_dim)
      end

      def prefill_tokens_last_hidden_resident_rows(weights : Gemma4Weights,
                                                   token_ids : Array(Int32),
                                                   start_pos : Int32,
                                                   state : ResidentState,
                                                   chunk_size : Int32 = 8,
                                                   stop_layer : Int32? = nil) : Array(Float32)?
        return nil unless available?
        raise ArgumentError.new("prefill token_ids must not be empty") if token_ids.empty?
        raise ArgumentError.new("prefill start_pos must be non-negative") if start_pos < 0
        raise ArgumentError.new("prefill chunk_size must be positive") unless chunk_size > 0

        hp = weights.hparams
        hidden_dim = hp.n_embd
        exact_chunk_cap = ENV["GEMMA4_ROW_PREFILL_EXACT_CHUNK_MAX"]?.try(&.to_i?) || 8
        exact_chunk_cap = 8 if exact_chunk_cap <= 0
        unless ENV["GEMMA4_ROW_PREFILL_ALLOW_GEMM"]? == "1"
          qwen_gemm_threshold = ENV["QWEN35_GEMM_BATCH_THRESHOLD"]?.try(&.to_i?) || 8
          qwen_gemm_threshold = 8 if qwen_gemm_threshold <= 0
          exact_chunk_cap = Math.min(exact_chunk_cap, qwen_gemm_threshold)
        end
        exact_chunk = Math.min(chunk_size, exact_chunk_cap)
        layer_count = stop_layer ? Math.min(stop_layer.not_nil!, weights.layers.size) : weights.layers.size
        scale = Math.sqrt(hidden_dim.to_f64).to_f32
        last_hidden = [] of Float32

        offset = 0
        while offset < token_ids.size
          batch = Math.min(exact_chunk, token_ids.size - offset)
          base_pos = start_pos + offset
          raise ArgumentError.new("prefill chunk exceeds max_seq") if base_pos + batch > state.max_seq

          x_rows = [] of Float32
          batch.times do |r|
            x = Qwen35Metal.embedding_q6k_from_token_id(weights.token_embd, token_ids[offset + r])
            return nil unless x
            x.size.times { |i| x[i] *= scale }
            x_rows.concat(x)
          end

          layer_count.times do |il|
            next_rows = forward_layer_resident_cache_rows(weights, il, x_rows, base_pos, batch, state)
            return nil unless next_rows
            x_rows = next_rows
          end

          last_hidden = x_rows[((batch - 1) * hidden_dim)...(batch * hidden_dim)].to_a
          offset += batch
        end

        last_hidden
      end

      def forward_layer(weights : Gemma4Weights,
                        il : Int32,
                        x : Array(Float32),
                        pos : Int32,
                        max_seq : Int32,
                        k_cache : Array(Float32),
                        v_cache : Array(Float32)) : NamedTuple(out: Array(Float32), k_cache: Array(Float32), v_cache: Array(Float32))?
        return nil unless available?

        hp = weights.hparams
        lw = weights.layers[il]
        hidden_dim = hp.n_embd
        head_dim = hp.head_dim_for_layer(il)
        kv_dim = hp.n_head_kv(il) * head_dim
        raise ArgumentError.new("forward_layer input size mismatch") unless x.size == hidden_dim
        raise ArgumentError.new("forward_layer k_cache size mismatch") unless k_cache.size == max_seq * kv_dim
        raise ArgumentError.new("forward_layer v_cache size mismatch") unless v_cache.size == max_seq * kv_dim

        x_norm = rms_norm(x, lw.attn_norm, hp.rms_eps).not_nil!
        raw = if v_qw = lw.attn_v_qw
                Qwen35Metal.matmul_many([lw.attn_q_qw, lw.attn_k_qw, v_qw], x_norm)
              else
                Qwen35Metal.matmul_many([lw.attn_q_qw, lw.attn_k_qw], x_norm)
              end
        return nil unless raw

        pre = if raw.size == 3
                Gemma4CPU::AttentionProjection.new(raw[0], raw[1], raw[2], false)
              else
                Gemma4CPU::AttentionProjection.new(raw[0], raw[1], raw[1].dup, true)
              end
        proj = normalize_and_rope_projection(pre, lw, hp, il, pos, hp.full_attention?(il) ? weights.rope_freqs : nil).not_nil!
        attn = attention_context_from_projection(proj, hp, il, pos, max_seq, k_cache, v_cache).not_nil!
        attn_projected = Qwen35Metal.matmul(lw.attn_output_qw, attn[:context], 1)
        return nil unless attn_projected
        out = layer_tail(x, attn_projected, lw, hp)
        return nil unless out

        {
          out: out,
          k_cache: attn[:k_cache],
          v_cache: attn[:v_cache],
        }
      end

      def forward_layer_resident_cache(weights : Gemma4Weights,
                                       il : Int32,
                                       x : Array(Float32),
                                       pos : Int32,
                                       state : ResidentState) : Array(Float32)?
        return nil unless available?

        hp = weights.hparams
        hidden_dim = hp.n_embd
        raise ArgumentError.new("forward_layer_resident_cache input size mismatch") unless x.size == hidden_dim

        x_buf = ML::MetalBuffer.from_array(x)
        out_buf = forward_layer_resident_cache_buf(weights, il, x_buf, pos, state)
        return nil unless out_buf
        out_buf.read(hidden_dim)
      rescue ex
        if ENV["GEMMA4_RESIDENT_LAYER_STRICT"]? == "1"
          raise ex
        else
          hp_fallback = weights.hparams
          lw_fallback = weights.layers[il]
          x_norm = rms_norm(x, lw_fallback.attn_norm, hp_fallback.rms_eps).not_nil!
          raw = if v_qw = lw_fallback.attn_v_qw
                  Qwen35Metal.matmul_many([lw_fallback.attn_q_qw, lw_fallback.attn_k_qw, v_qw], x_norm)
                else
                  Qwen35Metal.matmul_many([lw_fallback.attn_q_qw, lw_fallback.attn_k_qw], x_norm)
                end
          return nil unless raw

          pre = if raw.size == 3
                  Gemma4CPU::AttentionProjection.new(raw[0], raw[1], raw[2], false)
                else
                  Gemma4CPU::AttentionProjection.new(raw[0], raw[1], raw[1].dup, true)
                end
          proj = normalize_and_rope_projection(pre, lw_fallback, hp_fallback, il, pos, hp_fallback.full_attention?(il) ? weights.rope_freqs : nil).not_nil!
          ctx = attention_context_from_projection_resident(proj, hp_fallback, il, pos, state.layers[il]).not_nil!
          attn_projected = Qwen35Metal.matmul(lw_fallback.attn_output_qw, ctx, 1)
          return nil unless attn_projected
          layer_tail_resident_buffers(x, attn_projected, lw_fallback, hp_fallback)
        end
      end

      def forward_layer_resident_cache_buf(weights : Gemma4Weights,
                                           il : Int32,
                                           x_buf : ML::MetalBuffer,
                                           pos : Int32,
                                           state : ResidentState,
                                           scratch : ResidentScratch? = state.scratch) : ML::MetalBuffer?
        return nil unless available?

        hp = weights.hparams
        lw = weights.layers[il]
        hidden_dim = hp.n_embd
        head_dim = hp.head_dim_for_layer(il)
        rope_dim = hp.rope_dim_for_layer(il)
        q_dim = hp.n_head * head_dim
        kv_dim = hp.n_head_kv(il) * head_dim
        raise ArgumentError.new("forward_layer_resident_cache input buffer too small") if x_buf.size < hidden_dim.to_i64 * sizeof(Float32)
        raise ArgumentError.new("unsupported head_dim #{head_dim}; max 512") if head_dim > 512
        raise ArgumentError.new("rope_dim #{rope_dim} must be even") unless rope_dim.even?
        raise ArgumentError.new("position #{pos} exceeds max_seq #{state.layers[il].max_seq}") if pos < 0 || pos >= state.layers[il].max_seq

        attn_norm_w = ML::MetalBuffer.from_array(lw.attn_norm)
        x_norm_buf = scratch ? scratch.not_nil!.get("layer.x_norm", hidden_dim.to_i64 * sizeof(Float32)) : ML::MetalBuffer.new(hidden_dim.to_i64 * sizeof(Float32))

        q_buf = scratch ? scratch.not_nil!.get("layer.q", q_dim.to_i64 * sizeof(Float32)) : ML::MetalBuffer.new(q_dim.to_i64 * sizeof(Float32))
        k_buf = scratch ? scratch.not_nil!.get("layer.k", kv_dim.to_i64 * sizeof(Float32)) : ML::MetalBuffer.new(kv_dim.to_i64 * sizeof(Float32))
        v_buf = scratch ? scratch.not_nil!.get("layer.v", kv_dim.to_i64 * sizeof(Float32)) : ML::MetalBuffer.new(kv_dim.to_i64 * sizeof(Float32))
        ctx_buf = scratch ? scratch.not_nil!.get("attn.ctx", q_dim.to_i64 * sizeof(Float32)) : ML::MetalBuffer.new(q_dim.to_i64 * sizeof(Float32))
        attn_projected_buf = scratch ? scratch.not_nil!.get("layer.attn_projected", hidden_dim.to_i64 * sizeof(Float32)) : ML::MetalBuffer.new(hidden_dim.to_i64 * sizeof(Float32))
        out_buf = ML::MetalBuffer.new(hidden_dim.to_i64 * sizeof(Float32))

        q_weight = ML::MetalBuffer.from_array(lw.attn_q_norm)
        k_weight = ML::MetalBuffer.from_array(lw.attn_k_norm)
        rope_freqs = hp.full_attention?(il) ? weights.rope_freqs : nil
        factors = (hp.full_attention?(il) && rope_freqs) ? rope_freqs.not_nil! : [1.0_f32]
        factor_buf = ML::MetalBuffer.from_array(factors)
        use_factors = (hp.full_attention?(il) && rope_freqs) ? 1_u32 : 0_u32
        base = hp.rope_freq_base_for_layer(il)
        heads_per_group = hp.n_head // hp.n_head_kv(il)
        start_pos = hp.attention_start_pos(il, pos)
        attn_len = pos - start_pos + 1
        lstate = state.layers[il]

        cmd = ML::Metal::CommandBuffer.new
        enc = ML::Metal::ComputeEncoder.new(cmd)
        encode_rmsnorm_weighted_out(enc, x_buf, attn_norm_w, x_norm_buf, hidden_dim, hp.rms_eps)
        projected = if v_qw = lw.attn_v_qw
                      Qwen35Metal.encode_matmul_many_to_buffers(enc, [lw.attn_q_qw, lw.attn_k_qw, v_qw], x_norm_buf, [q_buf, k_buf, v_buf], 1)
                    else
                      # Full-attention Gemma4 layers reuse pre-normalized K as V.
                      # Project K twice so K can receive learned K-norm+RoPE while V
                      # receives plain RMSNorm and no RoPE.
                      Qwen35Metal.encode_matmul_many_to_buffers(enc, [lw.attn_q_qw, lw.attn_k_qw, lw.attn_k_qw], x_norm_buf, [q_buf, k_buf, v_buf], 1)
                    end
        unless projected
          enc.end_encoding
          return nil
        end
        encode_rmsnorm_weighted(enc, q_buf, q_weight, head_dim, hp.rms_eps, hp.n_head)
        encode_rmsnorm_weighted(enc, k_buf, k_weight, head_dim, hp.rms_eps, hp.n_head_kv(il))
        encode_rmsnorm_plain(enc, v_buf, head_dim, hp.rms_eps, hp.n_head_kv(il))
        encode_rope(enc, q_buf, factor_buf, head_dim, rope_dim, pos, base, use_factors, hp.n_head)
        encode_rope(enc, k_buf, factor_buf, head_dim, rope_dim, pos, base, use_factors, hp.n_head_kv(il))
        encode_kv_write_one(enc, k_buf, v_buf, lstate.k_cache_buf, lstate.v_cache_buf, pos, kv_dim)
        encode_attention_context_one(enc, q_buf, lstate.k_cache_buf, lstate.v_cache_buf, ctx_buf,
          start_pos, attn_len, hp.n_head, hp.n_head_kv(il), head_dim, heads_per_group)
        unless Qwen35Metal.encode_matmul_to_buffer(enc, lw.attn_output_qw, ctx_buf, attn_projected_buf, 1)
          enc.end_encoding
          return nil
        end
        unless encode_layer_tail_resident_buffer_inputs(enc, x_buf, attn_projected_buf, out_buf, lw, hp, scratch)
          enc.end_encoding
          return nil
        end
        enc.end_encoding
        cmd.commit
        cmd.wait
        out_buf
      end

      def forward_logits_from_hidden(weights : Gemma4Weights,
                                     hidden : Array(Float32)) : Array(Float32)?
        return nil unless available?

        hp = weights.hparams
        raise ArgumentError.new("forward_logits_from_hidden hidden size mismatch") unless hidden.size == hp.n_embd
        x = rms_norm(hidden, weights.output_norm, hp.rms_eps).not_nil!
        logits = Qwen35Metal.matmul(weights.token_embd, x, 1)
        return nil unless logits
        softcap!(logits, hp.final_logit_softcapping)
        logits
      end

      def forward_hidden(weights : Gemma4Weights,
                         token_id : Int32,
                         pos : Int32,
                         state : State,
                         stop_layer : Int32? = nil) : Array(Float32)?
        hp = weights.hparams
        x = Qwen35Metal.embedding_q6k_from_token_id(weights.token_embd, token_id)
        return nil unless x
        scale = Math.sqrt(hp.n_embd.to_f64).to_f32
        x.size.times { |i| x[i] *= scale }

        layer_count = stop_layer ? Math.min(stop_layer.not_nil!, weights.layers.size) : weights.layers.size
        layer_count.times do |il|
          lstate = state.layers[il]
          result = forward_layer(weights, il, x, pos, state.max_seq, lstate.k_cache, lstate.v_cache)
          return nil unless result
          x = result[:out]
          lstate.k_cache = result[:k_cache]
          lstate.v_cache = result[:v_cache]
        end
        x
      end

      def forward_hidden_resident_cache(weights : Gemma4Weights,
                                        token_id : Int32,
                                        pos : Int32,
                                        state : ResidentState,
                                        stop_layer : Int32? = nil) : Array(Float32)?
        hp = weights.hparams
        x = Qwen35Metal.embedding_q6k_from_token_id(weights.token_embd, token_id)
        return nil unless x
        scale = Math.sqrt(hp.n_embd.to_f64).to_f32
        x.size.times { |i| x[i] *= scale }

        x_buf = ML::MetalBuffer.from_array(x)
        layer_count = stop_layer ? Math.min(stop_layer.not_nil!, weights.layers.size) : weights.layers.size
        layer_count.times do |il|
          next_buf = forward_layer_resident_cache_buf(weights, il, x_buf, pos, state)
          return nil unless next_buf
          x_buf = next_buf
        end
        x_buf.read(hp.n_embd)
      rescue ex
        if ENV["GEMMA4_RESIDENT_LAYER_STRICT"]? == "1"
          raise ex
        else
          hp_fallback = weights.hparams
          x_fallback = Qwen35Metal.embedding_q6k_from_token_id(weights.token_embd, token_id)
          return nil unless x_fallback
          scale = Math.sqrt(hp_fallback.n_embd.to_f64).to_f32
          x_fallback.size.times { |i| x_fallback[i] *= scale }

          layer_count = stop_layer ? Math.min(stop_layer.not_nil!, weights.layers.size) : weights.layers.size
          layer_count.times do |il|
            x_fallback = forward_layer_resident_cache(weights, il, x_fallback, pos, state)
            return nil unless x_fallback
          end
          x_fallback
        end
      end

      def rms_norm(x : Array(Float32), weight : Array(Float32), eps : Float32) : Array(Float32)?
        return nil unless available?
        raise ArgumentError.new("rms_norm weight size mismatch") unless x.size == weight.size

        x_buf = ML::MetalBuffer.from_array(x)
        w_buf = ML::MetalBuffer.from_array(weight)
        out_buf = ML::MetalBuffer.new(x.size.to_i64 * sizeof(Float32))
        cmd = ML::Metal::CommandBuffer.new
        enc = ML::Metal::ComputeEncoder.new(cmd)
        encode_rmsnorm_weighted_out(enc, x_buf, w_buf, out_buf, x.size, eps)
        enc.end_encoding
        cmd.commit
        cmd.wait
        out_buf.read(x.size)
      end

      def add_vec(a : Array(Float32), b : Array(Float32)) : Array(Float32)?
        return nil unless available?
        raise ArgumentError.new("add_vec size mismatch") unless a.size == b.size

        a_buf = ML::MetalBuffer.from_array(a)
        b_buf = ML::MetalBuffer.from_array(b)
        out_buf = ML::MetalBuffer.new(a.size.to_i64 * sizeof(Float32))
        cmd = ML::Metal::CommandBuffer.new
        enc = ML::Metal::ComputeEncoder.new(cmd)
        encode_add_vec(enc, a_buf, b_buf, out_buf, a.size)
        enc.end_encoding
        cmd.commit
        cmd.wait
        out_buf.read(a.size)
      end

      def add_scaled_vec(a : Array(Float32), b : Array(Float32), scale : Float32) : Array(Float32)?
        return nil unless available?
        raise ArgumentError.new("add_scaled_vec size mismatch") unless a.size == b.size

        a_buf = ML::MetalBuffer.from_array(a)
        b_buf = ML::MetalBuffer.from_array(b)
        out_buf = ML::MetalBuffer.new(a.size.to_i64 * sizeof(Float32))
        cmd = ML::Metal::CommandBuffer.new
        enc = ML::Metal::ComputeEncoder.new(cmd)
        encode_add_scaled_vec(enc, a_buf, b_buf, out_buf, a.size, scale)
        enc.end_encoding
        cmd.commit
        cmd.wait
        out_buf.read(a.size)
      end

      def gelu_mul(gate : Array(Float32), up : Array(Float32)) : Array(Float32)?
        return nil unless available?
        raise ArgumentError.new("gelu_mul size mismatch") unless gate.size == up.size

        gate_buf = ML::MetalBuffer.from_array(gate)
        up_buf = ML::MetalBuffer.from_array(up)
        out_buf = ML::MetalBuffer.new(gate.size.to_i64 * sizeof(Float32))
        cmd = ML::Metal::CommandBuffer.new
        enc = ML::Metal::ComputeEncoder.new(cmd)
        encode_gelu_mul(enc, gate_buf, up_buf, out_buf, gate.size)
        enc.end_encoding
        cmd.commit
        cmd.wait
        out_buf.read(gate.size)
      end

      def softcap!(x : Array(Float32), cap : Float32) : Nil
        return if cap <= 0.0_f32
        return unless available?

        x_buf = ML::MetalBuffer.from_array(x)
        cmd = ML::Metal::CommandBuffer.new
        enc = ML::Metal::ComputeEncoder.new(cmd)
        encode_softcap(enc, x_buf, x.size, cap)
        enc.end_encoding
        cmd.commit
        cmd.wait
        updated = x_buf.read(x.size)
        x.size.times { |i| x[i] = updated[i] }
      end

      private def encode_rmsnorm_weighted(enc : ML::Metal::ComputeEncoder,
                                          x_buf : ML::MetalBuffer,
                                          weight_buf : ML::MetalBuffer,
                                          head_dim : Int32,
                                          eps : Float32,
                                          n_heads : Int32) : Nil
        enc.set_pipeline(rmsnorm_weighted_pipeline)
        enc.set_buffer(x_buf, 0, ML::Metal::BufferAccess::ReadWrite)
        enc.set_buffer(weight_buf, 1)
        enc.set_value(head_dim.to_u32, 2)
        enc.set_value(eps, 3)
        enc.dispatch_threadgroups({n_heads, 1, 1}, {32, 1, 1})
      end

      private def encode_rmsnorm_weighted_out(enc : ML::Metal::ComputeEncoder,
                                              x_buf : ML::MetalBuffer,
                                              weight_buf : ML::MetalBuffer,
                                              out_buf : ML::MetalBuffer,
                                              count : Int32,
                                              eps : Float32) : Nil
        enc.set_pipeline(rmsnorm_vec_weighted_pipeline)
        enc.set_buffer(x_buf, 0)
        enc.set_buffer(weight_buf, 1)
        enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
        enc.set_value(count.to_u32, 3)
        enc.set_value(eps, 4)
        enc.dispatch_threadgroups({1, 1, 1}, {256, 1, 1})
      end

      private def encode_rmsnorm_rows_weighted_out(enc : ML::Metal::ComputeEncoder,
                                                   x_buf : ML::MetalBuffer,
                                                   weight_buf : ML::MetalBuffer,
                                                   out_buf : ML::MetalBuffer,
                                                   row_dim : Int32,
                                                   rows : Int32,
                                                   eps : Float32) : Nil
        enc.set_pipeline(rmsnorm_rows_weighted_pipeline)
        enc.set_buffer(x_buf, 0)
        enc.set_buffer(weight_buf, 1)
        enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
        enc.set_value(row_dim.to_u32, 3)
        enc.set_value(eps, 4)
        enc.dispatch_threadgroups({rows, 1, 1}, {256, 1, 1})
      end

      private def encode_rmsnorm_weighted_rows(enc : ML::Metal::ComputeEncoder,
                                               x_buf : ML::MetalBuffer,
                                               weight_buf : ML::MetalBuffer,
                                               head_dim : Int32,
                                               eps : Float32,
                                               n_heads : Int32,
                                               rows : Int32) : Nil
        enc.set_pipeline(rmsnorm_heads_weighted_rows_pipeline)
        enc.set_buffer(x_buf, 0, ML::Metal::BufferAccess::ReadWrite)
        enc.set_buffer(weight_buf, 1)
        enc.set_value(head_dim.to_u32, 2)
        enc.set_value(eps, 3)
        enc.set_value(n_heads.to_u32, 4)
        enc.set_value(rows.to_u32, 5)
        enc.dispatch_threadgroups({n_heads, rows, 1}, {32, 1, 1})
      end

      private def encode_rmsnorm_plain(enc : ML::Metal::ComputeEncoder,
                                       x_buf : ML::MetalBuffer,
                                       head_dim : Int32,
                                       eps : Float32,
                                       n_heads : Int32) : Nil
        enc.set_pipeline(rmsnorm_plain_pipeline)
        enc.set_buffer(x_buf, 0, ML::Metal::BufferAccess::ReadWrite)
        enc.set_value(head_dim.to_u32, 1)
        enc.set_value(eps, 2)
        enc.dispatch_threadgroups({n_heads, 1, 1}, {32, 1, 1})
      end

      private def encode_rmsnorm_plain_rows(enc : ML::Metal::ComputeEncoder,
                                            x_buf : ML::MetalBuffer,
                                            head_dim : Int32,
                                            eps : Float32,
                                            n_heads : Int32,
                                            rows : Int32) : Nil
        enc.set_pipeline(rmsnorm_heads_plain_rows_pipeline)
        enc.set_buffer(x_buf, 0, ML::Metal::BufferAccess::ReadWrite)
        enc.set_value(head_dim.to_u32, 1)
        enc.set_value(eps, 2)
        enc.set_value(n_heads.to_u32, 3)
        enc.set_value(rows.to_u32, 4)
        enc.dispatch_threadgroups({n_heads, rows, 1}, {32, 1, 1})
      end

      private def encode_rope(enc : ML::Metal::ComputeEncoder,
                              x_buf : ML::MetalBuffer,
                              factors_buf : ML::MetalBuffer,
                              head_dim : Int32,
                              rope_dim : Int32,
                              pos : Int32,
                              freq_base : Float32,
                              use_factors : UInt32,
                              n_heads : Int32) : Nil
        enc.set_pipeline(rope_pipeline)
        enc.set_buffer(x_buf, 0, ML::Metal::BufferAccess::ReadWrite)
        enc.set_buffer(factors_buf, 1)
        enc.set_value(head_dim.to_u32, 2)
        enc.set_value(rope_dim.to_u32, 3)
        enc.set_value(pos.to_u32, 4)
        enc.set_value(freq_base, 5)
        enc.set_value(use_factors, 6)
        enc.dispatch_threadgroups({n_heads, 1, 1}, {32, 1, 1})
      end

      private def encode_rope_rows(enc : ML::Metal::ComputeEncoder,
                                   x_buf : ML::MetalBuffer,
                                   factors_buf : ML::MetalBuffer,
                                   head_dim : Int32,
                                   rope_dim : Int32,
                                   base_pos : Int32,
                                   freq_base : Float32,
                                   use_factors : UInt32,
                                   n_heads : Int32,
                                   rows : Int32) : Nil
        enc.set_pipeline(rope_rows_pipeline)
        enc.set_buffer(x_buf, 0, ML::Metal::BufferAccess::ReadWrite)
        enc.set_buffer(factors_buf, 1)
        enc.set_value(head_dim.to_u32, 2)
        enc.set_value(rope_dim.to_u32, 3)
        enc.set_value(base_pos.to_u32, 4)
        enc.set_value(freq_base, 5)
        enc.set_value(use_factors, 6)
        enc.set_value(n_heads.to_u32, 7)
        enc.set_value(rows.to_u32, 8)
        enc.dispatch_threadgroups({n_heads, rows, 1}, {32, 1, 1})
      end

      private def encode_kv_write_one(enc : ML::Metal::ComputeEncoder,
                                      k_buf : ML::MetalBuffer,
                                      v_buf : ML::MetalBuffer,
                                      k_cache_buf : ML::MetalBuffer,
                                      v_cache_buf : ML::MetalBuffer,
                                      pos : Int32,
                                      kv_dim : Int32) : Nil
        enc.set_pipeline(kv_write_pipeline)
        enc.set_buffer(k_buf, 0)
        enc.set_buffer(v_buf, 1)
        enc.set_buffer(k_cache_buf, 2, ML::Metal::BufferAccess::ReadWrite)
        enc.set_buffer(v_cache_buf, 3, ML::Metal::BufferAccess::ReadWrite)
        enc.set_value(pos.to_u32, 4)
        enc.set_value(kv_dim.to_u32, 5)
        enc.dispatch_1d(kv_dim, 256)
      end

      private def encode_kv_write_rows(enc : ML::Metal::ComputeEncoder,
                                       k_buf : ML::MetalBuffer,
                                       v_buf : ML::MetalBuffer,
                                       k_cache_buf : ML::MetalBuffer,
                                       v_cache_buf : ML::MetalBuffer,
                                       base_pos : Int32,
                                       kv_dim : Int32,
                                       rows : Int32) : Nil
        enc.set_pipeline(kv_write_rows_pipeline)
        enc.set_buffer(k_buf, 0)
        enc.set_buffer(v_buf, 1)
        enc.set_buffer(k_cache_buf, 2, ML::Metal::BufferAccess::ReadWrite)
        enc.set_buffer(v_cache_buf, 3, ML::Metal::BufferAccess::ReadWrite)
        enc.set_value(base_pos.to_u32, 4)
        enc.set_value(kv_dim.to_u32, 5)
        enc.set_value(rows.to_u32, 6)
        enc.dispatch_1d(rows * kv_dim, 256)
      end

      private def encode_attention_context_one(enc : ML::Metal::ComputeEncoder,
                                               q_buf : ML::MetalBuffer,
                                               k_cache_buf : ML::MetalBuffer,
                                               v_cache_buf : ML::MetalBuffer,
                                               out_buf : ML::MetalBuffer,
                                               start_pos : Int32,
                                               len : Int32,
                                               n_head : Int32,
                                               n_head_kv : Int32,
                                               head_dim : Int32,
                                               heads_per_group : Int32) : Nil
        enc.set_pipeline(attn_context_pipeline)
        enc.set_buffer(q_buf, 0)
        enc.set_buffer(k_cache_buf, 1)
        enc.set_buffer(v_cache_buf, 2)
        enc.set_buffer(out_buf, 3, ML::Metal::BufferAccess::Write)
        enc.set_value(start_pos.to_u32, 4)
        enc.set_value(len.to_u32, 5)
        enc.set_value(n_head.to_u32, 6)
        enc.set_value(n_head_kv.to_u32, 7)
        enc.set_value(head_dim.to_u32, 8)
        enc.set_value(heads_per_group.to_u32, 9)
        enc.dispatch_threadgroups({n_head, 1, 1}, {32, 1, 1})
      end

      private def encode_attention_context_rows(enc : ML::Metal::ComputeEncoder,
                                                q_buf : ML::MetalBuffer,
                                                k_cache_buf : ML::MetalBuffer,
                                                v_cache_buf : ML::MetalBuffer,
                                                out_buf : ML::MetalBuffer,
                                                base_pos : Int32,
                                                rows : Int32,
                                                n_head : Int32,
                                                n_head_kv : Int32,
                                                head_dim : Int32,
                                                heads_per_group : Int32,
                                                sliding_window : Int32) : Nil
        enc.set_pipeline(attn_context_rows_pipeline)
        enc.set_buffer(q_buf, 0)
        enc.set_buffer(k_cache_buf, 1)
        enc.set_buffer(v_cache_buf, 2)
        enc.set_buffer(out_buf, 3, ML::Metal::BufferAccess::Write)
        enc.set_value(base_pos.to_u32, 4)
        enc.set_value(rows.to_u32, 5)
        enc.set_value(n_head.to_u32, 6)
        enc.set_value(n_head_kv.to_u32, 7)
        enc.set_value(head_dim.to_u32, 8)
        enc.set_value(heads_per_group.to_u32, 9)
        enc.set_value(sliding_window.to_u32, 10)
        enc.dispatch_threadgroups({n_head, rows, 1}, {32, 1, 1})
      end

      private def encode_add_vec(enc : ML::Metal::ComputeEncoder,
                                 a_buf : ML::MetalBuffer,
                                 b_buf : ML::MetalBuffer,
                                 out_buf : ML::MetalBuffer,
                                 count : Int32) : Nil
        enc.set_pipeline(add_vec_pipeline)
        enc.set_buffer(a_buf, 0)
        enc.set_buffer(b_buf, 1)
        enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
        enc.set_value(count.to_u32, 3)
        enc.dispatch_1d(count, 256)
      end

      private def encode_add_scaled_vec(enc : ML::Metal::ComputeEncoder,
                                        a_buf : ML::MetalBuffer,
                                        b_buf : ML::MetalBuffer,
                                        out_buf : ML::MetalBuffer,
                                        count : Int32,
                                        scale : Float32) : Nil
        enc.set_pipeline(add_scaled_vec_pipeline)
        enc.set_buffer(a_buf, 0)
        enc.set_buffer(b_buf, 1)
        enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
        enc.set_value(count.to_u32, 3)
        enc.set_value(scale, 4)
        enc.dispatch_1d(count, 256)
      end

      private def encode_gelu_mul(enc : ML::Metal::ComputeEncoder,
                                  gate_buf : ML::MetalBuffer,
                                  up_buf : ML::MetalBuffer,
                                  out_buf : ML::MetalBuffer,
                                  count : Int32) : Nil
        enc.set_pipeline(gelu_mul_pipeline)
        enc.set_buffer(gate_buf, 0)
        enc.set_buffer(up_buf, 1)
        enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
        enc.set_value(count.to_u32, 3)
        enc.dispatch_1d(count, 256)
      end

      private def encode_softcap(enc : ML::Metal::ComputeEncoder,
                                 x_buf : ML::MetalBuffer,
                                 count : Int32,
                                 cap : Float32) : Nil
        enc.set_pipeline(softcap_pipeline)
        enc.set_buffer(x_buf, 0, ML::Metal::BufferAccess::ReadWrite)
        enc.set_value(count.to_u32, 1)
        enc.set_value(cap, 2)
        enc.dispatch_1d(count, 256)
      end

      private def rmsnorm_weighted_pipeline : ML::Metal::ComputePipeline
        @@rmsnorm_weighted_pipeline ||= ML::Metal::PipelineCache.get("gemma4_rmsnorm_heads_weighted") {
          ML::Metal::ComputePipeline.new("gemma4_rmsnorm_heads_weighted", GEMMA4_SOURCE)
        }
      end

      private def rmsnorm_vec_weighted_pipeline : ML::Metal::ComputePipeline
        @@rmsnorm_vec_weighted_pipeline ||= ML::Metal::PipelineCache.get("gemma4_rmsnorm_vec_weighted") {
          ML::Metal::ComputePipeline.new("gemma4_rmsnorm_vec_weighted", GEMMA4_SOURCE)
        }
      end

      private def rmsnorm_rows_weighted_pipeline : ML::Metal::ComputePipeline
        @@rmsnorm_rows_weighted_pipeline ||= ML::Metal::PipelineCache.get("gemma4_rmsnorm_rows_weighted") {
          ML::Metal::ComputePipeline.new("gemma4_rmsnorm_rows_weighted", GEMMA4_SOURCE)
        }
      end

      private def rmsnorm_plain_pipeline : ML::Metal::ComputePipeline
        @@rmsnorm_plain_pipeline ||= ML::Metal::PipelineCache.get("gemma4_rmsnorm_heads_plain") {
          ML::Metal::ComputePipeline.new("gemma4_rmsnorm_heads_plain", GEMMA4_SOURCE)
        }
      end

      private def rmsnorm_heads_weighted_rows_pipeline : ML::Metal::ComputePipeline
        @@rmsnorm_heads_weighted_rows_pipeline ||= ML::Metal::PipelineCache.get("gemma4_rmsnorm_heads_weighted_rows") {
          ML::Metal::ComputePipeline.new("gemma4_rmsnorm_heads_weighted_rows", GEMMA4_SOURCE)
        }
      end

      private def rmsnorm_heads_plain_rows_pipeline : ML::Metal::ComputePipeline
        @@rmsnorm_heads_plain_rows_pipeline ||= ML::Metal::PipelineCache.get("gemma4_rmsnorm_heads_plain_rows") {
          ML::Metal::ComputePipeline.new("gemma4_rmsnorm_heads_plain_rows", GEMMA4_SOURCE)
        }
      end

      private def rope_pipeline : ML::Metal::ComputePipeline
        @@rope_pipeline ||= ML::Metal::PipelineCache.get("gemma4_rope_neox") {
          ML::Metal::ComputePipeline.new("gemma4_rope_neox", GEMMA4_SOURCE)
        }
      end

      private def rope_rows_pipeline : ML::Metal::ComputePipeline
        @@rope_rows_pipeline ||= ML::Metal::PipelineCache.get("gemma4_rope_neox_rows") {
          ML::Metal::ComputePipeline.new("gemma4_rope_neox_rows", GEMMA4_SOURCE)
        }
      end

      private def kv_write_pipeline : ML::Metal::ComputePipeline
        @@kv_write_pipeline ||= ML::Metal::PipelineCache.get("gemma4_kv_write_one") {
          ML::Metal::ComputePipeline.new("gemma4_kv_write_one", GEMMA4_SOURCE)
        }
      end

      private def kv_write_rows_pipeline : ML::Metal::ComputePipeline
        @@kv_write_rows_pipeline ||= ML::Metal::PipelineCache.get("gemma4_kv_write_rows") {
          ML::Metal::ComputePipeline.new("gemma4_kv_write_rows", GEMMA4_SOURCE)
        }
      end

      private def attn_context_pipeline : ML::Metal::ComputePipeline
        @@attn_context_pipeline ||= ML::Metal::PipelineCache.get("gemma4_attn_context_one") {
          ML::Metal::ComputePipeline.new("gemma4_attn_context_one", GEMMA4_SOURCE)
        }
      end

      private def attn_context_rows_pipeline : ML::Metal::ComputePipeline
        @@attn_context_rows_pipeline ||= ML::Metal::PipelineCache.get("gemma4_attn_context_rows") {
          ML::Metal::ComputePipeline.new("gemma4_attn_context_rows", GEMMA4_SOURCE)
        }
      end

      private def add_vec_pipeline : ML::Metal::ComputePipeline
        @@add_vec_pipeline ||= ML::Metal::PipelineCache.get("gemma4_add_vec") {
          ML::Metal::ComputePipeline.new("gemma4_add_vec", GEMMA4_SOURCE)
        }
      end

      private def add_scaled_vec_pipeline : ML::Metal::ComputePipeline
        @@add_scaled_vec_pipeline ||= ML::Metal::PipelineCache.get("gemma4_add_scaled_vec") {
          ML::Metal::ComputePipeline.new("gemma4_add_scaled_vec", GEMMA4_SOURCE)
        }
      end

      private def gelu_mul_pipeline : ML::Metal::ComputePipeline
        @@gelu_mul_pipeline ||= ML::Metal::PipelineCache.get("gemma4_gelu_mul") {
          ML::Metal::ComputePipeline.new("gemma4_gelu_mul", GEMMA4_SOURCE)
        }
      end

      private def softcap_pipeline : ML::Metal::ComputePipeline
        @@softcap_pipeline ||= ML::Metal::PipelineCache.get("gemma4_logit_softcap") {
          ML::Metal::ComputePipeline.new("gemma4_logit_softcap", GEMMA4_SOURCE)
        }
      end
    {% end %}
  end
end
