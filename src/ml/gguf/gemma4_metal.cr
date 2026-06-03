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
    {% else %}
      GEMMA4_SOURCE = {{ read_file("#{__DIR__}/kernels/gemma4.metal") }}

      @@rmsnorm_weighted_pipeline : ML::Metal::ComputePipeline?
      @@rmsnorm_vec_weighted_pipeline : ML::Metal::ComputePipeline?
      @@rmsnorm_plain_pipeline : ML::Metal::ComputePipeline?
      @@rope_pipeline : ML::Metal::ComputePipeline?
      @@kv_write_pipeline : ML::Metal::ComputePipeline?
      @@attn_context_pipeline : ML::Metal::ComputePipeline?
      @@add_vec_pipeline : ML::Metal::ComputePipeline?
      @@add_scaled_vec_pipeline : ML::Metal::ComputePipeline?
      @@gelu_mul_pipeline : ML::Metal::ComputePipeline?
      @@softcap_pipeline : ML::Metal::ComputePipeline?

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

      private def rmsnorm_plain_pipeline : ML::Metal::ComputePipeline
        @@rmsnorm_plain_pipeline ||= ML::Metal::PipelineCache.get("gemma4_rmsnorm_heads_plain") {
          ML::Metal::ComputePipeline.new("gemma4_rmsnorm_heads_plain", GEMMA4_SOURCE)
        }
      end

      private def rope_pipeline : ML::Metal::ComputePipeline
        @@rope_pipeline ||= ML::Metal::PipelineCache.get("gemma4_rope_neox") {
          ML::Metal::ComputePipeline.new("gemma4_rope_neox", GEMMA4_SOURCE)
        }
      end

      private def kv_write_pipeline : ML::Metal::ComputePipeline
        @@kv_write_pipeline ||= ML::Metal::PipelineCache.get("gemma4_kv_write_one") {
          ML::Metal::ComputePipeline.new("gemma4_kv_write_one", GEMMA4_SOURCE)
        }
      end

      private def attn_context_pipeline : ML::Metal::ComputePipeline
        @@attn_context_pipeline ||= ML::Metal::PipelineCache.get("gemma4_attn_context_one") {
          ML::Metal::ComputePipeline.new("gemma4_attn_context_one", GEMMA4_SOURCE)
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
