require "./gemma4_cpu"

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
    {% else %}
      GEMMA4_SOURCE = {{ read_file("#{__DIR__}/kernels/gemma4.metal") }}

      @@rmsnorm_weighted_pipeline : ML::Metal::ComputePipeline?
      @@rmsnorm_plain_pipeline : ML::Metal::ComputePipeline?
      @@rope_pipeline : ML::Metal::ComputePipeline?

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

      private def rmsnorm_weighted_pipeline : ML::Metal::ComputePipeline
        @@rmsnorm_weighted_pipeline ||= ML::Metal::PipelineCache.get("gemma4_rmsnorm_heads_weighted") {
          ML::Metal::ComputePipeline.new("gemma4_rmsnorm_heads_weighted", GEMMA4_SOURCE)
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
    {% end %}
  end
end
