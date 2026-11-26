# Gemma4 Metal attention projection, RoPE, KV updates, and context dispatch.

module ML::GGUF
  module Gemma4Metal
    {% unless flag?(:cpu_only) %}
      private def attn_gqa2_enabled? : Bool
        ENV["GEMMA4_ROW_PREFILL_ATTN_GQA2_OFF"]? != "1"
      end

      private def attn_gqa_pair_full_enabled? : Bool
        ENV["GEMMA4_ROW_PREFILL_ATTN_GQA_PAIR_FULL"]? == "1"
      end

      private def attn_gqa4_enabled?(batch : Int32) : Bool
        return false if ENV["GEMMA4_ROW_PREFILL_ATTN_GQA4_OFF"]? == "1"
        return true if ENV["GEMMA4_ROW_PREFILL_ATTN_GQA4"]? == "1"
        min_batch = (ENV["GEMMA4_ROW_PREFILL_ATTN_GQA4_MIN_BATCH"]? || "128").to_i32
        batch >= min_batch
      end

      private def attn_swa256_vec_enabled?(batch : Int32) : Bool
        return false if ENV["GEMMA4_ROW_PREFILL_ATTN_SWA256_VEC_OFF"]? == "1"
        return true if batch == 1
        min_batch = (ENV["GEMMA4_ROW_PREFILL_ATTN_SWA256_VEC_MIN_BATCH"]? || "128").to_i32
        batch >= min_batch
      end

      private def attn_swa256_vec_gqa2_enabled?(batch : Int32) : Bool
        return false if ENV["GEMMA4_ROW_PREFILL_ATTN_SWA256_VEC_GQA2_OFF"]? == "1"
        return true if ENV["GEMMA4_ROW_PREFILL_ATTN_SWA256_VEC_GQA2"]? == "1"
        min_batch = (ENV["GEMMA4_ROW_PREFILL_ATTN_SWA256_VEC_GQA2_MIN_BATCH"]? || "128").to_i32
        batch >= min_batch && attn_swa256_vec_enabled?(batch)
      end

      private def attn_swa256_vec_tile16_enabled?(batch : Int32) : Bool
        return false if ENV["GEMMA4_ROW_PREFILL_ATTN_SWA256_VEC_TILE16_OFF"]? == "1"
        return false unless ENV["GEMMA4_ROW_PREFILL_ATTN_SWA256_VEC_TILE16"]? == "1"
        min_batch = (ENV["GEMMA4_ROW_PREFILL_ATTN_SWA256_VEC_TILE16_MIN_BATCH"]? || "128").to_i32
        batch >= min_batch && attn_swa256_vec_enabled?(batch)
      end

      private def attn_swa256_lane32_enabled?(batch : Int32) : Bool
        return false if ENV["GEMMA4_ROW_PREFILL_ATTN_SWA256_LANE32_OFF"]? == "1"
        min_batch = (ENV["GEMMA4_ROW_PREFILL_ATTN_SWA256_LANE32_MIN_BATCH"]? || "128").to_i32
        return true if ENV["GEMMA4_ROW_PREFILL_ATTN_SWA256_LANE32"]? == "1"
        batch >= min_batch && attn_swa256_vec_enabled?(batch)
      end

      private def attn_splitk_enabled?(batch : Int32, sliding_window : Int32) : Bool
        return false if ENV["GEMMA4_ROW_PREFILL_ATTN_SPLITK_OFF"]? == "1"
        return false unless sliding_window == 0
        return true if batch == 1
        return false unless ENV["GEMMA4_ROW_PREFILL_ATTN_SPLITK"]? == "1"
        min_batch = (ENV["GEMMA4_ROW_PREFILL_ATTN_SPLITK_MIN_BATCH"]? || "128").to_i32
        batch >= min_batch
      end

      private def attn_splitk_chunk_size : Int32
        value = (ENV["GEMMA4_ROW_PREFILL_ATTN_SPLITK_CHUNK"]? || "32").to_i32
        value > 0 ? value : 32
      end

      private def attn_splitk_query_tile : Int32
        value = (ENV["GEMMA4_ROW_PREFILL_ATTN_SPLITK_QTILE"]? || "32").to_i32
        value > 0 ? value : 32
      end

      private def attn_ctx_h16_oproj_enabled?(batch : Int32) : Bool
        return false if ENV["GEMMA4_ROW_PREFILL_ATTN_CTX_H16_OPROJ_OFF"]? == "1"
        return false unless ENV["GEMMA4_ROW_PREFILL_ATTN_CTX_H16_OPROJ"]? == "1"
        min_batch = (ENV["GEMMA4_ROW_PREFILL_ATTN_CTX_H16_OPROJ_MIN_BATCH"]? || "1024").to_i32
        max_batch = (ENV["GEMMA4_ROW_PREFILL_ATTN_CTX_H16_OPROJ_MAX_BATCH"]? || "1024").to_i32
        batch >= min_batch && (max_batch <= 0 || batch <= max_batch)
      end

      private def attn_kv_h16_cache_enabled? : Bool
        ENV["GEMMA4_ROW_PREFILL_ATTN_KV_H16_CACHE"]? == "1" &&
          ENV["GEMMA4_ROW_PREFILL_ATTN_KV_H16_CACHE_OFF"]? != "1"
      end

      private def copy_k_to_v_enabled? : Bool
        ENV["GEMMA4_COPY_K_TO_V_OFF"]? != "1"
      end

      private def attn_prep_fuse_enabled? : Bool
        ENV["GEMMA4_ATTN_PREP_FUSE"]? == "1" &&
          ENV["GEMMA4_ATTN_PREP_FUSE_OFF"]? != "1"
      end


      @@rope_pipeline : ML::Metal::ComputePipeline?
      @@rope_rows_pipeline : ML::Metal::ComputePipeline?
      @@q_norm_rope_rows_pipeline : ML::Metal::ComputePipeline?
      @@k_norm_rope_write_rows_pipeline : ML::Metal::ComputePipeline?
      @@v_norm_write_rows_pipeline : ML::Metal::ComputePipeline?
      @@kv_write_pipeline : ML::Metal::ComputePipeline?
      @@kv_write_rows_pipeline : ML::Metal::ComputePipeline?
      @@kv_write_rows_h16_pipeline : ML::Metal::ComputePipeline?
      @@attn_context_pipeline : ML::Metal::ComputePipeline?
      @@attn_context_rows_pipeline : ML::Metal::ComputePipeline?
      @@attn_context_rows_kv_h16_pipeline : ML::Metal::ComputePipeline?
      @@attn_context_rows_gqa2_pipeline : ML::Metal::ComputePipeline?
      @@attn_context_rows_gqa4_pipeline : ML::Metal::ComputePipeline?
      @@attn_context_rows_swa256_vec_pipeline : ML::Metal::ComputePipeline?
      @@attn_context_rows_swa256_vec_gqa2_pipeline : ML::Metal::ComputePipeline?
      @@attn_context_rows_swa256_lane32_gqa2_pipeline : ML::Metal::ComputePipeline?
      @@attn_context_rows_gqa2_kv_h16_pipeline : ML::Metal::ComputePipeline?
      @@attn_context_rows_splitk_kv_h16_stage1_pipeline : ML::Metal::ComputePipeline?
      @@attn_context_rows_splitk_stage1_pipeline : ML::Metal::ComputePipeline?
      @@attn_context_rows_splitk_stage2_pipeline : ML::Metal::ComputePipeline?
      @@attn_context_rows_h16_pipeline : ML::Metal::ComputePipeline?
      @@attn_context_rows_gqa2_h16_pipeline : ML::Metal::ComputePipeline?

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

      private def encode_q_norm_rope_rows(enc : ML::Metal::ComputeEncoder,
                                          q_buf : ML::MetalBuffer,
                                          weight_buf : ML::MetalBuffer,
                                          factors_buf : ML::MetalBuffer,
                                          head_dim : Int32,
                                          rope_dim : Int32,
                                          base_pos : Int32,
                                          eps : Float32,
                                          freq_base : Float32,
                                          use_factors : UInt32,
                                          n_heads : Int32,
                                          rows : Int32) : Nil
        enc.set_pipeline(q_norm_rope_rows_pipeline)
        enc.set_buffer(q_buf, 0, ML::Metal::BufferAccess::ReadWrite)
        enc.set_buffer(weight_buf, 1)
        enc.set_buffer(factors_buf, 2)
        enc.set_value(head_dim.to_u32, 3)
        enc.set_value(rope_dim.to_u32, 4)
        enc.set_value(base_pos.to_u32, 5)
        enc.set_value(eps, 6)
        enc.set_value(freq_base, 7)
        enc.set_value(use_factors, 8)
        enc.set_value(n_heads.to_u32, 9)
        enc.set_value(rows.to_u32, 10)
        enc.dispatch_threadgroups({n_heads, rows, 1}, {32, 1, 1})
      end

      private def encode_k_norm_rope_write_rows(enc : ML::Metal::ComputeEncoder,
                                                k_buf : ML::MetalBuffer,
                                                weight_buf : ML::MetalBuffer,
                                                factors_buf : ML::MetalBuffer,
                                                k_cache_buf : ML::MetalBuffer,
                                                head_dim : Int32,
                                                rope_dim : Int32,
                                                base_pos : Int32,
                                                eps : Float32,
                                                freq_base : Float32,
                                                use_factors : UInt32,
                                                n_heads : Int32,
                                                rows : Int32) : Nil
        enc.set_pipeline(k_norm_rope_write_rows_pipeline)
        enc.set_buffer(k_buf, 0, ML::Metal::BufferAccess::ReadWrite)
        enc.set_buffer(weight_buf, 1)
        enc.set_buffer(factors_buf, 2)
        enc.set_buffer(k_cache_buf, 3, ML::Metal::BufferAccess::ReadWrite)
        enc.set_value(head_dim.to_u32, 4)
        enc.set_value(rope_dim.to_u32, 5)
        enc.set_value(base_pos.to_u32, 6)
        enc.set_value(eps, 7)
        enc.set_value(freq_base, 8)
        enc.set_value(use_factors, 9)
        enc.set_value(n_heads.to_u32, 10)
        enc.set_value(rows.to_u32, 11)
        enc.dispatch_threadgroups({n_heads, rows, 1}, {32, 1, 1})
      end

      private def encode_v_norm_write_rows(enc : ML::Metal::ComputeEncoder,
                                           v_buf : ML::MetalBuffer,
                                           v_cache_buf : ML::MetalBuffer,
                                           head_dim : Int32,
                                           base_pos : Int32,
                                           eps : Float32,
                                           n_heads : Int32,
                                           rows : Int32) : Nil
        enc.set_pipeline(v_norm_write_rows_pipeline)
        enc.set_buffer(v_buf, 0, ML::Metal::BufferAccess::ReadWrite)
        enc.set_buffer(v_cache_buf, 1, ML::Metal::BufferAccess::ReadWrite)
        enc.set_value(head_dim.to_u32, 2)
        enc.set_value(base_pos.to_u32, 3)
        enc.set_value(eps, 4)
        enc.set_value(n_heads.to_u32, 5)
        enc.set_value(rows.to_u32, 6)
        enc.dispatch_threadgroups({n_heads, rows, 1}, {32, 1, 1})
      end

      private def encode_attn_prep_rows(enc : ML::Metal::ComputeEncoder,
                                        q_buf : ML::MetalBuffer,
                                        k_buf : ML::MetalBuffer,
                                        v_buf : ML::MetalBuffer,
                                        q_weight : ML::MetalBuffer,
                                        k_weight : ML::MetalBuffer,
                                        factor_buf : ML::MetalBuffer,
                                        lstate : ResidentLayerState,
                                        head_dim : Int32,
                                        rope_dim : Int32,
                                        base_pos : Int32,
                                        base : Float32,
                                        use_factors : UInt32,
                                        n_head : Int32,
                                        n_head_kv : Int32,
                                        kv_dim : Int32,
                                        rows : Int32,
                                        eps : Float32,
                                        kv_h16_cache : Bool,
                                        v_from_k_buf : Bool = false) : Nil
        if attn_prep_fuse_enabled? && !kv_h16_cache
          encode_q_norm_rope_rows(enc, q_buf, q_weight, factor_buf, head_dim, rope_dim, base_pos, eps, base, use_factors, n_head, rows)
          if v_from_k_buf
            encode_v_norm_write_rows(enc, k_buf, lstate.v_cache_buf, head_dim, base_pos, eps, n_head_kv, rows)
          else
            encode_v_norm_write_rows(enc, v_buf, lstate.v_cache_buf, head_dim, base_pos, eps, n_head_kv, rows)
          end
          encode_k_norm_rope_write_rows(enc, k_buf, k_weight, factor_buf, lstate.k_cache_buf, head_dim, rope_dim, base_pos, eps, base, use_factors, n_head_kv, rows)
        else
          encode_rmsnorm_weighted_rows(enc, q_buf, q_weight, head_dim, eps, n_head, rows)
          encode_rmsnorm_weighted_rows(enc, k_buf, k_weight, head_dim, eps, n_head_kv, rows)
          encode_rmsnorm_plain_rows(enc, v_buf, head_dim, eps, n_head_kv, rows)
          encode_rope_rows(enc, q_buf, factor_buf, head_dim, rope_dim, base_pos, base, use_factors, n_head, rows)
          encode_rope_rows(enc, k_buf, factor_buf, head_dim, rope_dim, base_pos, base, use_factors, n_head_kv, rows)
          encode_kv_write_rows(enc, k_buf, v_buf, lstate.k_cache_buf, lstate.v_cache_buf, base_pos, kv_dim, rows)
          if kv_h16_cache
            encode_kv_write_rows_h16(enc, k_buf, v_buf, lstate.k_cache_h16_buf.not_nil!, lstate.v_cache_h16_buf.not_nil!, base_pos, kv_dim, rows)
          end
        end
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

      private def encode_kv_write_rows_h16(enc : ML::Metal::ComputeEncoder,
                                           k_buf : ML::MetalBuffer,
                                           v_buf : ML::MetalBuffer,
                                           k_cache_buf : ML::MetalBuffer,
                                           v_cache_buf : ML::MetalBuffer,
                                           base_pos : Int32,
                                           kv_dim : Int32,
                                           rows : Int32) : Nil
        enc.set_pipeline(kv_write_rows_h16_pipeline)
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
        use_swa256_vec = attn_swa256_vec_enabled?(rows) && head_dim == 256 && sliding_window > 0
        use_swa256_lane32 = attn_swa256_lane32_enabled?(rows) && head_dim == 256 && sliding_window > 0
        use_swa256_vec_tile16 = attn_swa256_vec_tile16_enabled?(rows) && head_dim == 256 && sliding_window > 0
        use_swa256_vec_gqa2 = attn_swa256_vec_gqa2_enabled?(rows) && head_dim == 256 && sliding_window > 0 &&
          heads_per_group >= 2 && heads_per_group.even? && n_head.even?
        use_gqa4 = attn_gqa4_enabled?(rows) && heads_per_group >= 4 && (heads_per_group % 4 == 0) && (n_head % 4 == 0)
        use_gqa_pair = attn_gqa2_enabled? &&
          (heads_per_group == 2 || (attn_gqa_pair_full_enabled? && heads_per_group > 2 && heads_per_group.even?))
        if use_swa256_lane32 && use_swa256_vec_gqa2 && heads_per_group == 2
          enc.set_pipeline(attn_context_rows_swa256_lane32_gqa2_pipeline)
          enc.dispatch_threadgroups({n_head // 2, rows, 1}, {32, 1, 1})
        elsif use_swa256_vec_tile16 && use_swa256_vec_gqa2
          enc.set_pipeline(attn_context_rows_swa256_vec_gqa2_tile16_pipeline)
          enc.dispatch_threadgroups({n_head // 2, rows, 1}, {256, 1, 1})
        elsif use_swa256_vec_tile16
          enc.set_pipeline(attn_context_rows_swa256_vec_tile16_pipeline)
          enc.dispatch_threadgroups({n_head, rows, 1}, {256, 1, 1})
        elsif use_swa256_vec_gqa2
          enc.set_pipeline(attn_context_rows_swa256_vec_gqa2_pipeline)
          enc.dispatch_threadgroups({n_head // 2, rows, 1}, {256, 1, 1})
        elsif use_swa256_vec
          enc.set_pipeline(attn_context_rows_swa256_vec_pipeline)
          enc.dispatch_threadgroups({n_head, rows, 1}, {256, 1, 1})
        elsif use_gqa4
          enc.set_pipeline(attn_context_rows_gqa4_pipeline)
          enc.dispatch_threadgroups({n_head // 4, rows, 1}, {32, 1, 1})
        elsif use_gqa_pair
          enc.set_pipeline(attn_context_rows_gqa2_pipeline)
          enc.dispatch_threadgroups({n_head // 2, rows, 1}, {32, 1, 1})
        else
          enc.dispatch_threadgroups({n_head, rows, 1}, {32, 1, 1})
        end
      end

      private def encode_attention_context_rows_splitk(enc : ML::Metal::ComputeEncoder,
                                                       scratch : ResidentScratch,
                                                       q_buf : ML::MetalBuffer,
                                                       k_cache_buf : ML::MetalBuffer,
                                                       v_cache_buf : ML::MetalBuffer,
                                                       out_buf : ML::MetalBuffer,
                                                       base_pos : Int32,
                                                       rows : Int32,
                                                       n_head : Int32,
                                                       n_head_kv : Int32,
                                                       head_dim : Int32,
                                                       heads_per_group : Int32) : Nil
        chunk = attn_splitk_chunk_size
        qtile = Math.min(attn_splitk_query_tile, rows)
        max_context = base_pos + rows
        n_blocks = (max_context + chunk - 1) // chunk
        partial_o = scratch.get("rows.attn_splitk.o", qtile.to_i64 * n_head * n_blocks * head_dim * sizeof(Float32))
        partial_m = scratch.get("rows.attn_splitk.m", qtile.to_i64 * n_head * n_blocks * sizeof(Float32))
        partial_l = scratch.get("rows.attn_splitk.l", qtile.to_i64 * n_head * n_blocks * sizeof(Float32))

        query_start = 0
        while query_start < rows
          query_count = Math.min(qtile, rows - query_start)

          enc.set_pipeline(attn_context_rows_splitk_stage1_pipeline)
          enc.set_buffer(q_buf, 0)
          enc.set_buffer(k_cache_buf, 1)
          enc.set_buffer(v_cache_buf, 2)
          enc.set_buffer(partial_o, 3, ML::Metal::BufferAccess::Write)
          enc.set_buffer(partial_m, 4, ML::Metal::BufferAccess::Write)
          enc.set_buffer(partial_l, 5, ML::Metal::BufferAccess::Write)
          enc.set_value(base_pos.to_u32, 6)
          enc.set_value(query_start.to_u32, 7)
          enc.set_value(query_count.to_u32, 8)
          enc.set_value(n_head.to_u32, 9)
          enc.set_value(n_head_kv.to_u32, 10)
          enc.set_value(head_dim.to_u32, 11)
          enc.set_value(heads_per_group.to_u32, 12)
          enc.set_value(chunk.to_u32, 13)
          enc.set_value(n_blocks.to_u32, 14)
          enc.dispatch_threadgroups({n_head, query_count, n_blocks}, {32, 1, 1})

          enc.set_pipeline(attn_context_rows_splitk_stage2_pipeline)
          enc.set_buffer(partial_o, 0)
          enc.set_buffer(partial_m, 1)
          enc.set_buffer(partial_l, 2)
          enc.set_buffer(out_buf, 3, ML::Metal::BufferAccess::Write)
          enc.set_value(query_start.to_u32, 4)
          enc.set_value(query_count.to_u32, 5)
          enc.set_value(n_head.to_u32, 6)
          enc.set_value(head_dim.to_u32, 7)
          enc.set_value(n_blocks.to_u32, 8)
          enc.dispatch_threadgroups({n_head, query_count, 1}, {32, 1, 1})

          query_start += query_count
        end
      end

      private def encode_attention_context_rows_splitk_kv_h16(enc : ML::Metal::ComputeEncoder,
                                                              scratch : ResidentScratch,
                                                              q_buf : ML::MetalBuffer,
                                                              k_cache_buf : ML::MetalBuffer,
                                                              v_cache_buf : ML::MetalBuffer,
                                                              out_buf : ML::MetalBuffer,
                                                              base_pos : Int32,
                                                              rows : Int32,
                                                              n_head : Int32,
                                                              n_head_kv : Int32,
                                                              head_dim : Int32,
                                                              heads_per_group : Int32) : Nil
        chunk = attn_splitk_chunk_size
        qtile = Math.min(attn_splitk_query_tile, rows)
        max_context = base_pos + rows
        n_blocks = (max_context + chunk - 1) // chunk
        partial_o = scratch.get("rows.attn_splitk_h16.o", qtile.to_i64 * n_head * n_blocks * head_dim * sizeof(Float32))
        partial_m = scratch.get("rows.attn_splitk_h16.m", qtile.to_i64 * n_head * n_blocks * sizeof(Float32))
        partial_l = scratch.get("rows.attn_splitk_h16.l", qtile.to_i64 * n_head * n_blocks * sizeof(Float32))

        query_start = 0
        while query_start < rows
          query_count = Math.min(qtile, rows - query_start)

          enc.set_pipeline(attn_context_rows_splitk_kv_h16_stage1_pipeline)
          enc.set_buffer(q_buf, 0)
          enc.set_buffer(k_cache_buf, 1)
          enc.set_buffer(v_cache_buf, 2)
          enc.set_buffer(partial_o, 3, ML::Metal::BufferAccess::Write)
          enc.set_buffer(partial_m, 4, ML::Metal::BufferAccess::Write)
          enc.set_buffer(partial_l, 5, ML::Metal::BufferAccess::Write)
          enc.set_value(base_pos.to_u32, 6)
          enc.set_value(query_start.to_u32, 7)
          enc.set_value(query_count.to_u32, 8)
          enc.set_value(n_head.to_u32, 9)
          enc.set_value(n_head_kv.to_u32, 10)
          enc.set_value(head_dim.to_u32, 11)
          enc.set_value(heads_per_group.to_u32, 12)
          enc.set_value(chunk.to_u32, 13)
          enc.set_value(n_blocks.to_u32, 14)
          enc.dispatch_threadgroups({n_head, query_count, n_blocks}, {32, 1, 1})

          enc.set_pipeline(attn_context_rows_splitk_stage2_pipeline)
          enc.set_buffer(partial_o, 0)
          enc.set_buffer(partial_m, 1)
          enc.set_buffer(partial_l, 2)
          enc.set_buffer(out_buf, 3, ML::Metal::BufferAccess::Write)
          enc.set_value(query_start.to_u32, 4)
          enc.set_value(query_count.to_u32, 5)
          enc.set_value(n_head.to_u32, 6)
          enc.set_value(head_dim.to_u32, 7)
          enc.set_value(n_blocks.to_u32, 8)
          enc.dispatch_threadgroups({n_head, query_count, 1}, {32, 1, 1})

          query_start += query_count
        end
      end

      private def encode_attention_context_rows_kv_h16(enc : ML::Metal::ComputeEncoder,
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
        enc.set_pipeline(attn_context_rows_kv_h16_pipeline)
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
        use_gqa_pair = attn_gqa2_enabled? &&
          (heads_per_group == 2 || (attn_gqa_pair_full_enabled? && heads_per_group > 2 && heads_per_group.even?))
        if use_gqa_pair
          enc.set_pipeline(attn_context_rows_gqa2_kv_h16_pipeline)
          enc.dispatch_threadgroups({n_head // 2, rows, 1}, {32, 1, 1})
        else
          enc.dispatch_threadgroups({n_head, rows, 1}, {32, 1, 1})
        end
      end

      private def encode_attention_context_rows_h16(enc : ML::Metal::ComputeEncoder,
                                                    q_buf : ML::MetalBuffer,
                                                    k_cache_buf : ML::MetalBuffer,
                                                    v_cache_buf : ML::MetalBuffer,
                                                    out16_buf : ML::MetalBuffer,
                                                    base_pos : Int32,
                                                    rows : Int32,
                                                    n_head : Int32,
                                                    n_head_kv : Int32,
                                                    head_dim : Int32,
                                                    heads_per_group : Int32,
                                                    sliding_window : Int32) : Nil
        enc.set_pipeline(attn_context_rows_h16_pipeline)
        enc.set_buffer(q_buf, 0)
        enc.set_buffer(k_cache_buf, 1)
        enc.set_buffer(v_cache_buf, 2)
        enc.set_buffer(out16_buf, 3, ML::Metal::BufferAccess::Write)
        enc.set_value(base_pos.to_u32, 4)
        enc.set_value(rows.to_u32, 5)
        enc.set_value(n_head.to_u32, 6)
        enc.set_value(n_head_kv.to_u32, 7)
        enc.set_value(head_dim.to_u32, 8)
        enc.set_value(heads_per_group.to_u32, 9)
        enc.set_value(sliding_window.to_u32, 10)
        use_gqa_pair = attn_gqa2_enabled? &&
          (heads_per_group == 2 || (attn_gqa_pair_full_enabled? && heads_per_group > 2 && heads_per_group.even?))
        if use_gqa_pair
          enc.set_pipeline(attn_context_rows_gqa2_h16_pipeline)
          enc.dispatch_threadgroups({n_head // 2, rows, 1}, {32, 1, 1})
        else
          enc.dispatch_threadgroups({n_head, rows, 1}, {32, 1, 1})
        end
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

      private def q_norm_rope_rows_pipeline : ML::Metal::ComputePipeline
        @@q_norm_rope_rows_pipeline ||= ML::Metal::PipelineCache.get("gemma4_q_norm_rope_rows") {
          ML::Metal::ComputePipeline.new("gemma4_q_norm_rope_rows", GEMMA4_SOURCE)
        }
      end

      private def k_norm_rope_write_rows_pipeline : ML::Metal::ComputePipeline
        @@k_norm_rope_write_rows_pipeline ||= ML::Metal::PipelineCache.get("gemma4_k_norm_rope_write_rows") {
          ML::Metal::ComputePipeline.new("gemma4_k_norm_rope_write_rows", GEMMA4_SOURCE)
        }
      end

      private def v_norm_write_rows_pipeline : ML::Metal::ComputePipeline
        @@v_norm_write_rows_pipeline ||= ML::Metal::PipelineCache.get("gemma4_v_norm_write_rows") {
          ML::Metal::ComputePipeline.new("gemma4_v_norm_write_rows", GEMMA4_SOURCE)
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

      private def kv_write_rows_h16_pipeline : ML::Metal::ComputePipeline
        @@kv_write_rows_h16_pipeline ||= ML::Metal::PipelineCache.get("gemma4_kv_write_rows_h16") {
          ML::Metal::ComputePipeline.new("gemma4_kv_write_rows_h16", GEMMA4_SOURCE)
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

      private def attn_context_rows_kv_h16_pipeline : ML::Metal::ComputePipeline
        @@attn_context_rows_kv_h16_pipeline ||= ML::Metal::PipelineCache.get("gemma4_attn_context_rows_kv_h16") {
          ML::Metal::ComputePipeline.new("gemma4_attn_context_rows_kv_h16", GEMMA4_SOURCE)
        }
      end

      private def attn_context_rows_gqa2_pipeline : ML::Metal::ComputePipeline
        @@attn_context_rows_gqa2_pipeline ||= ML::Metal::PipelineCache.get("gemma4_attn_context_rows_gqa2") {
          ML::Metal::ComputePipeline.new("gemma4_attn_context_rows_gqa2", GEMMA4_SOURCE)
        }
      end

      private def attn_context_rows_gqa4_pipeline : ML::Metal::ComputePipeline
        @@attn_context_rows_gqa4_pipeline ||= ML::Metal::PipelineCache.get("gemma4_attn_context_rows_gqa4") {
          ML::Metal::ComputePipeline.new("gemma4_attn_context_rows_gqa4", GEMMA4_SOURCE)
        }
      end

      private def attn_context_rows_swa256_vec_pipeline : ML::Metal::ComputePipeline
        @@attn_context_rows_swa256_vec_pipeline ||= ML::Metal::PipelineCache.get("gemma4_attn_context_rows_swa256_vec") {
          ML::Metal::ComputePipeline.new("gemma4_attn_context_rows_swa256_vec", GEMMA4_SOURCE)
        }
      end

      private def attn_context_rows_swa256_vec_gqa2_pipeline : ML::Metal::ComputePipeline
        @@attn_context_rows_swa256_vec_gqa2_pipeline ||= ML::Metal::PipelineCache.get("gemma4_attn_context_rows_swa256_vec_gqa2") {
          ML::Metal::ComputePipeline.new("gemma4_attn_context_rows_swa256_vec_gqa2", GEMMA4_SOURCE)
        }
      end

      private def attn_context_rows_swa256_vec_tile16_pipeline : ML::Metal::ComputePipeline
        @@attn_context_rows_swa256_vec_tile16_pipeline ||= ML::Metal::PipelineCache.get("gemma4_attn_context_rows_swa256_vec_tile16") {
          ML::Metal::ComputePipeline.new("gemma4_attn_context_rows_swa256_vec_tile16", GEMMA4_SOURCE)
        }
      end

      private def attn_context_rows_swa256_vec_gqa2_tile16_pipeline : ML::Metal::ComputePipeline
        @@attn_context_rows_swa256_vec_gqa2_tile16_pipeline ||= ML::Metal::PipelineCache.get("gemma4_attn_context_rows_swa256_vec_gqa2_tile16") {
          ML::Metal::ComputePipeline.new("gemma4_attn_context_rows_swa256_vec_gqa2_tile16", GEMMA4_SOURCE)
        }
      end

      private def attn_context_rows_swa256_lane32_gqa2_pipeline : ML::Metal::ComputePipeline
        @@attn_context_rows_swa256_lane32_gqa2_pipeline ||= ML::Metal::PipelineCache.get("gemma4_attn_context_rows_swa256_lane32_gqa2") {
          ML::Metal::ComputePipeline.new("gemma4_attn_context_rows_swa256_lane32_gqa2", GEMMA4_SOURCE)
        }
      end

      private def attn_context_rows_gqa2_kv_h16_pipeline : ML::Metal::ComputePipeline
        @@attn_context_rows_gqa2_kv_h16_pipeline ||= ML::Metal::PipelineCache.get("gemma4_attn_context_rows_gqa2_kv_h16") {
          ML::Metal::ComputePipeline.new("gemma4_attn_context_rows_gqa2_kv_h16", GEMMA4_SOURCE)
        }
      end

      private def attn_context_rows_splitk_stage1_pipeline : ML::Metal::ComputePipeline
        @@attn_context_rows_splitk_stage1_pipeline ||= ML::Metal::PipelineCache.get("gemma4_attn_context_rows_splitk_stage1") {
          ML::Metal::ComputePipeline.new("gemma4_attn_context_rows_splitk_stage1", GEMMA4_SOURCE)
        }
      end

      private def attn_context_rows_splitk_kv_h16_stage1_pipeline : ML::Metal::ComputePipeline
        @@attn_context_rows_splitk_kv_h16_stage1_pipeline ||= ML::Metal::PipelineCache.get("gemma4_attn_context_rows_splitk_kv_h16_stage1") {
          ML::Metal::ComputePipeline.new("gemma4_attn_context_rows_splitk_kv_h16_stage1", GEMMA4_SOURCE)
        }
      end

      private def attn_context_rows_splitk_stage2_pipeline : ML::Metal::ComputePipeline
        @@attn_context_rows_splitk_stage2_pipeline ||= ML::Metal::PipelineCache.get("gemma4_attn_context_rows_splitk_stage2") {
          ML::Metal::ComputePipeline.new("gemma4_attn_context_rows_splitk_stage2", GEMMA4_SOURCE)
        }
      end

      private def attn_context_rows_h16_pipeline : ML::Metal::ComputePipeline
        @@attn_context_rows_h16_pipeline ||= ML::Metal::PipelineCache.get("gemma4_attn_context_rows_h16") {
          ML::Metal::ComputePipeline.new("gemma4_attn_context_rows_h16", GEMMA4_SOURCE)
        }
      end

      private def attn_context_rows_gqa2_h16_pipeline : ML::Metal::ComputePipeline
        @@attn_context_rows_gqa2_h16_pipeline ||= ML::Metal::PipelineCache.get("gemma4_attn_context_rows_gqa2_h16") {
          ML::Metal::ComputePipeline.new("gemma4_attn_context_rows_gqa2_h16", GEMMA4_SOURCE)
        }
      end

    {% end %}
  end
end
