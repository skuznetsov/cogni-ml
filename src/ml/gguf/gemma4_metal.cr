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

      private def q4_pair_ffn_enabled? : Bool
        ENV["GEMMA4_ROW_PREFILL_Q4_PAIR_FFN"]? == "1"
      end

      private def rmsnorm_h16_proj_enabled?(batch : Int32) : Bool
        return false if ENV["GEMMA4_ROW_PREFILL_RMSNORM_H16_PROJ_OFF"]? == "1"
        return false unless ENV["GEMMA4_ROW_PREFILL_RMSNORM_H16_PROJ"]? == "1"
        min_batch = (ENV["GEMMA4_ROW_PREFILL_RMSNORM_H16_PROJ_MIN_BATCH"]? || "512").to_i32
        max_batch = (ENV["GEMMA4_ROW_PREFILL_RMSNORM_H16_PROJ_MAX_BATCH"]? || "768").to_i32
        batch >= min_batch && (max_batch <= 0 || batch <= max_batch)
      end

      private def attn_gqa2_enabled? : Bool
        ENV["GEMMA4_ROW_PREFILL_ATTN_GQA2_OFF"]? != "1"
      end

      private def attn_gqa_pair_full_enabled? : Bool
        ENV["GEMMA4_ROW_PREFILL_ATTN_GQA_PAIR_FULL"]? == "1"
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

      private def row_prefill_resident_corridor_enabled? : Bool
        ENV["GEMMA4_ROW_PREFILL_RESIDENT_CORRIDOR_OFF"]? != "1"
      end

      private def row_prefill_profile_layers_enabled? : Bool
        ENV["GEMMA4_ROW_PREFILL_PROFILE_LAYERS"]? == "1"
      end

      private def row_prefill_profile_phases_enabled? : Bool
        ENV["GEMMA4_ROW_PREFILL_PROFILE_PHASES"]? == "1"
      end

      private def decode_profile_phases_enabled? : Bool
        ENV["GEMMA4_DECODE_PROFILE_PHASES"]? == "1"
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
      @@kv_write_rows_h16_pipeline : ML::Metal::ComputePipeline?
      @@attn_context_pipeline : ML::Metal::ComputePipeline?
      @@attn_context_rows_pipeline : ML::Metal::ComputePipeline?
      @@attn_context_rows_kv_h16_pipeline : ML::Metal::ComputePipeline?
      @@attn_context_rows_gqa2_pipeline : ML::Metal::ComputePipeline?
      @@attn_context_rows_gqa2_kv_h16_pipeline : ML::Metal::ComputePipeline?
      @@attn_context_rows_splitk_stage1_pipeline : ML::Metal::ComputePipeline?
      @@attn_context_rows_splitk_stage2_pipeline : ML::Metal::ComputePipeline?
      @@attn_context_rows_h16_pipeline : ML::Metal::ComputePipeline?
      @@attn_context_rows_gqa2_h16_pipeline : ML::Metal::ComputePipeline?
      @@add_vec_pipeline : ML::Metal::ComputePipeline?
      @@add_scaled_vec_pipeline : ML::Metal::ComputePipeline?
      @@gelu_mul_pipeline : ML::Metal::ComputePipeline?
      @@softcap_pipeline : ML::Metal::ComputePipeline?

      class ResidentLayerState
        getter k_cache_buf : ML::MetalBuffer
        getter v_cache_buf : ML::MetalBuffer
        getter k_cache_h16_buf : ML::MetalBuffer?
        getter v_cache_h16_buf : ML::MetalBuffer?
        getter kv_dim : Int32
        getter max_seq : Int32

        def initialize(@kv_dim : Int32, @max_seq : Int32, h16_cache : Bool = false)
          @k_cache_buf = ML::MetalBuffer.from_array(Array(Float32).new(max_seq * kv_dim, 0.0_f32))
          @v_cache_buf = ML::MetalBuffer.from_array(Array(Float32).new(max_seq * kv_dim, 0.0_f32))
          if h16_cache
            @k_cache_h16_buf = ML::MetalBuffer.new(max_seq.to_i64 * kv_dim * 2_i64)
            @v_cache_h16_buf = ML::MetalBuffer.new(max_seq.to_i64 * kv_dim * 2_i64)
          end
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
          h16_cache = ENV["GEMMA4_ROW_PREFILL_ATTN_KV_H16_CACHE"]? == "1" &&
            ENV["GEMMA4_ROW_PREFILL_ATTN_KV_H16_CACHE_OFF"]? != "1"
          @layers = Array(ResidentLayerState).new(hp.n_layer) do |il|
            kv_dim = hp.n_head_kv(il) * hp.head_dim_for_layer(il)
            ResidentLayerState.new(kv_dim, @max_seq, h16_cache)
          end
          @scratch = ResidentScratch.new
        end

        def initialize(kv_dims : Array(Int32), @max_seq : Int32, h16_cache : Bool = false)
          raise ArgumentError.new("Gemma4 resident KV dims must not be empty") if kv_dims.empty?

          @layers = kv_dims.map do |kv_dim|
            raise ArgumentError.new("Gemma4 resident KV dim must be positive") unless kv_dim > 0

            ResidentLayerState.new(kv_dim, @max_seq, h16_cache)
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
          pair_q4 = q4_pair_ffn_enabled? &&
            Qwen35Metal.encode_q4k_gemm_h16_pair_to_buffers(enc, lw.ffn_gate_qw, lw.ffn_up_qw, ffn_in_buf, gate_buf, up_buf, batch)
          unless pair_q4 || Qwen35Metal.encode_matmul_many_to_buffers(enc, [lw.ffn_gate_qw, lw.ffn_up_qw], ffn_in_buf, [gate_buf, up_buf], batch)
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

      private def write_scratch_f32(scratch : ResidentScratch, name : String, data : Array(Float32)) : ML::MetalBuffer
        buf = scratch.get(name, data.size.to_i64 * sizeof(Float32))
        buf.write(data)
        buf
      end

      private def self.profile_rows_phase(label : String, &block : ML::Metal::ComputeEncoder -> Bool) : Bool
        t0 = Time.instant
        cmd = ML::Metal::CommandBuffer.new
        enc = ML::Metal::ComputeEncoder.new(cmd)
        ok = yield enc
        enc.end_encoding
        return false unless ok
        tenc = Time.instant
        cmd.commit
        cmd.wait
        twait = Time.instant
        Qwen35Metal::Profile.bump_group(label,
          (tenc - t0).total_nanoseconds.to_i64,
          (twait - tenc).total_nanoseconds.to_i64,
          0_i64)
        true
      end

      private def self.forward_layer_resident_cache_rows_profile_phases_to_buffer(weights : Gemma4Weights,
                                                                                  il : Int32,
                                                                                  x_buf : ML::MetalBuffer,
                                                                                  out_buf : ML::MetalBuffer,
                                                                                  base_pos : Int32,
                                                                                  batch : Int32,
                                                                                  state : ResidentState,
                                                                                  scratch : ResidentScratch = state.scratch) : Bool
        return false unless available?

        hp = weights.hparams
        lw = weights.layers[il]
        hidden_dim = hp.n_embd
        head_dim = hp.head_dim_for_layer(il)
        rope_dim = hp.rope_dim_for_layer(il)
        q_dim = hp.n_head * head_dim
        kv_dim = hp.n_head_kv(il) * head_dim
        ffn_dim = lw.ffn_gate_qw.out_dim
        hidden_bytes = batch.to_i64 * hidden_dim * sizeof(Float32)
        raise ArgumentError.new("batch must be positive") unless batch > 0
        raise ArgumentError.new("profile layer rows input buffer too small") if x_buf.size < hidden_bytes
        raise ArgumentError.new("profile layer rows output buffer too small") if out_buf.size < hidden_bytes
        raise ArgumentError.new("unsupported head_dim #{head_dim}; max 512") if head_dim > 512
        raise ArgumentError.new("rope_dim #{rope_dim} must be even") unless rope_dim.even?
        raise ArgumentError.new("base_pos #{base_pos} exceeds max_seq #{state.layers[il].max_seq}") if base_pos < 0 || base_pos + batch > state.layers[il].max_seq

        lstate = state.layers[il]
        raise ArgumentError.new("resident kv_dim mismatch") unless lstate.kv_dim == kv_dim
        kv_h16_cache = attn_kv_h16_cache_enabled?


        attn_norm_w = write_scratch_f32(scratch, "rows.attn_norm_w.#{il}", lw.attn_norm)
        post_attn_w = write_scratch_f32(scratch, "rows.post_attn_w.#{il}", lw.post_attention_norm)
        ffn_w = write_scratch_f32(scratch, "rows.ffn_w.#{il}", lw.ffn_norm)
        post_ffw_w = write_scratch_f32(scratch, "rows.post_ffw_w.#{il}", lw.post_ffw_norm)
        q_weight = write_scratch_f32(scratch, "rows.q_norm_w.#{il}", lw.attn_q_norm)
        k_weight = write_scratch_f32(scratch, "rows.k_norm_w.#{il}", lw.attn_k_norm)
        rope_freqs = hp.full_attention?(il) ? weights.rope_freqs : nil
        factors = (hp.full_attention?(il) && rope_freqs) ? rope_freqs.not_nil! : [1.0_f32]
        factor_buf = write_scratch_f32(scratch, "rows.rope_factors.#{il}", factors)
        use_factors = (hp.full_attention?(il) && rope_freqs) ? 1_u32 : 0_u32
        base = hp.rope_freq_base_for_layer(il)
        heads_per_group = hp.n_head // hp.n_head_kv(il)
        sliding_window = hp.sliding_window?(il) ? hp.sliding_window : 0

        x_norm_buf = scratch.get("rows.x_norm", hidden_bytes)
        norm_h16_proj = rmsnorm_h16_proj_enabled?(batch)
        x_norm_h16_buf = norm_h16_proj ? scratch.get("rows.x_norm_h16", batch.to_i64 * hidden_dim * 2_i64) : nil
        q_buf = scratch.get("rows.q", batch.to_i64 * q_dim * sizeof(Float32))
        k_buf = scratch.get("rows.k", batch.to_i64 * kv_dim * sizeof(Float32))
        v_buf = scratch.get("rows.v", batch.to_i64 * kv_dim * sizeof(Float32))
        ctx_buf = scratch.get("rows.ctx", batch.to_i64 * q_dim * sizeof(Float32))
        ctx_h16_oproj = attn_ctx_h16_oproj_enabled?(batch) && !kv_h16_cache
        ctx_h16_buf = ctx_h16_oproj ? scratch.get("rows.ctx_h16", batch.to_i64 * q_dim * 2_i64) : nil
        attn_projected_buf = scratch.get("rows.attn_projected", hidden_bytes)
        attn_normed_buf = scratch.get("rows.attn_normed", hidden_bytes)
        attn_out_buf = scratch.get("rows.attn_out", hidden_bytes)
        ffn_in_buf = scratch.get("rows.ffn_in", hidden_bytes)
        ffn_in_h16_buf = norm_h16_proj ? scratch.get("rows.ffn_in_h16", batch.to_i64 * hidden_dim * 2_i64) : nil
        gate_buf = scratch.get("rows.gate", batch.to_i64 * ffn_dim * sizeof(Float32))
        up_buf = scratch.get("rows.up", batch.to_i64 * ffn_dim * sizeof(Float32))
        combined_buf = scratch.get("rows.combined", batch.to_i64 * ffn_dim * sizeof(Float32))
        ffn_buf = scratch.get("rows.ffn", hidden_bytes)
        ffn_normed_buf = scratch.get("rows.ffn_normed", hidden_bytes)

        prefix = "gemma4.rows.layer#{il}"
        return false unless profile_rows_phase("#{prefix}.attn_qkv") do |enc|
          if x16 = x_norm_h16_buf
            Qwen35Metal.encode_rmsnorm_rows_f32_h16_to_buffers(enc, x_buf, attn_norm_w, x_norm_buf, x16, hidden_dim, batch, hp.rms_eps)
            projected = if v_qw = lw.attn_v_qw
                          Qwen35Metal.encode_matmul_from_h16_to_buffer(enc, lw.attn_q_qw, x16, q_buf, batch) &&
                            Qwen35Metal.encode_matmul_from_h16_to_buffer(enc, lw.attn_k_qw, x16, k_buf, batch) &&
                            Qwen35Metal.encode_matmul_from_h16_to_buffer(enc, v_qw, x16, v_buf, batch)
                        else
                          Qwen35Metal.encode_matmul_from_h16_to_buffer(enc, lw.attn_q_qw, x16, q_buf, batch) &&
                            Qwen35Metal.encode_matmul_from_h16_to_buffer(enc, lw.attn_k_qw, x16, k_buf, batch) &&
                            Qwen35Metal.encode_matmul_from_h16_to_buffer(enc, lw.attn_k_qw, x16, v_buf, batch)
                        end
            projected || Qwen35Metal.encode_matmul_many_to_buffers(enc, [lw.attn_q_qw, lw.attn_k_qw, (lw.attn_v_qw || lw.attn_k_qw)], x_norm_buf, [q_buf, k_buf, v_buf], batch)
          else
            encode_rmsnorm_rows_weighted_out(enc, x_buf, attn_norm_w, x_norm_buf, hidden_dim, batch, hp.rms_eps)
            Qwen35Metal.encode_matmul_many_to_buffers(enc, [lw.attn_q_qw, lw.attn_k_qw, (lw.attn_v_qw || lw.attn_k_qw)], x_norm_buf, [q_buf, k_buf, v_buf], batch)
          end
        end

        return false unless profile_rows_phase("#{prefix}.attn_prep") do |enc|
          encode_rmsnorm_weighted_rows(enc, q_buf, q_weight, head_dim, hp.rms_eps, hp.n_head, batch)
          encode_rmsnorm_weighted_rows(enc, k_buf, k_weight, head_dim, hp.rms_eps, hp.n_head_kv(il), batch)
          encode_rmsnorm_plain_rows(enc, v_buf, head_dim, hp.rms_eps, hp.n_head_kv(il), batch)
          encode_rope_rows(enc, q_buf, factor_buf, head_dim, rope_dim, base_pos, base, use_factors, hp.n_head, batch)
          encode_rope_rows(enc, k_buf, factor_buf, head_dim, rope_dim, base_pos, base, use_factors, hp.n_head_kv(il), batch)
          encode_kv_write_rows(enc, k_buf, v_buf, lstate.k_cache_buf, lstate.v_cache_buf, base_pos, kv_dim, batch)
          if kv_h16_cache
            encode_kv_write_rows_h16(enc, k_buf, v_buf, lstate.k_cache_h16_buf.not_nil!, lstate.v_cache_h16_buf.not_nil!, base_pos, kv_dim, batch)
          end
          true
        end

        return false unless profile_rows_phase("#{prefix}.attn_ctx") do |enc|
          if kv_h16_cache
            encode_attention_context_rows_kv_h16(enc, q_buf, lstate.k_cache_h16_buf.not_nil!, lstate.v_cache_h16_buf.not_nil!, ctx_buf,
              base_pos, batch, hp.n_head, hp.n_head_kv(il), head_dim, heads_per_group, sliding_window)
          elsif ctx_h16_buf
            encode_attention_context_rows_h16(enc, q_buf, lstate.k_cache_buf, lstate.v_cache_buf, ctx_h16_buf,
              base_pos, batch, hp.n_head, hp.n_head_kv(il), head_dim, heads_per_group, sliding_window)
          elsif attn_splitk_enabled?(batch, sliding_window)
            encode_attention_context_rows_splitk(enc, scratch, q_buf, lstate.k_cache_buf, lstate.v_cache_buf, ctx_buf,
              base_pos, batch, hp.n_head, hp.n_head_kv(il), head_dim, heads_per_group)
          else
            encode_attention_context_rows(enc, q_buf, lstate.k_cache_buf, lstate.v_cache_buf, ctx_buf,
              base_pos, batch, hp.n_head, hp.n_head_kv(il), head_dim, heads_per_group, sliding_window)
          end
          true
        end

        return false unless profile_rows_phase("#{prefix}.attn_out") do |enc|
          if ctx_h16_buf
            Qwen35Metal.encode_matmul_from_h16_to_buffer(enc, lw.attn_output_qw, ctx_h16_buf, attn_projected_buf, batch)
          else
            Qwen35Metal.encode_matmul_to_buffer(enc, lw.attn_output_qw, ctx_buf, attn_projected_buf, batch)
          end
        end

        return false unless profile_rows_phase("#{prefix}.ffn_in") do |enc|
          encode_rmsnorm_rows_weighted_out(enc, attn_projected_buf, post_attn_w, attn_normed_buf, hidden_dim, batch, hp.rms_eps)
          encode_add_vec(enc, x_buf, attn_normed_buf, attn_out_buf, batch * hidden_dim)
          if ffn16 = ffn_in_h16_buf
            Qwen35Metal.encode_rmsnorm_rows_f32_h16_to_buffers(enc, attn_out_buf, ffn_w, ffn_in_buf, ffn16, hidden_dim, batch, hp.rms_eps)
          else
            encode_rmsnorm_rows_weighted_out(enc, attn_out_buf, ffn_w, ffn_in_buf, hidden_dim, batch, hp.rms_eps)
          end
          true
        end

        return false unless profile_rows_phase("#{prefix}.ffn_upgate") do |enc|
          fused_gelu = q4_gelu_fuse_enabled? &&
            Qwen35Metal.encode_q4k_gemm_h16_pair_b64_gelu_mul(enc, lw.ffn_gate_qw, lw.ffn_up_qw, ffn_in_buf, gate_buf, combined_buf, batch)
          unless fused_gelu
            ok = if ffn16 = ffn_in_h16_buf
                   Qwen35Metal.encode_matmul_from_h16_to_buffer(enc, lw.ffn_gate_qw, ffn16, gate_buf, batch) &&
                     Qwen35Metal.encode_matmul_from_h16_to_buffer(enc, lw.ffn_up_qw, ffn16, up_buf, batch)
                 else
                   (q4_pair_ffn_enabled? &&
                     Qwen35Metal.encode_q4k_gemm_h16_pair_to_buffers(enc, lw.ffn_gate_qw, lw.ffn_up_qw, ffn_in_buf, gate_buf, up_buf, batch)) ||
                     Qwen35Metal.encode_matmul_many_to_buffers(enc, [lw.ffn_gate_qw, lw.ffn_up_qw], ffn_in_buf, [gate_buf, up_buf], batch)
                 end
            next false unless ok
            encode_gelu_mul(enc, gate_buf, up_buf, combined_buf, batch * ffn_dim)
          end
          true
        end

        return false unless profile_rows_phase("#{prefix}.ffn_down") do |enc|
          Qwen35Metal.encode_matmul_to_buffer(enc, lw.ffn_down_qw, combined_buf, ffn_buf, batch)
        end

        scale = lw.layer_output_scale.first? || 1.0_f32
        profile_rows_phase("#{prefix}.ffn_out") do |enc|
          encode_rmsnorm_rows_weighted_out(enc, ffn_buf, post_ffw_w, ffn_normed_buf, hidden_dim, batch, hp.rms_eps)
          encode_add_scaled_vec(enc, attn_out_buf, ffn_normed_buf, out_buf, batch * hidden_dim, scale)
          true
        end
      end

      def encode_forward_layer_resident_cache_rows_to_buffer(enc : ML::Metal::ComputeEncoder,
                                                             weights : Gemma4Weights,
                                                             il : Int32,
                                                             x_buf : ML::MetalBuffer,
                                                             out_buf : ML::MetalBuffer,
                                                             base_pos : Int32,
                                                             batch : Int32,
                                                             state : ResidentState,
                                                             scratch : ResidentScratch = state.scratch) : Bool
        return false unless available?

        hp = weights.hparams
        lw = weights.layers[il]
        hidden_dim = hp.n_embd
        head_dim = hp.head_dim_for_layer(il)
        rope_dim = hp.rope_dim_for_layer(il)
        q_dim = hp.n_head * head_dim
        kv_dim = hp.n_head_kv(il) * head_dim
        ffn_dim = lw.ffn_gate_qw.out_dim
        hidden_bytes = batch.to_i64 * hidden_dim * sizeof(Float32)
        raise ArgumentError.new("batch must be positive") unless batch > 0
        raise ArgumentError.new("forward_layer_resident_cache_rows input buffer too small") if x_buf.size < hidden_bytes
        raise ArgumentError.new("forward_layer_resident_cache_rows output buffer too small") if out_buf.size < hidden_bytes
        raise ArgumentError.new("unsupported head_dim #{head_dim}; max 512") if head_dim > 512
        raise ArgumentError.new("rope_dim #{rope_dim} must be even") unless rope_dim.even?
        raise ArgumentError.new("base_pos #{base_pos} exceeds max_seq #{state.layers[il].max_seq}") if base_pos < 0 || base_pos + batch > state.layers[il].max_seq

        lstate = state.layers[il]
        raise ArgumentError.new("resident kv_dim mismatch") unless lstate.kv_dim == kv_dim
        kv_h16_cache = attn_kv_h16_cache_enabled?


        attn_norm_w = write_scratch_f32(scratch, "rows.attn_norm_w.#{il}", lw.attn_norm)
        post_attn_w = write_scratch_f32(scratch, "rows.post_attn_w.#{il}", lw.post_attention_norm)
        ffn_w = write_scratch_f32(scratch, "rows.ffn_w.#{il}", lw.ffn_norm)
        post_ffw_w = write_scratch_f32(scratch, "rows.post_ffw_w.#{il}", lw.post_ffw_norm)
        q_weight = write_scratch_f32(scratch, "rows.q_norm_w.#{il}", lw.attn_q_norm)
        k_weight = write_scratch_f32(scratch, "rows.k_norm_w.#{il}", lw.attn_k_norm)
        rope_freqs = hp.full_attention?(il) ? weights.rope_freqs : nil
        factors = (hp.full_attention?(il) && rope_freqs) ? rope_freqs.not_nil! : [1.0_f32]
        factor_buf = write_scratch_f32(scratch, "rows.rope_factors.#{il}", factors)
        use_factors = (hp.full_attention?(il) && rope_freqs) ? 1_u32 : 0_u32
        base = hp.rope_freq_base_for_layer(il)
        heads_per_group = hp.n_head // hp.n_head_kv(il)
        sliding_window = hp.sliding_window?(il) ? hp.sliding_window : 0

        x_norm_buf = scratch.get("rows.x_norm", hidden_bytes)
        norm_h16_proj = rmsnorm_h16_proj_enabled?(batch)
        x_norm_h16_buf = norm_h16_proj ? scratch.get("rows.x_norm_h16", batch.to_i64 * hidden_dim * 2_i64) : nil
        q_buf = scratch.get("rows.q", batch.to_i64 * q_dim * sizeof(Float32))
        k_buf = scratch.get("rows.k", batch.to_i64 * kv_dim * sizeof(Float32))
        v_buf = scratch.get("rows.v", batch.to_i64 * kv_dim * sizeof(Float32))
        ctx_buf = scratch.get("rows.ctx", batch.to_i64 * q_dim * sizeof(Float32))
        ctx_h16_oproj = attn_ctx_h16_oproj_enabled?(batch) && !kv_h16_cache
        ctx_h16_buf = ctx_h16_oproj ? scratch.get("rows.ctx_h16", batch.to_i64 * q_dim * 2_i64) : nil
        attn_projected_buf = scratch.get("rows.attn_projected", hidden_bytes)
        attn_normed_buf = scratch.get("rows.attn_normed", hidden_bytes)
        attn_out_buf = scratch.get("rows.attn_out", hidden_bytes)
        ffn_in_buf = scratch.get("rows.ffn_in", hidden_bytes)
        ffn_in_h16_buf = norm_h16_proj ? scratch.get("rows.ffn_in_h16", batch.to_i64 * hidden_dim * 2_i64) : nil
        gate_buf = scratch.get("rows.gate", batch.to_i64 * ffn_dim * sizeof(Float32))
        up_buf = scratch.get("rows.up", batch.to_i64 * ffn_dim * sizeof(Float32))
        combined_buf = scratch.get("rows.combined", batch.to_i64 * ffn_dim * sizeof(Float32))
        ffn_buf = scratch.get("rows.ffn", hidden_bytes)
        ffn_normed_buf = scratch.get("rows.ffn_normed", hidden_bytes)

        if x16 = x_norm_h16_buf
          Qwen35Metal.encode_rmsnorm_rows_f32_h16_to_buffers(enc, x_buf, attn_norm_w, x_norm_buf, x16, hidden_dim, batch, hp.rms_eps)
          projected = if v_qw = lw.attn_v_qw
                        Qwen35Metal.encode_matmul_from_h16_to_buffer(enc, lw.attn_q_qw, x16, q_buf, batch) &&
                          Qwen35Metal.encode_matmul_from_h16_to_buffer(enc, lw.attn_k_qw, x16, k_buf, batch) &&
                          Qwen35Metal.encode_matmul_from_h16_to_buffer(enc, v_qw, x16, v_buf, batch)
                      else
                        Qwen35Metal.encode_matmul_from_h16_to_buffer(enc, lw.attn_q_qw, x16, q_buf, batch) &&
                          Qwen35Metal.encode_matmul_from_h16_to_buffer(enc, lw.attn_k_qw, x16, k_buf, batch) &&
                          Qwen35Metal.encode_matmul_from_h16_to_buffer(enc, lw.attn_k_qw, x16, v_buf, batch)
                      end
          return false unless projected || Qwen35Metal.encode_matmul_many_to_buffers(enc, [lw.attn_q_qw, lw.attn_k_qw, (lw.attn_v_qw || lw.attn_k_qw)], x_norm_buf, [q_buf, k_buf, v_buf], batch)
        else
          encode_rmsnorm_rows_weighted_out(enc, x_buf, attn_norm_w, x_norm_buf, hidden_dim, batch, hp.rms_eps)
          if v_qw = lw.attn_v_qw
            Qwen35Metal.encode_matmul_many_to_buffers(enc, [lw.attn_q_qw, lw.attn_k_qw, v_qw], x_norm_buf, [q_buf, k_buf, v_buf], batch)
          else
            Qwen35Metal.encode_matmul_many_to_buffers(enc, [lw.attn_q_qw, lw.attn_k_qw, lw.attn_k_qw], x_norm_buf, [q_buf, k_buf, v_buf], batch)
          end
        end
        encode_rmsnorm_weighted_rows(enc, q_buf, q_weight, head_dim, hp.rms_eps, hp.n_head, batch)
        encode_rmsnorm_weighted_rows(enc, k_buf, k_weight, head_dim, hp.rms_eps, hp.n_head_kv(il), batch)
        encode_rmsnorm_plain_rows(enc, v_buf, head_dim, hp.rms_eps, hp.n_head_kv(il), batch)
        encode_rope_rows(enc, q_buf, factor_buf, head_dim, rope_dim, base_pos, base, use_factors, hp.n_head, batch)
        encode_rope_rows(enc, k_buf, factor_buf, head_dim, rope_dim, base_pos, base, use_factors, hp.n_head_kv(il), batch)
        encode_kv_write_rows(enc, k_buf, v_buf, lstate.k_cache_buf, lstate.v_cache_buf, base_pos, kv_dim, batch)
        if kv_h16_cache
          encode_kv_write_rows_h16(enc, k_buf, v_buf, lstate.k_cache_h16_buf.not_nil!, lstate.v_cache_h16_buf.not_nil!, base_pos, kv_dim, batch)
        end
        if kv_h16_cache
          encode_attention_context_rows_kv_h16(enc, q_buf, lstate.k_cache_h16_buf.not_nil!, lstate.v_cache_h16_buf.not_nil!, ctx_buf,
            base_pos, batch, hp.n_head, hp.n_head_kv(il), head_dim, heads_per_group, sliding_window)
          return false unless Qwen35Metal.encode_matmul_to_buffer(enc, lw.attn_output_qw, ctx_buf, attn_projected_buf, batch)
        elsif ctx_h16_buf
          encode_attention_context_rows_h16(enc, q_buf, lstate.k_cache_buf, lstate.v_cache_buf, ctx_h16_buf,
            base_pos, batch, hp.n_head, hp.n_head_kv(il), head_dim, heads_per_group, sliding_window)
          return false unless Qwen35Metal.encode_matmul_from_h16_to_buffer(enc, lw.attn_output_qw, ctx_h16_buf, attn_projected_buf, batch)
        elsif attn_splitk_enabled?(batch, sliding_window)
          encode_attention_context_rows_splitk(enc, scratch, q_buf, lstate.k_cache_buf, lstate.v_cache_buf, ctx_buf,
            base_pos, batch, hp.n_head, hp.n_head_kv(il), head_dim, heads_per_group)
          return false unless Qwen35Metal.encode_matmul_to_buffer(enc, lw.attn_output_qw, ctx_buf, attn_projected_buf, batch)
        else
          encode_attention_context_rows(enc, q_buf, lstate.k_cache_buf, lstate.v_cache_buf, ctx_buf,
            base_pos, batch, hp.n_head, hp.n_head_kv(il), head_dim, heads_per_group, sliding_window)
          return false unless Qwen35Metal.encode_matmul_to_buffer(enc, lw.attn_output_qw, ctx_buf, attn_projected_buf, batch)
        end

        encode_rmsnorm_rows_weighted_out(enc, attn_projected_buf, post_attn_w, attn_normed_buf, hidden_dim, batch, hp.rms_eps)
        encode_add_vec(enc, x_buf, attn_normed_buf, attn_out_buf, batch * hidden_dim)
        if ffn16 = ffn_in_h16_buf
          Qwen35Metal.encode_rmsnorm_rows_f32_h16_to_buffers(enc, attn_out_buf, ffn_w, ffn_in_buf, ffn16, hidden_dim, batch, hp.rms_eps)
        else
          encode_rmsnorm_rows_weighted_out(enc, attn_out_buf, ffn_w, ffn_in_buf, hidden_dim, batch, hp.rms_eps)
        end
        fused_gelu = q4_gelu_fuse_enabled? &&
          Qwen35Metal.encode_q4k_gemm_h16_pair_b64_gelu_mul(enc, lw.ffn_gate_qw, lw.ffn_up_qw, ffn_in_buf, gate_buf, combined_buf, batch)
        unless fused_gelu
          ok = if ffn16 = ffn_in_h16_buf
                 Qwen35Metal.encode_matmul_from_h16_to_buffer(enc, lw.ffn_gate_qw, ffn16, gate_buf, batch) &&
                   Qwen35Metal.encode_matmul_from_h16_to_buffer(enc, lw.ffn_up_qw, ffn16, up_buf, batch)
               else
                 pair_q4 = q4_pair_ffn_enabled? &&
                   Qwen35Metal.encode_q4k_gemm_h16_pair_to_buffers(enc, lw.ffn_gate_qw, lw.ffn_up_qw, ffn_in_buf, gate_buf, up_buf, batch)
                 pair_q4 || Qwen35Metal.encode_matmul_many_to_buffers(enc, [lw.ffn_gate_qw, lw.ffn_up_qw], ffn_in_buf, [gate_buf, up_buf], batch)
               end
          return false unless ok
          encode_gelu_mul(enc, gate_buf, up_buf, combined_buf, batch * ffn_dim)
        end
        return false unless Qwen35Metal.encode_matmul_to_buffer(enc, lw.ffn_down_qw, combined_buf, ffn_buf, batch)
        scale = lw.layer_output_scale.first? || 1.0_f32
        encode_rmsnorm_rows_weighted_out(enc, ffn_buf, post_ffw_w, ffn_normed_buf, hidden_dim, batch, hp.rms_eps)
        encode_add_scaled_vec(enc, attn_out_buf, ffn_normed_buf, out_buf, batch * hidden_dim, scale)
        true
      end

      def forward_layer_resident_cache_rows(weights : Gemma4Weights,
                                            il : Int32,
                                            x_rows : Array(Float32),
                                            base_pos : Int32,
                                            batch : Int32,
                                            state : ResidentState) : Array(Float32)?
        return nil unless available?

        hidden_dim = weights.hparams.n_embd
        raise ArgumentError.new("forward_layer_resident_cache_rows input size mismatch") unless x_rows.size == batch * hidden_dim

        x_buf = ML::MetalBuffer.from_array(x_rows)
        out_buf = ML::MetalBuffer.new(batch.to_i64 * hidden_dim * sizeof(Float32))
        cmd = ML::Metal::CommandBuffer.new
        enc = ML::Metal::ComputeEncoder.new(cmd)
        ok = encode_forward_layer_resident_cache_rows_to_buffer(enc, weights, il, x_buf, out_buf, base_pos, batch, state)
        enc.end_encoding
        return nil unless ok
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
        # Exact default remains clamped to the Qwen GEMM threshold below.
        # When the operator explicitly allows GEMM drift, use the measured
        # Gemma4 row-prefill sweet spot unless overridden.
        default_chunk_cap = ENV["GEMMA4_ROW_PREFILL_ALLOW_GEMM"]? == "1" ? 512 : 8
        exact_chunk_cap = ENV["GEMMA4_ROW_PREFILL_EXACT_CHUNK_MAX"]?.try(&.to_i?) || default_chunk_cap
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

          if row_prefill_resident_corridor_enabled?
            hidden_bytes = batch.to_i64 * hidden_dim * sizeof(Float32)
            in_buf = ML::MetalBuffer.new(hidden_bytes)
            out_buf = ML::MetalBuffer.new(hidden_bytes)
            embed_t0 = Time.instant if Qwen35Metal::Profile.enabled?
            Qwen35Metal.embedding_q6k_rows_scaled_to_buffer(weights.token_embd, token_ids[offset, batch], in_buf, scale)
            if Qwen35Metal::Profile.enabled?
              embed_done = Time.instant
              Qwen35Metal::Profile.bump_group("gemma4.rows.embedding", 0_i64,
                (embed_done - embed_t0.not_nil!).total_nanoseconds.to_i64, 0_i64)
              Qwen35Metal::Profile.bump_group_transfer("gemma4.rows.embedding", token_ids.size.to_i64 * sizeof(UInt32), 0_i64)
            end

            if Qwen35Metal::Profile.enabled? && row_prefill_profile_layers_enabled?
              layer_count.times do |il|
                if row_prefill_profile_phases_enabled?
                  ok = forward_layer_resident_cache_rows_profile_phases_to_buffer(weights, il, in_buf, out_buf, base_pos, batch, state)
                  return nil unless ok
                else
                  layer_t0 = Time.instant
                  cmd = ML::Metal::CommandBuffer.new
                  enc = ML::Metal::ComputeEncoder.new(cmd)
                  ok = encode_forward_layer_resident_cache_rows_to_buffer(enc, weights, il, in_buf, out_buf, base_pos, batch, state)
                  unless ok
                    enc.end_encoding
                    return nil
                  end
                  enc.end_encoding
                  layer_tenc = Time.instant
                  cmd.commit
                  cmd.wait
                  layer_wait = Time.instant
                  Qwen35Metal::Profile.bump_group("gemma4.rows.layer#{il}",
                    (layer_tenc - layer_t0).total_nanoseconds.to_i64,
                    (layer_wait - layer_tenc).total_nanoseconds.to_i64,
                    0_i64)
                end
                in_buf, out_buf = out_buf, in_buf
              end
            else
              layer_t0 = Time.instant if Qwen35Metal::Profile.enabled?
              cmd = ML::Metal::CommandBuffer.new
              enc = ML::Metal::ComputeEncoder.new(cmd)
              layer_count.times do |il|
                ok = encode_forward_layer_resident_cache_rows_to_buffer(enc, weights, il, in_buf, out_buf, base_pos, batch, state)
                unless ok
                  enc.end_encoding
                  return nil
                end
                in_buf, out_buf = out_buf, in_buf
              end
              enc.end_encoding
              layer_tenc = Time.instant if Qwen35Metal::Profile.enabled?
              cmd.commit
              cmd.wait
              layer_wait = Time.instant if Qwen35Metal::Profile.enabled?
              if Qwen35Metal::Profile.enabled?
                Qwen35Metal::Profile.bump_group("gemma4.rows.layers",
                  (layer_tenc.not_nil! - layer_t0.not_nil!).total_nanoseconds.to_i64,
                  (layer_wait.not_nil! - layer_tenc.not_nil!).total_nanoseconds.to_i64,
                  0_i64)
              end
            end
            read_t0 = Time.instant if Qwen35Metal::Profile.enabled?
            x_rows = in_buf.read(batch * hidden_dim)
            if Qwen35Metal::Profile.enabled?
              read_done = Time.instant
              Qwen35Metal::Profile.bump_group("gemma4.rows.read", 0_i64, 0_i64,
                (read_done - read_t0.not_nil!).total_nanoseconds.to_i64)
              Qwen35Metal::Profile.bump_group_transfer("gemma4.rows.read", 0_i64, hidden_bytes)
            end
          else
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

      def forward_hidden_resident_cache_wave(weights : Gemma4Weights,
                                             token_id : Int32,
                                             pos : Int32,
                                             state : ResidentState,
                                             stop_layer : Int32? = nil) : Array(Float32)?
        return nil unless available?

        hp = weights.hparams
        hidden_dim = hp.n_embd
        hidden_bytes = hidden_dim.to_i64 * sizeof(Float32)
        raise ArgumentError.new("position #{pos} exceeds max_seq #{state.max_seq}") if pos < 0 || pos >= state.max_seq

        scale = Math.sqrt(hidden_dim.to_f64).to_f32
        in_buf = ML::MetalBuffer.new(hidden_bytes)
        out_buf = ML::MetalBuffer.new(hidden_bytes)
        embed_t0 = Time.instant if Qwen35Metal::Profile.enabled?
        Qwen35Metal.embedding_q6k_rows_scaled_to_buffer(weights.token_embd, [token_id], in_buf, scale)
        if Qwen35Metal::Profile.enabled?
          embed_done = Time.instant
          Qwen35Metal::Profile.bump_group("gemma4.decode_wave.embedding", 0_i64,
            (embed_done - embed_t0.not_nil!).total_nanoseconds.to_i64, 0_i64)
          Qwen35Metal::Profile.bump_group_transfer("gemma4.decode_wave.embedding", sizeof(UInt32).to_i64, 0_i64)
        end

        layer_count = stop_layer ? Math.min(stop_layer.not_nil!, weights.layers.size) : weights.layers.size
        if Qwen35Metal::Profile.enabled? && decode_profile_phases_enabled?
          layer_count.times do |il|
            ok = forward_layer_resident_cache_rows_profile_phases_to_buffer(weights, il, in_buf, out_buf, pos, 1, state)
            return nil unless ok
            in_buf, out_buf = out_buf, in_buf
          end
          read_t0 = Time.instant
          result = in_buf.read(hidden_dim)
          read_done = Time.instant
          Qwen35Metal::Profile.bump_group("gemma4.decode_wave.read", 0_i64, 0_i64,
            (read_done - read_t0).total_nanoseconds.to_i64)
          Qwen35Metal::Profile.bump_group_transfer("gemma4.decode_wave.read", 0_i64, hidden_bytes)
          return result
        end

        layer_t0 = Time.instant if Qwen35Metal::Profile.enabled?
        cmd = ML::Metal::CommandBuffer.new
        enc = ML::Metal::ComputeEncoder.new(cmd)
        layer_count.times do |il|
          ok = encode_forward_layer_resident_cache_rows_to_buffer(enc, weights, il, in_buf, out_buf, pos, 1, state)
          unless ok
            enc.end_encoding
            return nil
          end
          in_buf, out_buf = out_buf, in_buf
        end
        enc.end_encoding
        layer_tenc = Time.instant if Qwen35Metal::Profile.enabled?
        cmd.commit
        cmd.wait
        layer_wait = Time.instant if Qwen35Metal::Profile.enabled?
        if Qwen35Metal::Profile.enabled?
          Qwen35Metal::Profile.bump_group("gemma4.decode_wave.layers",
            (layer_tenc.not_nil! - layer_t0.not_nil!).total_nanoseconds.to_i64,
            (layer_wait.not_nil! - layer_tenc.not_nil!).total_nanoseconds.to_i64,
            0_i64)
        end
        read_t0 = Time.instant if Qwen35Metal::Profile.enabled?
        result = in_buf.read(hidden_dim)
        if Qwen35Metal::Profile.enabled?
          read_done = Time.instant
          Qwen35Metal::Profile.bump_group("gemma4.decode_wave.read", 0_i64, 0_i64,
            (read_done - read_t0.not_nil!).total_nanoseconds.to_i64)
          Qwen35Metal::Profile.bump_group_transfer("gemma4.decode_wave.read", 0_i64, hidden_bytes)
        end
        result
      rescue ex
        if ENV["GEMMA4_RESIDENT_LAYER_STRICT"]? == "1"
          raise ex
        else
          nil
        end
      end

      def forward_top1_resident_cache_wave(weights : Gemma4Weights,
                                           token_id : Int32,
                                           pos : Int32,
                                           state : ResidentState,
                                           stop_layer : Int32? = nil) : Int32?
        return nil unless available?

        hp = weights.hparams
        hidden_dim = hp.n_embd
        hidden_bytes = hidden_dim.to_i64 * sizeof(Float32)
        raise ArgumentError.new("position #{pos} exceeds max_seq #{state.max_seq}") if pos < 0 || pos >= state.max_seq

        scale = Math.sqrt(hidden_dim.to_f64).to_f32
        scratch = state.scratch
        in_buf = scratch.get("decode.top1.in", hidden_bytes)
        out_buf = scratch.get("decode.top1.out", hidden_bytes)
        embed_t0 = Time.instant if Qwen35Metal::Profile.enabled?
        Qwen35Metal.embedding_q6k_rows_scaled_to_buffer(weights.token_embd, [token_id], in_buf, scale)
        if Qwen35Metal::Profile.enabled?
          embed_done = Time.instant
          Qwen35Metal::Profile.bump_group("gemma4.decode_wave_top1.embedding", 0_i64,
            (embed_done - embed_t0.not_nil!).total_nanoseconds.to_i64, 0_i64)
          Qwen35Metal::Profile.bump_group_transfer("gemma4.decode_wave_top1.embedding", sizeof(UInt32).to_i64, 0_i64)
        end

        layer_count = stop_layer ? Math.min(stop_layer.not_nil!, weights.layers.size) : weights.layers.size
        norm_w_buf = write_scratch_f32(scratch, "decode.top1.output_norm", weights.output_norm)
        normed_buf = scratch.get("decode.top1.normed", hidden_bytes)
        tile_count = Qwen35Metal.head_top1_tile_count(weights.token_embd)
        tile_values_buf = scratch.get("decode.top1.tile_values", tile_count.to_i64 * sizeof(Float32))
        tile_ids_buf = scratch.get("decode.top1.tile_ids", tile_count.to_i64 * sizeof(UInt32))
        top1_id_buf = scratch.get("decode.top1.id", sizeof(UInt32).to_i64)
        top1_value_buf = scratch.get("decode.top1.value", sizeof(Float32).to_i64)

        t0 = Time.instant if Qwen35Metal::Profile.enabled?
        cmd = ML::Metal::CommandBuffer.new
        enc = ML::Metal::ComputeEncoder.new(cmd)
        layer_count.times do |il|
          ok = encode_forward_layer_resident_cache_rows_to_buffer(enc, weights, il, in_buf, out_buf, pos, 1, state)
          unless ok
            enc.end_encoding
            return nil
          end
          in_buf, out_buf = out_buf, in_buf
        end
        encode_rmsnorm_weighted_out(enc, in_buf, norm_w_buf, normed_buf, hidden_dim, hp.rms_eps)
        unless Qwen35Metal.encode_head_top1_no_norm_to_buffers(enc, weights.token_embd, normed_buf, tile_values_buf, tile_ids_buf, top1_id_buf, top1_value_buf)
          enc.end_encoding
          return nil
        end
        enc.end_encoding
        t_enc = Time.instant if Qwen35Metal::Profile.enabled?
        cmd.commit
        cmd.wait
        t_wait = Time.instant if Qwen35Metal::Profile.enabled?
        top1 = Qwen35Metal.read_head_top1_buffers(top1_id_buf, top1_value_buf)
        if Qwen35Metal::Profile.enabled?
          t_read = Time.instant
          Qwen35Metal::Profile.bump_group("gemma4.decode_wave_top1.layers_head",
            (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
            (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
            (t_read - t_wait.not_nil!).total_nanoseconds.to_i64)
          Qwen35Metal::Profile.bump_group_transfer("gemma4.decode_wave_top1.layers_head", 0_i64, sizeof(UInt32).to_i64 + sizeof(Float32).to_i64)
        end
        top1[0].to_i32
      rescue ex
        if ENV["GEMMA4_RESIDENT_LAYER_STRICT"]? == "1"
          raise ex
        else
          nil
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
        use_gqa_pair = attn_gqa2_enabled? &&
          (heads_per_group == 2 || (attn_gqa_pair_full_enabled? && heads_per_group > 2 && heads_per_group.even?))
        if use_gqa_pair
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
