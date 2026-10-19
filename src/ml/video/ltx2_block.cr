# Adapted from Lightricks/LTX-2 BasicAVTransformerBlock at
# e98f6d927d83c92ef2be6aec76c8440b19306e39. Changed for native Metal/Crystal;
# covered by the LTX-2.x Community License (LICENSE-2_x).
require "./metal_ops"
require "./checkpoint"

module ML::Video
  # One dual-stream block, batch size one, with text cross-attention and the
  # two AV-cross directions enabled. AdaLN values and split-RoPE tables are
  # supplied as resident buffers. In particular, RoPE here is a diagnostic
  # conditioning input: the native on-GPU positional encoder is still open.
  record LTX2DualStreamConfig,
    video_dim : Int32,
    video_context_dim : Int32,
    video_context_tokens : Int32,
    video_tokens : Int32,
    video_heads : Int32,
    video_head_dim : Int32,
    audio_dim : Int32,
    audio_context_dim : Int32,
    audio_context_tokens : Int32,
    audio_tokens : Int32,
    audio_heads : Int32,
    audio_head_dim : Int32,
    video_modulation_rows : Int32 = 1,
    audio_modulation_rows : Int32 = 1,
    video_prompt_modulation_rows : Int32 = 1,
    audio_prompt_modulation_rows : Int32 = 1,
    cross_attention_adaln : Bool = false,
    video_apply_gated_attention : Bool = false,
    audio_apply_gated_attention : Bool = false,
    video_ff_bias : Bool = true,
    audio_ff_bias : Bool = true,
    norm_eps : Float32 = 1e-6_f32

  class LTX2DualStreamBlock
    # Map the exact BasicAVTransformerBlock state-dict keys consumed by this
    # native slice. The returned targets are Arena keys; the source names are
    # upstream block.state_dict() keys. The model shard's block prefix is not
    # inferred here because gated checkpoint metadata has not been admitted.
    def self.checkpoint_bindings(config : LTX2DualStreamConfig,
                                 dtype : ML::GGUF::SafeTensorDType? = nil) : Array(WeightBinding)
      bindings = [] of WeightBinding
      append_attention_bindings(bindings, "attn1", config.video_dim, config.video_dim,
        config.video_heads, config.video_head_dim, config.video_apply_gated_attention, dtype)
      append_attention_bindings(bindings, "attn2", config.video_dim, config.video_context_dim,
        config.video_heads, config.video_head_dim, config.video_apply_gated_attention, dtype)
      append_attention_bindings(bindings, "audio_to_video_attn", config.video_dim, config.audio_dim,
        config.audio_heads, config.audio_head_dim, config.video_apply_gated_attention, dtype)
      append_attention_bindings(bindings, "audio_attn1", config.audio_dim, config.audio_dim,
        config.audio_heads, config.audio_head_dim, config.audio_apply_gated_attention, dtype)
      append_attention_bindings(bindings, "audio_attn2", config.audio_dim, config.audio_context_dim,
        config.audio_heads, config.audio_head_dim, config.audio_apply_gated_attention, dtype)
      append_attention_bindings(bindings, "video_to_audio_attn", config.audio_dim, config.video_dim,
        config.audio_heads, config.audio_head_dim, config.audio_apply_gated_attention, dtype)
      append_ff_bindings(bindings, "ff", config.video_dim, config.video_ff_bias, dtype)
      append_ff_bindings(bindings, "audio_ff", config.audio_dim, config.audio_ff_bias, dtype)
      bindings
    end

    private def self.append_attention_bindings(bindings : Array(WeightBinding), name : String,
                                               query_dim : Int32, context_dim : Int32,
                                               heads : Int32, head_dim : Int32,
                                               gated : Bool,
                                               dtype : ML::GGUF::SafeTensorDType?) : Nil
      inner = heads.to_i64 * head_dim
      q_dim = query_dim.to_i64
      c_dim = context_dim.to_i64
      add_binding(bindings, name, "to_q.weight", [inner, q_dim], dtype)
      add_binding(bindings, name, "to_q.bias", [inner], dtype)
      add_binding(bindings, name, "to_k.weight", [inner, c_dim], dtype)
      add_binding(bindings, name, "to_k.bias", [inner], dtype)
      add_binding(bindings, name, "to_v.weight", [inner, c_dim], dtype)
      add_binding(bindings, name, "to_v.bias", [inner], dtype)
      add_binding(bindings, name, "q_norm.weight", [inner], dtype)
      add_binding(bindings, name, "k_norm.weight", [inner], dtype)
      add_binding(bindings, name, "to_out.0.weight", [q_dim, inner], dtype)
      add_binding(bindings, name, "to_out.0.bias", [q_dim], dtype)
      if gated
        add_binding(bindings, name, "to_gate_logits.weight", [heads.to_i64, q_dim], dtype)
        add_binding(bindings, name, "to_gate_logits.bias", [heads.to_i64], dtype)
      end
    end

    private def self.append_ff_bindings(bindings : Array(WeightBinding), name : String,
                                        dim : Int32, bias : Bool,
                                        dtype : ML::GGUF::SafeTensorDType?) : Nil
      width = 4_i64 * dim
      d = dim.to_i64
      add_binding(bindings, name, "net.0.proj.weight", [width, d], dtype)
      add_binding(bindings, name, "net.2.weight", [d, width], dtype)
      if bias
        add_binding(bindings, name, "net.0.proj.bias", [width], dtype)
        add_binding(bindings, name, "net.2.bias", [d], dtype)
      end
    end

    private def self.add_binding(bindings : Array(WeightBinding), prefix : String,
                                 suffix : String, shape : Array(Int64),
                                 dtype : ML::GGUF::SafeTensorDType?) : Nil
      source = "#{prefix}.#{suffix}"
      bindings << WeightBinding.new("w.#{source}", source, shape, dtype)
    end

    struct AttentionWeights
      getter query : ML::MetalBuffer
      getter key : ML::MetalBuffer
      getter value : ML::MetalBuffer
      getter query_norm : ML::MetalBuffer
      getter key_norm : ML::MetalBuffer
      getter projection : ML::MetalBuffer
      getter query_bias : ML::MetalBuffer
      getter key_bias : ML::MetalBuffer
      getter value_bias : ML::MetalBuffer
      getter projection_bias : ML::MetalBuffer
      getter gate : ML::MetalBuffer?
      getter gate_bias : ML::MetalBuffer?

      def initialize(@query, @key, @value, @query_norm, @key_norm, @projection,
                     @query_bias, @key_bias, @value_bias, @projection_bias,
                     @gate, @gate_bias)
      end
    end

    struct AttentionScratch
      getter query : ML::MetalBuffer
      getter key : ML::MetalBuffer
      getter value : ML::MetalBuffer
      getter query_norm : ML::MetalBuffer
      getter key_norm : ML::MetalBuffer
      getter query_rope : ML::MetalBuffer
      getter key_rope : ML::MetalBuffer
      getter attended : ML::MetalBuffer
      getter module_output : ML::MetalBuffer
      getter gate_logits : ML::MetalBuffer?
      getter gated_attended : ML::MetalBuffer?

      def initialize(@query, @key, @value, @query_norm, @key_norm, @query_rope,
                     @key_rope, @attended, @module_output, @gate_logits,
                     @gated_attended)
      end
    end

    struct AttentionLayer
      getter weights : AttentionWeights
      getter scratch : AttentionScratch
      getter query_rows : Int32
      getter key_rows : Int32
      getter query_dim : Int32
      getter context_dim : Int32
      getter heads : Int32
      getter head_dim : Int32

      def initialize(@weights, @scratch, @query_rows, @key_rows, @query_dim,
                     @context_dim, @heads, @head_dim)
      end

      def inner_dim : Int32
        @heads * @head_dim
      end
    end

    struct FeedForwardWeights
      getter input : ML::MetalBuffer
      getter output : ML::MetalBuffer
      getter input_bias : ML::MetalBuffer
      getter output_bias : ML::MetalBuffer

      def initialize(@input, @output, @input_bias, @output_bias)
      end
    end

    getter checkpoints : Hash(String, ML::MetalBuffer)
    getter video_output : ML::MetalBuffer
    getter audio_output : ML::MetalBuffer

    SOURCE = {{ read_file("#{__DIR__}/kernels/ltx2.metal") }}

    @zero : ML::MetalBuffer
    @rope_pipeline : ML::Metal::ComputePipeline
    @gate_pipeline : ML::Metal::ComputePipeline
    @video_sa : AttentionLayer
    @video_text : AttentionLayer
    @video_a2v : AttentionLayer
    @audio_sa : AttentionLayer
    @audio_text : AttentionLayer
    @audio_v2a : AttentionLayer
    @video_ff_weights : FeedForwardWeights
    @audio_ff_weights : FeedForwardWeights

    @video_text_shift : ML::MetalBuffer?
    @video_text_scale : ML::MetalBuffer?
    @video_text_gate : ML::MetalBuffer?
    @video_prompt_shift : ML::MetalBuffer?
    @video_prompt_scale : ML::MetalBuffer?
    @audio_text_shift : ML::MetalBuffer?
    @audio_text_scale : ML::MetalBuffer?
    @audio_text_gate : ML::MetalBuffer?
    @audio_prompt_shift : ML::MetalBuffer?
    @audio_prompt_scale : ML::MetalBuffer?

    @video_x : ML::MetalBuffer
    @audio_x : ML::MetalBuffer
    @video_context : ML::MetalBuffer
    @audio_context : ML::MetalBuffer
    @video_sa_shift : ML::MetalBuffer
    @video_sa_scale : ML::MetalBuffer
    @video_sa_gate : ML::MetalBuffer
    @video_ff_shift : ML::MetalBuffer
    @video_ff_scale : ML::MetalBuffer
    @video_ff_gate : ML::MetalBuffer
    @audio_sa_shift : ML::MetalBuffer
    @audio_sa_scale : ML::MetalBuffer
    @audio_sa_gate : ML::MetalBuffer
    @audio_ff_shift : ML::MetalBuffer
    @audio_ff_scale : ML::MetalBuffer
    @audio_ff_gate : ML::MetalBuffer
    @a2v_video_shift : ML::MetalBuffer
    @a2v_video_scale : ML::MetalBuffer
    @a2v_video_gate : ML::MetalBuffer
    @a2v_audio_shift : ML::MetalBuffer
    @a2v_audio_scale : ML::MetalBuffer
    @v2a_audio_shift : ML::MetalBuffer
    @v2a_audio_scale : ML::MetalBuffer
    @v2a_audio_gate : ML::MetalBuffer
    @v2a_video_shift : ML::MetalBuffer
    @v2a_video_scale : ML::MetalBuffer

    @video_self_cos : ML::MetalBuffer
    @video_self_sin : ML::MetalBuffer
    @audio_self_cos : ML::MetalBuffer
    @audio_self_sin : ML::MetalBuffer
    @video_cross_cos : ML::MetalBuffer
    @video_cross_sin : ML::MetalBuffer
    @audio_cross_cos : ML::MetalBuffer
    @audio_cross_sin : ML::MetalBuffer

    @video_sa_norm : ML::MetalBuffer
    @video_sa_modulated : ML::MetalBuffer
    @video_after_sa : ML::MetalBuffer
    @video_text_norm : ML::MetalBuffer
    @video_text_query_modulated : ML::MetalBuffer
    @video_text_context_modulated : ML::MetalBuffer
    @video_after_text : ML::MetalBuffer
    @video_a2v_query_norm : ML::MetalBuffer
    @video_a2v_query_modulated : ML::MetalBuffer
    @video_a2v_context_norm : ML::MetalBuffer
    @video_a2v_context_modulated : ML::MetalBuffer
    @video_after_av : ML::MetalBuffer
    @video_ff_norm : ML::MetalBuffer
    @video_ff_modulated : ML::MetalBuffer
    @video_ff_hidden : ML::MetalBuffer
    @video_ff_activated : ML::MetalBuffer
    @video_ff_module_output : ML::MetalBuffer
    @video_output : ML::MetalBuffer

    @audio_sa_norm : ML::MetalBuffer
    @audio_sa_modulated : ML::MetalBuffer
    @audio_after_sa : ML::MetalBuffer
    @audio_text_norm : ML::MetalBuffer
    @audio_text_query_modulated : ML::MetalBuffer
    @audio_text_context_modulated : ML::MetalBuffer
    @audio_after_text : ML::MetalBuffer
    @audio_v2a_query_norm : ML::MetalBuffer
    @audio_v2a_query_modulated : ML::MetalBuffer
    @audio_v2a_context_norm : ML::MetalBuffer
    @audio_v2a_context_modulated : ML::MetalBuffer
    @audio_after_av : ML::MetalBuffer
    @audio_ff_norm : ML::MetalBuffer
    @audio_ff_modulated : ML::MetalBuffer
    @audio_ff_hidden : ML::MetalBuffer
    @audio_ff_activated : ML::MetalBuffer
    @audio_ff_module_output : ML::MetalBuffer
    @audio_output : ML::MetalBuffer

    def initialize(@arena : Arena, @config : LTX2DualStreamConfig, prefix : String = "ltx2")
      validate_config

      @video_x = input("video_x")
      @audio_x = input("audio_x")
      @video_context = input("video_context")
      @audio_context = input("audio_context")
      if @config.cross_attention_adaln
        @video_text_shift = input("video_text_shift")
        @video_text_scale = input("video_text_scale")
        @video_text_gate = input("video_text_gate")
        @video_prompt_shift = input("video_prompt_shift")
        @video_prompt_scale = input("video_prompt_scale")
        @audio_text_shift = input("audio_text_shift")
        @audio_text_scale = input("audio_text_scale")
        @audio_text_gate = input("audio_text_gate")
        @audio_prompt_shift = input("audio_prompt_shift")
        @audio_prompt_scale = input("audio_prompt_scale")
      end
      @video_sa_shift = input("video_sa_shift")
      @video_sa_scale = input("video_sa_scale")
      @video_sa_gate = input("video_sa_gate")
      @video_ff_shift = input("video_ff_shift")
      @video_ff_scale = input("video_ff_scale")
      @video_ff_gate = input("video_ff_gate")
      @audio_sa_shift = input("audio_sa_shift")
      @audio_sa_scale = input("audio_sa_scale")
      @audio_sa_gate = input("audio_sa_gate")
      @audio_ff_shift = input("audio_ff_shift")
      @audio_ff_scale = input("audio_ff_scale")
      @audio_ff_gate = input("audio_ff_gate")
      @a2v_video_shift = input("a2v_video_shift")
      @a2v_video_scale = input("a2v_video_scale")
      @a2v_video_gate = input("a2v_video_gate")
      @a2v_audio_shift = input("a2v_audio_shift")
      @a2v_audio_scale = input("a2v_audio_scale")
      @v2a_audio_shift = input("v2a_audio_shift")
      @v2a_audio_scale = input("v2a_audio_scale")
      @v2a_audio_gate = input("v2a_audio_gate")
      @v2a_video_shift = input("v2a_video_shift")
      @v2a_video_scale = input("v2a_video_scale")

      @video_self_cos = input("video_self_cos")
      @video_self_sin = input("video_self_sin")
      @audio_self_cos = input("audio_self_cos")
      @audio_self_sin = input("audio_self_sin")
      @video_cross_cos = input("video_cross_cos")
      @video_cross_sin = input("video_cross_sin")
      @audio_cross_cos = input("audio_cross_cos")
      @audio_cross_sin = input("audio_cross_sin")

      max_dim = [@config.video_dim * 4, @config.audio_dim * 4,
                 @config.video_context_dim, @config.audio_context_dim,
                 @config.video_heads * @config.video_head_dim,
                 @config.audio_heads * @config.audio_head_dim].max
      @zero = @arena.alloc("#{prefix}.zero", max_dim)
      @zero.write(Array(Float32).new(max_dim, 0.0_f32))
      @rope_pipeline = ML::Metal::ComputePipeline.new("ltx2_split_rope_f32", SOURCE)
      @gate_pipeline = ML::Metal::PipelineCache.get("ltx2_apply_head_gates") do
        ML::Metal::ComputePipeline.new("ltx2_apply_head_gates_f32", SOURCE)
      end

      @video_sa = build_attention(prefix, "video.attn1", "attn1", @config.video_tokens,
        @config.video_tokens, @config.video_dim, @config.video_dim,
        @config.video_heads, @config.video_head_dim, @config.video_apply_gated_attention)
      @video_text = build_attention(prefix, "video.attn2", "attn2", @config.video_tokens,
        @config.video_context_tokens, @config.video_dim, @config.video_context_dim,
        @config.video_heads, @config.video_head_dim, @config.video_apply_gated_attention)
      @video_a2v = build_attention(prefix, "video.audio_to_video_attn", "audio_to_video_attn",
        @config.video_tokens, @config.audio_tokens, @config.video_dim, @config.audio_dim,
        @config.audio_heads, @config.audio_head_dim, @config.video_apply_gated_attention)
      @audio_sa = build_attention(prefix, "audio.audio_attn1", "audio_attn1", @config.audio_tokens,
        @config.audio_tokens, @config.audio_dim, @config.audio_dim,
        @config.audio_heads, @config.audio_head_dim, @config.audio_apply_gated_attention)
      @audio_text = build_attention(prefix, "audio.audio_attn2", "audio_attn2", @config.audio_tokens,
        @config.audio_context_tokens, @config.audio_dim, @config.audio_context_dim,
        @config.audio_heads, @config.audio_head_dim, @config.audio_apply_gated_attention)
      @audio_v2a = build_attention(prefix, "audio.video_to_audio_attn", "video_to_audio_attn",
        @config.audio_tokens, @config.video_tokens, @config.audio_dim, @config.video_dim,
        @config.audio_heads, @config.audio_head_dim, @config.audio_apply_gated_attention)
      @video_ff_weights = build_ff_weights("ff", @config.video_dim, @config.video_ff_bias)
      @audio_ff_weights = build_ff_weights("audio_ff", @config.audio_dim, @config.audio_ff_bias)

      @video_sa_norm = alloc(prefix, "video.sa_norm", @config.video_tokens * @config.video_dim)
      @video_sa_modulated = alloc(prefix, "video.sa_modulated", @config.video_tokens * @config.video_dim)
      @video_after_sa = alloc(prefix, "video.after_attn1", @config.video_tokens * @config.video_dim)
      @video_text_norm = alloc(prefix, "video.text_norm", @config.video_tokens * @config.video_dim)
      @video_text_query_modulated = alloc(prefix, "video.text_query_modulated", @config.video_tokens * @config.video_dim)
      @video_text_context_modulated = alloc(prefix, "video.text_context_modulated", @config.video_context_tokens * @config.video_context_dim)
      @video_after_text = alloc(prefix, "video.after_attn2", @config.video_tokens * @config.video_dim)
      @video_a2v_query_norm = alloc(prefix, "video.a2v_query_norm", @config.video_tokens * @config.video_dim)
      @video_a2v_query_modulated = alloc(prefix, "video.a2v_query_modulated", @config.video_tokens * @config.video_dim)
      @video_a2v_context_norm = alloc(prefix, "video.a2v_context_norm", @config.audio_tokens * @config.audio_dim)
      @video_a2v_context_modulated = alloc(prefix, "video.a2v_context_modulated", @config.audio_tokens * @config.audio_dim)
      @video_after_av = alloc(prefix, "video.after_av", @config.video_tokens * @config.video_dim)
      @video_ff_norm = alloc(prefix, "video.ff_norm", @config.video_tokens * @config.video_dim)
      @video_ff_modulated = alloc(prefix, "video.ff_modulated", @config.video_tokens * @config.video_dim)
      @video_ff_hidden = alloc(prefix, "video.ff_hidden", @config.video_tokens * @config.video_dim * 4)
      @video_ff_activated = alloc(prefix, "video.ff_activated", @config.video_tokens * @config.video_dim * 4)
      @video_ff_module_output = alloc(prefix, "video.ff_output", @config.video_tokens * @config.video_dim)
      @video_output = alloc(prefix, "video.output", @config.video_tokens * @config.video_dim)

      @audio_sa_norm = alloc(prefix, "audio.sa_norm", @config.audio_tokens * @config.audio_dim)
      @audio_sa_modulated = alloc(prefix, "audio.sa_modulated", @config.audio_tokens * @config.audio_dim)
      @audio_after_sa = alloc(prefix, "audio.after_audio_attn1", @config.audio_tokens * @config.audio_dim)
      @audio_text_norm = alloc(prefix, "audio.text_norm", @config.audio_tokens * @config.audio_dim)
      @audio_text_query_modulated = alloc(prefix, "audio.text_query_modulated", @config.audio_tokens * @config.audio_dim)
      @audio_text_context_modulated = alloc(prefix, "audio.text_context_modulated", @config.audio_context_tokens * @config.audio_context_dim)
      @audio_after_text = alloc(prefix, "audio.after_audio_attn2", @config.audio_tokens * @config.audio_dim)
      @audio_v2a_query_norm = alloc(prefix, "audio.v2a_query_norm", @config.audio_tokens * @config.audio_dim)
      @audio_v2a_query_modulated = alloc(prefix, "audio.v2a_query_modulated", @config.audio_tokens * @config.audio_dim)
      @audio_v2a_context_norm = alloc(prefix, "audio.v2a_context_norm", @config.video_tokens * @config.video_dim)
      @audio_v2a_context_modulated = alloc(prefix, "audio.v2a_context_modulated", @config.video_tokens * @config.video_dim)
      @audio_after_av = alloc(prefix, "audio.after_av", @config.audio_tokens * @config.audio_dim)
      @audio_ff_norm = alloc(prefix, "audio.ff_norm", @config.audio_tokens * @config.audio_dim)
      @audio_ff_modulated = alloc(prefix, "audio.ff_modulated", @config.audio_tokens * @config.audio_dim)
      @audio_ff_hidden = alloc(prefix, "audio.ff_hidden", @config.audio_tokens * @config.audio_dim * 4)
      @audio_ff_activated = alloc(prefix, "audio.ff_activated", @config.audio_tokens * @config.audio_dim * 4)
      @audio_ff_module_output = alloc(prefix, "audio.ff_output", @config.audio_tokens * @config.audio_dim)
      @audio_output = alloc(prefix, "audio.output", @config.audio_tokens * @config.audio_dim)

      @checkpoints = {
        "video.attn1.output"               => @video_sa.scratch.module_output,
        "video.after_attn1"                => @video_after_sa,
        "video.attn2.output"               => @video_text.scratch.module_output,
        "video.after_attn2"                => @video_after_text,
        "video.audio_to_video_attn.output" => @video_a2v.scratch.module_output,
        "video.after_av_cross_attention"   => @video_after_av,
        "video.ff.output"                  => @video_ff_module_output,
        "video.ff.input"                   => @video_ff_modulated,
        "video.ff.hidden"                  => @video_ff_hidden,
        "video.ff.activated"               => @video_ff_activated,
        "video.output"                     => @video_output,
        "audio.audio_attn1.output"         => @audio_sa.scratch.module_output,
        "audio.after_audio_attn1"          => @audio_after_sa,
        "audio.audio_attn2.output"         => @audio_text.scratch.module_output,
        "audio.after_audio_attn2"          => @audio_after_text,
        "audio.video_to_audio_attn.output" => @audio_v2a.scratch.module_output,
        "audio.after_av_cross_attention"   => @audio_after_av,
        "audio.audio_ff.output"            => @audio_ff_module_output,
        "audio.audio_ff.input"             => @audio_ff_modulated,
        "audio.audio_ff.hidden"            => @audio_ff_hidden,
        "audio.audio_ff.activated"         => @audio_ff_activated,
        "audio.output"                     => @audio_output,
      }
      validate_buffers
    end

    # Caller owns encode/commit/wait. This path only binds prebuilt pipelines
    # and dispatches; all Metal buffers were allocated in initialize.
    def encode(e : ML::Metal::ComputeEncoder) : Tuple(ML::MetalBuffer, ML::MetalBuffer)
      c = @config
      encode_stream_self(e, @video_x, @video_sa_shift, @video_sa_scale, @video_sa_gate,
        @video_sa, @video_self_cos, @video_self_sin, @video_sa_norm, @video_sa_modulated,
        @video_after_sa, c.video_tokens, c.video_dim, c.video_modulation_rows)
      ML::Video::MetalOps.norm(e, @video_after_sa, @zero, @zero, @video_text_norm,
        c.video_tokens, c.video_dim, c.norm_eps, rms: true)
      video_text_query = @video_text_norm
      video_text_context = @video_context
      if c.cross_attention_adaln
        video_text_query = @video_text_query_modulated
        video_text_context = @video_text_context_modulated
        ML::Video::MetalOps.modulate(e, @video_text_norm, @video_text_shift.not_nil!,
          @video_text_scale.not_nil!, @video_text_query_modulated, c.video_tokens,
          c.video_dim, c.video_modulation_rows)
        ML::Video::MetalOps.modulate(e, @video_context, @video_prompt_shift.not_nil!,
          @video_prompt_scale.not_nil!, @video_text_context_modulated,
          c.video_context_tokens, c.video_context_dim, c.video_prompt_modulation_rows)
      end
      encode_attention(e, @video_text, video_text_query, video_text_context)
      ML::Video::MetalOps.add(e, @video_after_sa, @video_text.scratch.module_output,
        @video_after_text, c.video_tokens * c.video_dim,
        c.cross_attention_adaln ? @video_text_gate : nil,
        c.video_dim, c.video_modulation_rows)

      encode_stream_self(e, @audio_x, @audio_sa_shift, @audio_sa_scale, @audio_sa_gate,
        @audio_sa, @audio_self_cos, @audio_self_sin, @audio_sa_norm, @audio_sa_modulated,
        @audio_after_sa, c.audio_tokens, c.audio_dim, c.audio_modulation_rows)
      ML::Video::MetalOps.norm(e, @audio_after_sa, @zero, @zero, @audio_text_norm,
        c.audio_tokens, c.audio_dim, c.norm_eps, rms: true)
      audio_text_query = @audio_text_norm
      audio_text_context = @audio_context
      if c.cross_attention_adaln
        audio_text_query = @audio_text_query_modulated
        audio_text_context = @audio_text_context_modulated
        ML::Video::MetalOps.modulate(e, @audio_text_norm, @audio_text_shift.not_nil!,
          @audio_text_scale.not_nil!, @audio_text_query_modulated, c.audio_tokens,
          c.audio_dim, c.audio_modulation_rows)
        ML::Video::MetalOps.modulate(e, @audio_context, @audio_prompt_shift.not_nil!,
          @audio_prompt_scale.not_nil!, @audio_text_context_modulated,
          c.audio_context_tokens, c.audio_context_dim, c.audio_prompt_modulation_rows)
      end
      encode_attention(e, @audio_text, audio_text_query, audio_text_context)
      ML::Video::MetalOps.add(e, @audio_after_sa, @audio_text.scratch.module_output,
        @audio_after_text, c.audio_tokens * c.audio_dim,
        c.cross_attention_adaln ? @audio_text_gate : nil,
        c.audio_dim, c.audio_modulation_rows)

      # Both AV directions consume the same post-text state, as upstream does.
      ML::Video::MetalOps.norm(e, @video_after_text, @zero, @zero, @video_a2v_query_norm,
        c.video_tokens, c.video_dim, c.norm_eps, rms: true)
      ML::Video::MetalOps.modulate(e, @video_a2v_query_norm, @a2v_video_shift, @a2v_video_scale,
        @video_a2v_query_modulated, c.video_tokens, c.video_dim, c.video_modulation_rows)
      ML::Video::MetalOps.norm(e, @audio_after_text, @zero, @zero, @video_a2v_context_norm,
        c.audio_tokens, c.audio_dim, c.norm_eps, rms: true)
      ML::Video::MetalOps.modulate(e, @video_a2v_context_norm, @a2v_audio_shift, @a2v_audio_scale,
        @video_a2v_context_modulated, c.audio_tokens, c.audio_dim, c.audio_modulation_rows)
      encode_attention(e, @video_a2v, @video_a2v_query_modulated, @video_a2v_context_modulated,
        @video_cross_cos, @video_cross_sin, @audio_cross_cos, @audio_cross_sin)
      ML::Video::MetalOps.add(e, @video_after_text, @video_a2v.scratch.module_output,
        @video_after_av, c.video_tokens * c.video_dim, @a2v_video_gate,
        c.video_dim, c.video_modulation_rows)

      ML::Video::MetalOps.norm(e, @audio_after_text, @zero, @zero, @audio_v2a_query_norm,
        c.audio_tokens, c.audio_dim, c.norm_eps, rms: true)
      ML::Video::MetalOps.modulate(e, @audio_v2a_query_norm, @v2a_audio_shift, @v2a_audio_scale,
        @audio_v2a_query_modulated, c.audio_tokens, c.audio_dim, c.audio_modulation_rows)
      ML::Video::MetalOps.norm(e, @video_after_text, @zero, @zero, @audio_v2a_context_norm,
        c.video_tokens, c.video_dim, c.norm_eps, rms: true)
      ML::Video::MetalOps.modulate(e, @audio_v2a_context_norm, @v2a_video_shift, @v2a_video_scale,
        @audio_v2a_context_modulated, c.video_tokens, c.video_dim, c.video_modulation_rows)
      encode_attention(e, @audio_v2a, @audio_v2a_query_modulated, @audio_v2a_context_modulated,
        @audio_cross_cos, @audio_cross_sin, @video_cross_cos, @video_cross_sin)
      ML::Video::MetalOps.add(e, @audio_after_text, @audio_v2a.scratch.module_output,
        @audio_after_av, c.audio_tokens * c.audio_dim, @v2a_audio_gate,
        c.audio_dim, c.audio_modulation_rows)

      encode_stream_ff(e, @video_after_av, @video_ff_shift, @video_ff_scale, @video_ff_gate,
        @video_ff_norm, @video_ff_modulated, @video_ff_hidden, @video_ff_activated,
        @video_ff_module_output, @video_output, @video_ff_weights,
        c.video_tokens, c.video_dim, c.video_modulation_rows)
      encode_stream_ff(e, @audio_after_av, @audio_ff_shift, @audio_ff_scale, @audio_ff_gate,
        @audio_ff_norm, @audio_ff_modulated, @audio_ff_hidden, @audio_ff_activated,
        @audio_ff_module_output, @audio_output, @audio_ff_weights,
        c.audio_tokens, c.audio_dim, c.audio_modulation_rows)
      {@video_output, @audio_output}
    end

    private def validate_config : Nil
      c = @config
      values = [c.video_dim, c.video_context_dim, c.video_context_tokens, c.video_tokens,
                c.video_heads, c.video_head_dim, c.audio_dim, c.audio_context_dim,
                c.audio_context_tokens, c.audio_tokens, c.audio_heads, c.audio_head_dim,
                c.video_modulation_rows, c.audio_modulation_rows,
                c.video_prompt_modulation_rows, c.audio_prompt_modulation_rows]
      raise ArgumentError.new("LTX dimensions must be positive") unless values.all? { |v| v > 0 }
      unless c.video_dim.to_i64 == c.video_heads.to_i64 * c.video_head_dim &&
             c.audio_dim.to_i64 == c.audio_heads.to_i64 * c.audio_head_dim
        raise ArgumentError.new("LTX modality dimensions must equal heads times head dimension")
      end
      unless c.video_modulation_rows == 1 || c.video_modulation_rows == c.video_tokens
        raise ArgumentError.new("video modulation rows must be one or video_tokens")
      end
      unless c.audio_modulation_rows == 1 || c.audio_modulation_rows == c.audio_tokens
        raise ArgumentError.new("audio modulation rows must be one or audio_tokens")
      end
      unless c.video_prompt_modulation_rows == 1 || c.video_prompt_modulation_rows == c.video_context_tokens
        raise ArgumentError.new("video prompt modulation rows must be one or video_context_tokens")
      end
      unless c.audio_prompt_modulation_rows == 1 || c.audio_prompt_modulation_rows == c.audio_context_tokens
        raise ArgumentError.new("audio prompt modulation rows must be one or audio_context_tokens")
      end
      if c.video_head_dim > 256 || c.audio_head_dim > 256
        raise ArgumentError.new("LTX Metal attention head dimensions cannot exceed 256")
      end
      raise ArgumentError.new("LTX split RoPE needs even head dimensions") if c.video_head_dim.odd? || c.audio_head_dim.odd?
      raise ArgumentError.new("Invalid LTX norm epsilon") unless c.norm_eps.finite? && c.norm_eps > 0
    end

    private def input(name : String) : ML::MetalBuffer
      buffer = @arena.buffers[name]?
      raise ArgumentError.new("Missing LTX input #{name}") unless buffer && buffer.valid?
      buffer
    end

    private def alloc(prefix : String, name : String, count : Int32) : ML::MetalBuffer
      @arena.alloc("#{prefix}.#{name}", count)
    end

    private def required_weight(name : String) : ML::MetalBuffer
      buffer = @arena.buffers["w.#{name}"]?
      raise ArgumentError.new("Missing LTX parameter #{name}") unless buffer && buffer.valid?
      buffer
    end

    private def optional_bias(name : String) : ML::MetalBuffer
      @arena.buffers["w.#{name}"]? || @zero
    end

    private def build_attention(prefix : String, scratch_name : String, weight_name : String,
                                query_rows : Int32, key_rows : Int32, query_dim : Int32,
                                context_dim : Int32, heads : Int32, head_dim : Int32,
                                gated : Bool) : AttentionLayer
      inner = heads * head_dim
      gate = gated ? required_weight("#{weight_name}.to_gate_logits.weight") : nil
      gate_bias = gated ? required_weight("#{weight_name}.to_gate_logits.bias") : nil
      weights = AttentionWeights.new(
        required_weight("#{weight_name}.to_q.weight"),
        required_weight("#{weight_name}.to_k.weight"),
        required_weight("#{weight_name}.to_v.weight"),
        required_weight("#{weight_name}.q_norm.weight"),
        required_weight("#{weight_name}.k_norm.weight"),
        required_weight("#{weight_name}.to_out.0.weight"),
        required_weight("#{weight_name}.to_q.bias"),
        required_weight("#{weight_name}.to_k.bias"),
        required_weight("#{weight_name}.to_v.bias"),
        required_weight("#{weight_name}.to_out.0.bias"),
        gate,
        gate_bias,
      )
      scratch = AttentionScratch.new(
        alloc(prefix, "#{scratch_name}.query", query_rows * inner),
        alloc(prefix, "#{scratch_name}.key", key_rows * inner),
        alloc(prefix, "#{scratch_name}.value", key_rows * inner),
        alloc(prefix, "#{scratch_name}.query_norm", query_rows * inner),
        alloc(prefix, "#{scratch_name}.key_norm", key_rows * inner),
        alloc(prefix, "#{scratch_name}.query_rope", query_rows * inner),
        alloc(prefix, "#{scratch_name}.key_rope", key_rows * inner),
        alloc(prefix, "#{scratch_name}.attended", query_rows * inner),
        alloc(prefix, "#{scratch_name}.module_output", query_rows * query_dim),
        gated ? alloc(prefix, "#{scratch_name}.gate_logits", query_rows * heads) : nil,
        gated ? alloc(prefix, "#{scratch_name}.gated_attended", query_rows * inner) : nil,
      )
      AttentionLayer.new(weights, scratch, query_rows, key_rows, query_dim,
        context_dim, heads, head_dim)
    end

    private def build_ff_weights(name : String, dim : Int32, bias : Bool) : FeedForwardWeights
      FeedForwardWeights.new(
        required_weight("#{name}.net.0.proj.weight"),
        required_weight("#{name}.net.2.weight"),
        bias ? required_weight("#{name}.net.0.proj.bias") : @zero,
        bias ? required_weight("#{name}.net.2.bias") : @zero,
      )
    end

    private def validate_buffers : Nil
      c = @config
      ML::Video::MetalOps.check(@video_x, c.video_tokens.to_i64 * c.video_dim)
      ML::Video::MetalOps.check(@audio_x, c.audio_tokens.to_i64 * c.audio_dim)
      ML::Video::MetalOps.check(@video_context, c.video_context_tokens.to_i64 * c.video_context_dim)
      ML::Video::MetalOps.check(@audio_context, c.audio_context_tokens.to_i64 * c.audio_context_dim)
      {
        {@video_sa_shift, c.video_modulation_rows * c.video_dim},
        {@video_sa_scale, c.video_modulation_rows * c.video_dim},
        {@video_sa_gate, c.video_modulation_rows * c.video_dim},
        {@video_ff_shift, c.video_modulation_rows * c.video_dim},
        {@video_ff_scale, c.video_modulation_rows * c.video_dim},
        {@video_ff_gate, c.video_modulation_rows * c.video_dim},
        {@audio_sa_shift, c.audio_modulation_rows * c.audio_dim},
        {@audio_sa_scale, c.audio_modulation_rows * c.audio_dim},
        {@audio_sa_gate, c.audio_modulation_rows * c.audio_dim},
        {@audio_ff_shift, c.audio_modulation_rows * c.audio_dim},
        {@audio_ff_scale, c.audio_modulation_rows * c.audio_dim},
        {@audio_ff_gate, c.audio_modulation_rows * c.audio_dim},
        {@a2v_video_shift, c.video_modulation_rows * c.video_dim},
        {@a2v_video_scale, c.video_modulation_rows * c.video_dim},
        {@a2v_video_gate, c.video_modulation_rows * c.video_dim},
        {@a2v_audio_shift, c.audio_modulation_rows * c.audio_dim},
        {@a2v_audio_scale, c.audio_modulation_rows * c.audio_dim},
        {@v2a_audio_shift, c.audio_modulation_rows * c.audio_dim},
        {@v2a_audio_scale, c.audio_modulation_rows * c.audio_dim},
        {@v2a_audio_gate, c.audio_modulation_rows * c.audio_dim},
        {@v2a_video_shift, c.video_modulation_rows * c.video_dim},
        {@v2a_video_scale, c.video_modulation_rows * c.video_dim},
      }.each { |buffer, count| ML::Video::MetalOps.check(buffer, count.to_i64) }

      if c.cross_attention_adaln
        {
          {@video_text_shift.not_nil!, c.video_modulation_rows.to_i64 * c.video_dim},
          {@video_text_scale.not_nil!, c.video_modulation_rows.to_i64 * c.video_dim},
          {@video_text_gate.not_nil!, c.video_modulation_rows.to_i64 * c.video_dim},
          {@video_prompt_shift.not_nil!, c.video_prompt_modulation_rows.to_i64 * c.video_context_dim},
          {@video_prompt_scale.not_nil!, c.video_prompt_modulation_rows.to_i64 * c.video_context_dim},
          {@audio_text_shift.not_nil!, c.audio_modulation_rows.to_i64 * c.audio_dim},
          {@audio_text_scale.not_nil!, c.audio_modulation_rows.to_i64 * c.audio_dim},
          {@audio_text_gate.not_nil!, c.audio_modulation_rows.to_i64 * c.audio_dim},
          {@audio_prompt_shift.not_nil!, c.audio_prompt_modulation_rows.to_i64 * c.audio_context_dim},
          {@audio_prompt_scale.not_nil!, c.audio_prompt_modulation_rows.to_i64 * c.audio_context_dim},
        }.each { |buffer, count| ML::Video::MetalOps.check(buffer, count) }
      end

      video_self_rope = c.video_tokens.to_i64 * c.video_heads * (c.video_head_dim // 2)
      audio_self_rope = c.audio_tokens.to_i64 * c.audio_heads * (c.audio_head_dim // 2)
      # Both AV cross attentions use the audio module's head count and head size.
      video_cross_rope = c.video_tokens.to_i64 * c.audio_heads * (c.audio_head_dim // 2)
      audio_cross_rope = c.audio_tokens.to_i64 * c.audio_heads * (c.audio_head_dim // 2)
      {
        {@video_self_cos, video_self_rope}, {@video_self_sin, video_self_rope},
        {@audio_self_cos, audio_self_rope}, {@audio_self_sin, audio_self_rope},
        {@video_cross_cos, video_cross_rope}, {@video_cross_sin, video_cross_rope},
        {@audio_cross_cos, audio_cross_rope}, {@audio_cross_sin, audio_cross_rope},
      }.each { |buffer, count| ML::Video::MetalOps.check(buffer, count) }

      validate_attention_weights(@video_sa, c.video_dim, c.video_dim)
      validate_attention_weights(@video_text, c.video_dim, c.video_context_dim)
      validate_attention_weights(@video_a2v, c.video_dim, c.audio_dim)
      validate_attention_weights(@audio_sa, c.audio_dim, c.audio_dim)
      validate_attention_weights(@audio_text, c.audio_dim, c.audio_context_dim)
      validate_attention_weights(@audio_v2a, c.audio_dim, c.video_dim)
      validate_ff_weights(@video_ff_weights, c.video_dim)
      validate_ff_weights(@audio_ff_weights, c.audio_dim)
    end

    private def validate_attention_weights(layer : AttentionLayer, query_dim : Int32,
                                           context_dim : Int32) : Nil
      inner = layer.inner_dim.to_i64
      {
        {layer.weights.query, inner * query_dim}, {layer.weights.key, inner * context_dim},
        {layer.weights.value, inner * context_dim}, {layer.weights.projection, query_dim.to_i64 * inner},
        {layer.weights.query_norm, inner}, {layer.weights.key_norm, inner},
        {layer.weights.query_bias, inner}, {layer.weights.key_bias, inner},
        {layer.weights.value_bias, inner}, {layer.weights.projection_bias, query_dim.to_i64},
      }.each { |buffer, count| ML::Video::MetalOps.check(buffer, count) }
      if layer.weights.gate
        gate = layer.weights.gate || raise ArgumentError.new("Missing LTX gated-attention weights")
        gate_bias = layer.weights.gate_bias || raise ArgumentError.new("Missing LTX gated-attention bias")
        ML::Video::MetalOps.check(gate, layer.heads.to_i64 * query_dim)
        ML::Video::MetalOps.check(gate_bias, layer.heads.to_i64)
        ML::Video::MetalOps.check(layer.scratch.gate_logits.not_nil!, layer.query_rows.to_i64 * layer.heads)
        ML::Video::MetalOps.check(layer.scratch.gated_attended.not_nil!, layer.query_rows.to_i64 * inner)
      end
    end

    private def validate_ff_weights(weights : FeedForwardWeights, dim : Int32) : Nil
      d = dim.to_i64
      ML::Video::MetalOps.check(weights.input, 4_i64 * d * d)
      ML::Video::MetalOps.check(weights.output, 4_i64 * d * d)
      ML::Video::MetalOps.check(weights.input_bias, 4_i64 * d)
      ML::Video::MetalOps.check(weights.output_bias, d)
    end

    private def encode_split_rope(e : ML::Metal::ComputeEncoder, x : ML::MetalBuffer,
                                  cosine : ML::MetalBuffer, sine : ML::MetalBuffer,
                                  y : ML::MetalBuffer, rows : Int32,
                                  heads : Int32, head_dim : Int32) : Nil
      count = rows.to_i64 * heads * head_dim
      ML::Video::MetalOps.check(x, count)
      ML::Video::MetalOps.check(cosine, count // 2)
      ML::Video::MetalOps.check(sine, count // 2)
      ML::Video::MetalOps.check(y, count)
      ML::Video::MetalOps.distinct(y, x, cosine, sine)
      e.set_pipeline(@rope_pipeline)
      e.set_buffer(x, 0).set_buffer(cosine, 1).set_buffer(sine, 2)
      e.set_buffer(y, 3, ML::Metal::BufferAccess::Write)
      e.set_value(rows.to_u32, 4).set_value(heads.to_u32, 5).set_value(head_dim.to_u32, 6)
      e.dispatch_1d(count.to_i32)
    end

    private def encode_attention(e : ML::Metal::ComputeEncoder, layer : AttentionLayer,
                                 query : ML::MetalBuffer, context : ML::MetalBuffer,
                                 query_cos : ML::MetalBuffer? = nil, query_sin : ML::MetalBuffer? = nil,
                                 key_cos : ML::MetalBuffer? = nil, key_sin : ML::MetalBuffer? = nil) : Nil
      s = layer.scratch
      w = layer.weights
      ML::Video::MetalOps.linear(e, query, w.query, w.query_bias, s.query,
        layer.query_rows, layer.query_dim, layer.inner_dim)
      ML::Video::MetalOps.linear(e, context, w.key, w.key_bias, s.key,
        layer.key_rows, layer.context_dim, layer.inner_dim)
      ML::Video::MetalOps.linear(e, context, w.value, w.value_bias, s.value,
        layer.key_rows, layer.context_dim, layer.inner_dim)
      ML::Video::MetalOps.norm(e, s.query, w.query_norm, @zero, s.query_norm,
        layer.query_rows, layer.inner_dim, @config.norm_eps, rms: true, affine: true)
      ML::Video::MetalOps.norm(e, s.key, w.key_norm, @zero, s.key_norm,
        layer.key_rows, layer.inner_dim, @config.norm_eps, rms: true, affine: true)

      query_rope = s.query_norm
      key_rope = s.key_norm
      if query_cos && query_sin && key_cos && key_sin
        query_rope = s.query_rope
        key_rope = s.key_rope
        encode_split_rope(e, s.query_norm, query_cos, query_sin, s.query_rope,
          layer.query_rows, layer.heads, layer.head_dim)
        encode_split_rope(e, s.key_norm, key_cos, key_sin, s.key_rope,
          layer.key_rows, layer.heads, layer.head_dim)
      end
      ML::Video::MetalOps.attention(e, query_rope, key_rope, s.value, s.attended,
        layer.query_rows, layer.key_rows, layer.heads, layer.head_dim)
      attention_output = s.attended
      if gate_weight = w.gate
        gate_bias = w.gate_bias || raise ArgumentError.new("Missing LTX gated-attention bias")
        gate_logits = s.gate_logits || raise ArgumentError.new("Missing LTX gated-attention scratch")
        gated_attended = s.gated_attended || raise ArgumentError.new("Missing LTX gated-attention scratch")
        ML::Video::MetalOps.linear(e, query, gate_weight, gate_bias, gate_logits,
          layer.query_rows, layer.query_dim, layer.heads)
        encode_apply_head_gates(e, s.attended, gate_logits, gated_attended,
          layer.query_rows, layer.heads, layer.head_dim)
        attention_output = gated_attended
      end
      ML::Video::MetalOps.linear(e, attention_output, w.projection, w.projection_bias, s.module_output,
        layer.query_rows, layer.inner_dim, layer.query_dim)
    end

    private def encode_apply_head_gates(e : ML::Metal::ComputeEncoder,
                                        attended : ML::MetalBuffer,
                                        gate_logits : ML::MetalBuffer,
                                        output : ML::MetalBuffer,
                                        rows : Int32, heads : Int32,
                                        head_dim : Int32) : Nil
      count = rows.to_i64 * heads * head_dim
      ML::Video::MetalOps.check(attended, count)
      ML::Video::MetalOps.check(gate_logits, rows.to_i64 * heads)
      ML::Video::MetalOps.check(output, count)
      ML::Video::MetalOps.distinct(output, attended, gate_logits)
      e.set_pipeline(@gate_pipeline)
      e.set_buffer(attended, 0).set_buffer(gate_logits, 1)
      e.set_buffer(output, 2, ML::Metal::BufferAccess::Write)
      e.set_value(head_dim.to_u32, 3).set_value(count.to_u32, 4)
      e.dispatch_1d(count.to_i32)
    end

    private def encode_stream_self(e : ML::Metal::ComputeEncoder,
                                   x : ML::MetalBuffer, shift : ML::MetalBuffer,
                                   scale : ML::MetalBuffer, gate : ML::MetalBuffer,
                                   layer : AttentionLayer, cosine : ML::MetalBuffer,
                                   sine : ML::MetalBuffer, norm : ML::MetalBuffer,
                                   modulated : ML::MetalBuffer, after : ML::MetalBuffer,
                                   rows : Int32, dim : Int32, mod_rows : Int32) : Nil
      ML::Video::MetalOps.norm(e, x, @zero, @zero, norm, rows, dim, @config.norm_eps, rms: true)
      ML::Video::MetalOps.modulate(e, norm, shift, scale, modulated, rows, dim, mod_rows)
      # Upstream passes norm_vx/norm_ax as x and omits context, so Q/K/V all
      # consume the same AdaLN-modulated self-attention input.
      encode_attention(e, layer, modulated, modulated, cosine, sine, cosine, sine)
      ML::Video::MetalOps.add(e, x, layer.scratch.module_output, after,
        rows * dim, gate, dim, mod_rows)
    end

    private def encode_stream_ff(e : ML::Metal::ComputeEncoder,
                                 x : ML::MetalBuffer, shift : ML::MetalBuffer,
                                 scale : ML::MetalBuffer, gate : ML::MetalBuffer,
                                 norm : ML::MetalBuffer, modulated : ML::MetalBuffer,
                                 hidden : ML::MetalBuffer, activated : ML::MetalBuffer,
                                 module_output : ML::MetalBuffer, output : ML::MetalBuffer,
                                 weights : FeedForwardWeights, rows : Int32, dim : Int32,
                                 mod_rows : Int32) : Nil
      ML::Video::MetalOps.norm(e, x, @zero, @zero, norm, rows, dim, @config.norm_eps, rms: true)
      ML::Video::MetalOps.modulate(e, norm, shift, scale, modulated, rows, dim, mod_rows)
      ML::Video::MetalOps.linear(e, modulated, weights.input, weights.input_bias,
        hidden, rows, dim, dim * 4)
      ML::Video::MetalOps.gelu(e, hidden, activated, rows * dim * 4, approximate: true)
      ML::Video::MetalOps.linear(e, activated, weights.output, weights.output_bias,
        module_output, rows, dim * 4, dim)
      ML::Video::MetalOps.add(e, x, module_output, output, rows * dim, gate, dim, mod_rows)
    end
  end
end
