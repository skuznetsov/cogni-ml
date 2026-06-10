require "./diffusion_gemma_weights"

module ML::GGUF
  struct DiffusionGemmaDenoiseParams
    getter max_steps : Int32
    getter t_min : Float32
    getter t_max : Float32
    getter entropy_bound : Float32
    getter stability_threshold : Int32
    getter confidence_threshold : Float32
    getter kv_cache : Bool
    getter seed : Int32

    def initialize(@max_steps : Int32,
                   @t_min : Float32,
                   @t_max : Float32,
                   @entropy_bound : Float32,
                   @stability_threshold : Int32,
                   @confidence_threshold : Float32,
                   @kv_cache : Bool = true,
                   @seed : Int32 = 123)
      raise ArgumentError.new("DiffusionGemma max_steps must be positive") unless @max_steps > 0
      raise ArgumentError.new("DiffusionGemma t_min must be non-negative") unless @t_min >= 0.0_f32
      raise ArgumentError.new("DiffusionGemma t_max must be >= t_min") unless @t_max >= @t_min
      raise ArgumentError.new("DiffusionGemma entropy_bound must be non-negative") unless @entropy_bound >= 0.0_f32
      raise ArgumentError.new("DiffusionGemma stability_threshold must be positive") unless @stability_threshold > 0
      raise ArgumentError.new("DiffusionGemma confidence_threshold must be non-negative") unless @confidence_threshold >= 0.0_f32
    end

    def self.from_hparams(hp : DiffusionGemmaHparams,
                          max_steps : Int32? = nil,
                          t_min : Float32? = nil,
                          t_max : Float32? = nil,
                          entropy_bound : Float32? = nil,
                          stability_threshold : Int32? = nil,
                          confidence_threshold : Float32? = nil,
                          kv_cache : Bool = true,
                          seed : Int32 = 123) : DiffusionGemmaDenoiseParams
      DiffusionGemmaDenoiseParams.new(
        max_steps: max_steps || hp.eb_max_steps,
        t_min: t_min || hp.eb_t_min,
        t_max: t_max || hp.eb_t_max,
        entropy_bound: entropy_bound || hp.eb_entropy_bound,
        stability_threshold: stability_threshold || hp.eb_stability_threshold,
        confidence_threshold: confidence_threshold || hp.eb_confidence_threshold,
        kv_cache: kv_cache,
        seed: seed,
      )
    end
  end

  struct DiffusionGemmaRequest
    getter prompt_tokens : Array(Int32)
    getter canvas_tokens : Array(Int32)

    def initialize(@prompt_tokens : Array(Int32), @canvas_tokens : Array(Int32), hp : DiffusionGemmaHparams)
      raise ArgumentError.new("DiffusionGemma prompt must not be empty") if @prompt_tokens.empty?
      raise ArgumentError.new("DiffusionGemma canvas size #{@canvas_tokens.size} != #{hp.canvas_length}") unless @canvas_tokens.size == hp.canvas_length
      raise ArgumentError.new("DiffusionGemma prompt+canvas exceeds context_length") if total_tokens > hp.context_length
    end

    def self.with_blank_canvas(prompt_tokens : Array(Int32),
                               hp : DiffusionGemmaHparams,
                               mask_token_id : Int32) : DiffusionGemmaRequest
      raise ArgumentError.new("DiffusionGemma mask_token_id must be non-negative") if mask_token_id < 0
      DiffusionGemmaRequest.new(prompt_tokens, Array.new(hp.canvas_length, mask_token_id), hp)
    end

    def prompt_len : Int32
      @prompt_tokens.size
    end

    def canvas_len : Int32
      @canvas_tokens.size
    end

    def total_tokens : Int32
      prompt_len + canvas_len
    end

    def canvas_start : Int32
      prompt_len
    end

    def packed_tokens : Array(Int32)
      @prompt_tokens + @canvas_tokens
    end

    def canvas_position?(pos : Int32) : Bool
      pos >= canvas_start && pos < total_tokens
    end
  end

  struct DiffusionGemmaAttentionMask
    getter prompt_len : Int32
    getter canvas_len : Int32
    getter sliding_window : Int32

    def initialize(@prompt_len : Int32, @canvas_len : Int32, @sliding_window : Int32)
      raise ArgumentError.new("prompt_len must be non-negative") if @prompt_len < 0
      raise ArgumentError.new("canvas_len must be positive") unless @canvas_len > 0
      raise ArgumentError.new("sliding_window must be positive") unless @sliding_window > 0
    end

    def total_tokens : Int32
      @prompt_len + @canvas_len
    end

    def allow_unified?(query_pos : Int32, key_pos : Int32, sliding : Bool) : Bool
      check_square_bounds(query_pos, key_pos)
      query_canvas = query_pos >= @prompt_len
      key_canvas = key_pos >= @prompt_len
      if query_canvas
        return true unless sliding

        key_canvas || key_pos >= canvas_prompt_low
      else
        !key_canvas && key_pos <= query_pos && (!sliding || key_pos >= prompt_query_low(query_pos))
      end
    end

    def allow_decode?(canvas_query_index : Int32, key_pos : Int32, sliding : Bool) : Bool
      raise ArgumentError.new("canvas query out of bounds") if canvas_query_index < 0 || canvas_query_index >= @canvas_len
      raise ArgumentError.new("key out of bounds") if key_pos < 0 || key_pos >= total_tokens
      return true if key_pos >= @prompt_len
      return true unless sliding

      key_pos >= canvas_prompt_low
    end

    def canvas_prompt_low : Int32
      Math.max(0, @prompt_len - @sliding_window + 1)
    end

    private def prompt_query_low(query_pos : Int32) : Int32
      Math.max(0, query_pos - @sliding_window + 1)
    end

    private def check_square_bounds(query_pos : Int32, key_pos : Int32) : Nil
      raise ArgumentError.new("query out of bounds") if query_pos < 0 || query_pos >= total_tokens
      raise ArgumentError.new("key out of bounds") if key_pos < 0 || key_pos >= total_tokens
    end
  end

  struct DiffusionGemmaRuntimePlan
    getter prompt_len : Int32
    getter canvas_len : Int32
    getter vocab_size : Int32
    getter n_embd : Int32
    getter max_length : Int32
    getter self_conditioning_logits_bytes : Int64

    def initialize(hp : DiffusionGemmaHparams, request : DiffusionGemmaRequest)
      @prompt_len = request.prompt_len
      @canvas_len = request.canvas_len
      @vocab_size = hp.vocab_size
      @n_embd = hp.n_embd
      @max_length = request.total_tokens
      @self_conditioning_logits_bytes = @canvas_len.to_i64 * @vocab_size.to_i64 * 4_i64
    end

    def mask(hp : DiffusionGemmaHparams) : DiffusionGemmaAttentionMask
      DiffusionGemmaAttentionMask.new(@prompt_len, @canvas_len, hp.sliding_window)
    end

    def kv_dim_for_layer(hp : DiffusionGemmaHparams, il : Int32) : Int32
      hp.n_head_kv(il) * hp.head_dim_for_layer(il)
    end

    def layer_scale_for_pos(weights : DiffusionGemmaWeights, il : Int32, pos : Int32) : Float32
      layer = weights.layers[il]
      if pos < @prompt_len
        layer.encoder_layer_output_scale[0]
      else
        layer.layer_output_scale[0]
      end
    end
  end
end
