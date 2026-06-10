require "./reader"

# DiffusionGemma GGUF hyperparameter parser.
#
# DiffusionGemma uses a Gemma4-like backbone, but the GGUF architecture and
# metadata prefix are `diffusion-gemma`. The prompt/canvas split is controlled
# by `diffusion.canvas_length`, and the decoder is non-causal over the canvas.
module ML::GGUF
  struct DiffusionGemmaHparams
    getter arch : String
    getter n_layer : Int32
    getter n_embd : Int32
    getter n_ff : Int32
    getter expert_ff : Int32
    getter expert_count : Int32
    getter expert_used_count : Int32
    getter context_length : Int32
    getter canvas_length : Int32
    getter vocab_size : Int32
    getter eb_max_steps : Int32
    getter eb_t_min : Float32
    getter eb_t_max : Float32
    getter eb_entropy_bound : Float32
    getter eb_stability_threshold : Int32
    getter eb_confidence_threshold : Float32

    getter n_head : Int32
    getter n_head_kv_by_layer : Array(Int32)
    getter head_dim : Int32
    getter head_dim_swa : Int32
    getter sliding_window : Int32
    getter sliding_window_pattern : Array(Bool)
    getter shared_kv_layers : Int32

    getter rms_eps : Float32
    getter rope_freq_base : Float32
    getter rope_freq_base_swa : Float32
    getter rope_dim_count : Int32
    getter rope_dim_count_swa : Int32
    getter final_logit_softcapping : Float32
    getter embedding_length_per_layer_input : Int32
    getter causal_attention : Bool

    def initialize(g : GGUFFile)
      @arch = g.get_string("general.architecture") || "diffusion-gemma"
      raise "Not a diffusion-gemma model: arch=#{@arch.inspect}" unless @arch == "diffusion-gemma"

      prefix = @arch
      @n_layer = req_int(g, "#{prefix}.block_count")
      @n_embd = req_int(g, "#{prefix}.embedding_length")
      @n_ff = req_int(g, "#{prefix}.feed_forward_length")
      @expert_ff = req_int(g, "#{prefix}.expert_feed_forward_length")
      @expert_count = req_int(g, "#{prefix}.expert_count")
      @expert_used_count = req_int(g, "#{prefix}.expert_used_count")
      @context_length = req_int(g, "#{prefix}.context_length")
      @canvas_length = req_int(g, "diffusion.canvas_length")
      raise "diffusion-gemma: diffusion.canvas_length must be positive" unless @canvas_length > 0
      @eb_max_steps = (g.get_int("diffusion.eb_max_steps") || 48_i64).to_i32
      @eb_t_min = g.get_float("diffusion.eb_t_min").try(&.to_f32) || 0.4_f32
      @eb_t_max = g.get_float("diffusion.eb_t_max").try(&.to_f32) || 0.8_f32
      @eb_entropy_bound = g.get_float("diffusion.eb_entropy_bound").try(&.to_f32) || 0.1_f32
      @eb_stability_threshold = (g.get_int("diffusion.eb_stability_threshold") || 1_i64).to_i32
      @eb_confidence_threshold = g.get_float("diffusion.eb_confidence_threshold").try(&.to_f32) || 0.005_f32

      @n_head = req_int(g, "#{prefix}.attention.head_count")
      @n_head_kv_by_layer = req_int_or_array(g, "#{prefix}.attention.head_count_kv", @n_layer)

      @head_dim = req_int(g, "#{prefix}.attention.key_length")
      value_dim = req_int(g, "#{prefix}.attention.value_length")
      raise "diffusion-gemma: key_length (#{@head_dim}) != value_length (#{value_dim})" if @head_dim != value_dim

      @head_dim_swa = req_int(g, "#{prefix}.attention.key_length_swa")
      value_dim_swa = req_int(g, "#{prefix}.attention.value_length_swa")
      raise "diffusion-gemma: key_length_swa (#{@head_dim_swa}) != value_length_swa (#{value_dim_swa})" if @head_dim_swa != value_dim_swa

      @sliding_window = req_int(g, "#{prefix}.attention.sliding_window")
      @sliding_window_pattern = g.get_bool_array("#{prefix}.attention.sliding_window_pattern") ||
                                raise "diffusion-gemma: missing required bool array #{prefix}.attention.sliding_window_pattern"
      raise "diffusion-gemma: sliding_window_pattern size #{@sliding_window_pattern.size} != n_layer #{@n_layer}" if @sliding_window_pattern.size != @n_layer
      @shared_kv_layers = (g.get_int("#{prefix}.attention.shared_kv_layers") || 0_i64).to_i32

      @rms_eps = g.get_float("#{prefix}.attention.layer_norm_rms_epsilon").try(&.to_f32) || 1.0e-6_f32
      @rope_freq_base = g.get_float("#{prefix}.rope.freq_base").try(&.to_f32) || 1_000_000.0_f32
      @rope_freq_base_swa = g.get_float("#{prefix}.rope.freq_base_swa").try(&.to_f32) || 10_000.0_f32
      @rope_dim_count = req_int(g, "#{prefix}.rope.dimension_count")
      @rope_dim_count_swa = req_int(g, "#{prefix}.rope.dimension_count_swa")
      @final_logit_softcapping = g.get_float("#{prefix}.final_logit_softcapping").try(&.to_f32) || 0.0_f32
      @embedding_length_per_layer_input = (g.get_int("#{prefix}.embedding_length_per_layer_input") || 0_i64).to_i32
      @causal_attention = opt_bool(g, "#{prefix}.attention.causal") || false

      token_embd = g.tensor("token_embd.weight") || raise "diffusion-gemma: missing token_embd.weight"
      @vocab_size = token_embd.dims.size >= 2 ? token_embd.dims[1].to_i32 : 0
    end

    def sliding_window?(il : Int32) : Bool
      @sliding_window_pattern[il]
    end

    def full_attention?(il : Int32) : Bool
      !sliding_window?(il)
    end

    def n_head_kv(il : Int32) : Int32
      @n_head_kv_by_layer[il]
    end

    def has_kv?(il : Int32) : Bool
      il < @n_layer - @shared_kv_layers
    end

    def head_dim_for_layer(il : Int32) : Int32
      sliding_window?(il) ? @head_dim_swa : @head_dim
    end

    def rope_dim_for_layer(il : Int32) : Int32
      sliding_window?(il) ? @rope_dim_count_swa : @rope_dim_count
    end

    def rope_freq_base_for_layer(il : Int32) : Float32
      sliding_window?(il) ? @rope_freq_base_swa : @rope_freq_base
    end

    def full_attention_layers : Array(Int32)
      (0...@n_layer).select { |il| full_attention?(il) }
    end

    def sliding_window_layers : Array(Int32)
      (0...@n_layer).select { |il| sliding_window?(il) }
    end

    def attention_start_pos(il : Int32, pos : Int32) : Int32
      return 0 unless sliding_window?(il)

      Math.max(0, pos - @sliding_window + 1)
    end

    private def req_int(g : GGUFFile, key : String) : Int32
      v = g.get_int(key)
      raise "diffusion-gemma: missing required key #{key.inspect}" unless v
      v.to_i32
    end

    private def opt_bool(g : GGUFFile, key : String) : Bool?
      g.metadata[key]?.as?(Bool)
    end

    private def req_int_or_array(g : GGUFFile, key : String, expected_size : Int32) : Array(Int32)
      if arr = g.get_int_array(key)
        raise "diffusion-gemma: #{key} size #{arr.size} != n_layer #{expected_size}" if arr.size != expected_size
        return arr.map(&.to_i32)
      end
      if scalar = g.get_int(key)
        return Array.new(expected_size, scalar.to_i32)
      end
      raise "diffusion-gemma: missing required key #{key.inspect}"
    end
  end
end
