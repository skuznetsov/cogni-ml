require "./reader"

# Qwen 3.5 / 3.6 (arch=qwen35) hyperparameters parser.
#
# Parses both text (9B/27B) and VL variants. Verified against:
#   ~/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf
#   ~/.cache/lm-studio/models/lmstudio-community/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf
#
# Architecture notes (see LANDMARKS.md):
#   - Hybrid: full-attention at layers where ((il+1) % full_attention_interval == 0),
#     otherwise DeltaNet/GatedDeltaRule recurrent
#   - head_dim = key_length = value_length = 256
#   - M-RoPE partial on first rope.dimension_count=64 dims with rope.dimension_sections
#   - SwiGLU FFN (not MoE for 9B/27B; qwen35moe is a separate arch)
module ML::GGUF
  struct Qwen35Hparams
    # Model topology
    getter arch : String
    getter n_layer : Int32
    getter n_embd : Int32
    getter n_ff : Int32
    getter context_length : Int32
    getter vocab_size : Int32

    # Full-attention layers
    getter n_head : Int32
    getter n_head_kv : Int32
    getter head_dim : Int32   # = key_length = value_length
    getter rms_eps : Float32

    # Full-attention vs recurrent cadence: full-attn when ((il+1) % full_attention_interval == 0)
    getter full_attention_interval : Int32

    # M-RoPE
    getter rope_freq_base : Float32
    getter rope_dim_count : Int32        # partial RoPE dim (== 64 for Qwen 3.5/3.6)
    getter rope_sections : Array(Int32)  # e.g. [11, 11, 10, 0]

    # DeltaNet / GatedDeltaRule (for recurrent layers)
    getter ssm_conv_kernel : Int32
    getter ssm_state_size : Int32        # head_k_dim = head_v_dim
    getter ssm_group_count : Int32       # num_k_heads
    getter ssm_time_step_rank : Int32    # num_v_heads
    getter ssm_inner_size : Int32        # d_inner

    def initialize(g : GGUFFile)
      @arch = g.get_string("general.architecture") || "qwen35"
      raise "Not a qwen35 model: arch=#{@arch.inspect}" unless @arch == "qwen35"

      prefix = @arch

      @n_layer        = req_int(g, "#{prefix}.block_count")
      @n_embd         = req_int(g, "#{prefix}.embedding_length")
      @n_ff           = req_int(g, "#{prefix}.feed_forward_length")
      @context_length = req_int(g, "#{prefix}.context_length")

      @n_head    = req_int(g, "#{prefix}.attention.head_count")
      @n_head_kv = req_int(g, "#{prefix}.attention.head_count_kv")
      key_len    = req_int(g, "#{prefix}.attention.key_length")
      val_len    = req_int(g, "#{prefix}.attention.value_length")
      raise "qwen35: key_length (#{key_len}) != value_length (#{val_len})" if key_len != val_len
      @head_dim  = key_len

      @rms_eps = g.get_float("#{prefix}.attention.layer_norm_rms_epsilon").try(&.to_f32) || 1.0e-6_f32

      @full_attention_interval = req_int(g, "#{prefix}.full_attention_interval")

      @rope_freq_base  = g.get_float("#{prefix}.rope.freq_base").try(&.to_f32) || 10_000_000.0_f32
      @rope_dim_count  = req_int(g, "#{prefix}.rope.dimension_count")
      sections         = g.get_int_array("#{prefix}.rope.dimension_sections")
      raise "qwen35: missing rope.dimension_sections" unless sections
      raise "qwen35: rope.dimension_sections must have 4 elements, got #{sections.size}" if sections.size != 4
      @rope_sections = sections.map(&.to_i32)

      @ssm_conv_kernel     = req_int(g, "#{prefix}.ssm.conv_kernel")
      @ssm_state_size      = req_int(g, "#{prefix}.ssm.state_size")
      @ssm_group_count     = req_int(g, "#{prefix}.ssm.group_count")
      @ssm_time_step_rank  = req_int(g, "#{prefix}.ssm.time_step_rank")
      @ssm_inner_size      = req_int(g, "#{prefix}.ssm.inner_size")

      @vocab_size = 0  # Set externally after tokenizer load
    end

    # Is layer `il` a full-attention layer?
    def full_attention?(il : Int32) : Bool
      (il + 1) % @full_attention_interval == 0
    end

    # Is layer `il` a DeltaNet recurrent layer?
    def recurrent?(il : Int32) : Bool
      !full_attention?(il)
    end

    # Derived: full-attention head_dim * n_head (hidden size for Q projection input)
    def attn_head_total_dim : Int32
      @head_dim * @n_head
    end

    # Derived: DeltaNet num_v_heads / num_k_heads ratio (for repeat in non-fused path)
    def ssm_head_v_dim : Int32
      @ssm_inner_size // @ssm_time_step_rank
    end

    # Derived: list of full-attention layer indices
    def full_attention_layers : Array(Int32)
      (0...@n_layer).select { |il| full_attention?(il) }
    end

    # Derived: list of recurrent layer indices
    def recurrent_layers : Array(Int32)
      (0...@n_layer).select { |il| recurrent?(il) }
    end

    private def req_int(g : GGUFFile, key : String) : Int32
      v = g.get_int(key)
      raise "qwen35: missing required key #{key.inspect}" unless v
      v.to_i32
    end
  end
end
