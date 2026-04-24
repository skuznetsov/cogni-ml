require "./reader"
require "./compute" # for QuantWeight
require "./qwen35_meta"
require "./qwen35_metal"

# Qwen 3.5 / 3.6 weight loader.
#
# Maps GGUF tensor names to per-layer structured weights.
# Two layer variants:
#   Full-attention (every full_attention_interval-th layer, 0-indexed il where (il+1)%interval == 0):
#     attn_norm, attn_q (Q+gate combined), attn_q_norm, attn_k, attn_k_norm,
#     attn_v, attn_output, post_attention_norm, ffn_*
#   Recurrent (DeltaNet): the remaining layers:
#     attn_norm, attn_qkv (combined), attn_gate, ssm_* (a, alpha, beta, conv1d, dt.bias, norm, out),
#     post_attention_norm, ffn_*
#
# FFN in all layers: ffn_gate + ffn_up (SwiGLU pair) + ffn_down.
# Global: token_embd, output_norm, output (lm_head).
#
# Tensor names verified by enumeration against Qwen3.5-9B-Q4_K_M.gguf (2026-04-22).

module ML::GGUF
  # Per-layer weights for full-attention layers.
  struct Qwen35FullAttnWeights
    getter attn_norm : Array(Float32)           # [n_embd]
    getter attn_q_qw : QuantWeight              # [n_embd, 2*head_dim*n_head]  Q+gate
    getter attn_q_norm : Array(Float32)         # [head_dim]
    getter attn_k_qw : QuantWeight              # [n_embd, head_dim*n_head_kv]
    getter attn_k_norm : Array(Float32)         # [head_dim]
    getter attn_v_qw : QuantWeight              # [n_embd, head_dim*n_head_kv]
    getter attn_output_qw : QuantWeight         # [head_dim*n_head, n_embd]
    getter post_attention_norm : Array(Float32) # [n_embd]
    getter ffn_gate_qw : QuantWeight            # [n_embd, n_ff]
    getter ffn_up_qw : QuantWeight              # [n_embd, n_ff]
    getter ffn_down_qw : QuantWeight            # [n_ff, n_embd]

    def initialize(@attn_norm, @attn_q_qw, @attn_q_norm, @attn_k_qw, @attn_k_norm,
                   @attn_v_qw, @attn_output_qw, @post_attention_norm,
                   @ffn_gate_qw, @ffn_up_qw, @ffn_down_qw)
    end
  end

  # Per-layer weights for recurrent (DeltaNet) layers.
  struct Qwen35RecurrentWeights
    getter attn_norm : Array(Float32)           # [n_embd]
    getter attn_qkv_qw : QuantWeight            # [n_embd, qkv_dim] where qkv_dim = 2*num_k_heads*state + num_v_heads*state
    getter attn_gate_qw : QuantWeight           # [n_embd, n_embd]
    getter ssm_a : Array(Float32)               # [num_v_heads]  (also known as "A" decay param)
    getter ssm_alpha_qw : QuantWeight           # [n_embd, num_v_heads]
    getter ssm_beta_qw : QuantWeight            # [n_embd, num_v_heads]
    getter ssm_conv1d : Array(Float32)          # [conv_kernel, qkv_dim - num_v_heads*state]  = [4, 2*num_k_heads*state]
    getter ssm_dt_bias : Array(Float32)         # [num_v_heads]
    getter ssm_norm : Array(Float32)            # [state_size]
    getter ssm_out_qw : QuantWeight             # [n_embd, n_embd]
    getter post_attention_norm : Array(Float32) # [n_embd]
    getter ffn_gate_qw : QuantWeight
    getter ffn_up_qw : QuantWeight
    getter ffn_down_qw : QuantWeight

    def initialize(@attn_norm, @attn_qkv_qw, @attn_gate_qw,
                   @ssm_a, @ssm_alpha_qw, @ssm_beta_qw, @ssm_conv1d,
                   @ssm_dt_bias, @ssm_norm, @ssm_out_qw,
                   @post_attention_norm,
                   @ffn_gate_qw, @ffn_up_qw, @ffn_down_qw)
    end
  end

  alias Qwen35LayerWeights = Qwen35FullAttnWeights | Qwen35RecurrentWeights

  # Top-level weight container for Qwen 3.5 / 3.6 (arch=qwen35).
  class Qwen35Weights
    getter hparams : Qwen35Hparams
    getter token_embd : QuantWeight     # [n_embd, vocab_size]
    getter output_norm : Array(Float32) # [n_embd]
    getter output : QuantWeight         # [n_embd, vocab_size]  (lm_head)
    getter layers : Array(Qwen35LayerWeights)

    # Kept alive so the mmap region backing every QuantWeight.raw stays
    # mapped for the lifetime of the model. Closing it would invalidate
    # both the heap-free slices and any whole-mmap Metal buffer built on
    # top of them.
    @gguf : GGUFFile

    def initialize(@gguf : GGUFFile, @hparams : Qwen35Hparams)
      @token_embd = load_qw(@gguf, "token_embd.weight")
      @output_norm = load_f32(@gguf, "output_norm.weight")
      @output = if @gguf.tensor("output.weight")
                  load_qw(@gguf, "output.weight")
                else
                  # Some small Qwen GGUFs tie lm_head to token embeddings and
                  # omit output.weight. The embedding layout is already
                  # [n_embd, vocab_size], matching the lm-head projection.
                  @token_embd
                end
      @layers = Array(Qwen35LayerWeights).new(@hparams.n_layer) do |il|
        if @hparams.full_attention?(il)
          load_full_attn_layer(@gguf, il)
        else
          load_recurrent_layer(@gguf, il)
        end
      end

      # Register the whole mmap region as a zero-copy Metal buffer so
      # subsequent matmuls dispatch against a byte-offset into it
      # instead of re-uploading weights every call. Cheap no-op in
      # cpu_only mode.
      {% unless flag?(:cpu_only) %}
        if Qwen35Metal.available?
          if region = @gguf.mmap_region
            base, size = region
            Qwen35Metal.register_mmap(base, size)
          end
        end
      {% end %}
    end

    def self.from_gguf(path : String) : Qwen35Weights
      g = GGUFFile.new(path)
      hp = Qwen35Hparams.new(g)
      # Do NOT close `g` here — the mmap backs every QuantWeight.raw;
      # Qwen35Weights keeps a reference to it for its lifetime.
      Qwen35Weights.new(g, hp)
    end

    private def load_full_attn_layer(g : GGUFFile, il : Int32) : Qwen35FullAttnWeights
      p = "blk.#{il}"
      Qwen35FullAttnWeights.new(
        attn_norm: load_f32(g, "#{p}.attn_norm.weight"),
        attn_q_qw: load_qw(g, "#{p}.attn_q.weight"),
        attn_q_norm: load_f32(g, "#{p}.attn_q_norm.weight"),
        attn_k_qw: load_qw(g, "#{p}.attn_k.weight"),
        attn_k_norm: load_f32(g, "#{p}.attn_k_norm.weight"),
        attn_v_qw: load_qw(g, "#{p}.attn_v.weight"),
        attn_output_qw: load_qw(g, "#{p}.attn_output.weight"),
        post_attention_norm: load_f32(g, "#{p}.post_attention_norm.weight"),
        ffn_gate_qw: load_qw(g, "#{p}.ffn_gate.weight"),
        ffn_up_qw: load_qw(g, "#{p}.ffn_up.weight"),
        ffn_down_qw: load_qw(g, "#{p}.ffn_down.weight"),
      )
    end

    private def load_recurrent_layer(g : GGUFFile, il : Int32) : Qwen35RecurrentWeights
      p = "blk.#{il}"
      Qwen35RecurrentWeights.new(
        attn_norm: load_f32(g, "#{p}.attn_norm.weight"),
        attn_qkv_qw: load_qw(g, "#{p}.attn_qkv.weight"),
        attn_gate_qw: load_qw(g, "#{p}.attn_gate.weight"),
        ssm_a: load_f32(g, "#{p}.ssm_a"),
        ssm_alpha_qw: load_qw(g, "#{p}.ssm_alpha.weight"),
        ssm_beta_qw: load_qw(g, "#{p}.ssm_beta.weight"),
        ssm_conv1d: load_f32(g, "#{p}.ssm_conv1d.weight"),
        ssm_dt_bias: load_f32(g, "#{p}.ssm_dt.bias"),
        ssm_norm: load_f32(g, "#{p}.ssm_norm.weight"),
        ssm_out_qw: load_qw(g, "#{p}.ssm_out.weight"),
        post_attention_norm: load_f32(g, "#{p}.post_attention_norm.weight"),
        ffn_gate_qw: load_qw(g, "#{p}.ffn_gate.weight"),
        ffn_up_qw: load_qw(g, "#{p}.ffn_up.weight"),
        ffn_down_qw: load_qw(g, "#{p}.ffn_down.weight"),
      )
    end

    private def load_qw(g : GGUFFile, name : String) : QuantWeight
      info = g.tensor(name) || raise "qwen35_weights: missing tensor #{name.inspect}"
      # Keep raw as a slice into the mmap — GGUFFile stays alive as a
      # field of this Qwen35Weights, so the pointer remains valid and
      # the whole-mmap Metal buffer can address it by offset.
      raw = g.read_tensor_raw(info)
      # GGUF convention: dims=[in_dim, out_dim], row-major with out_dim rows.
      in_dim = info.dims[0].to_i32
      out_dim = info.dims.size >= 2 ? info.dims[1].to_i32 : 1
      QuantWeight.new(raw, info.type, out_dim, in_dim)
    end

    private def load_f32(g : GGUFFile, name : String) : Array(Float32)
      info = g.tensor(name) || raise "qwen35_weights: missing tensor #{name.inspect}"
      g.read_tensor_f32(info)
    end
  end
end
