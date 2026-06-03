require "./reader"
require "./compute"
require "./gemma4_meta"

# Structural Gemma4 GGUF weight loader.
#
# This first slice maps tensor names and keeps the mmap alive. It does not
# implement forward math yet.
module ML::GGUF
  struct Gemma4LayerWeights
    getter attn_norm : Array(Float32)
    getter attn_q_qw : QuantWeight
    getter attn_q_norm : Array(Float32)
    getter attn_k_qw : QuantWeight
    getter attn_k_norm : Array(Float32)
    getter attn_v_qw : QuantWeight?
    getter attn_output_qw : QuantWeight

    getter ffn_norm : Array(Float32)
    getter ffn_gate_qw : QuantWeight
    getter ffn_up_qw : QuantWeight
    getter ffn_down_qw : QuantWeight

    getter layer_output_scale : Array(Float32)
    getter post_attention_norm : Array(Float32)
    getter post_ffw_norm : Array(Float32)

    def initialize(
      @attn_norm,
      @attn_q_qw,
      @attn_q_norm,
      @attn_k_qw,
      @attn_k_norm,
      @attn_v_qw,
      @attn_output_qw,
      @ffn_norm,
      @ffn_gate_qw,
      @ffn_up_qw,
      @ffn_down_qw,
      @layer_output_scale,
      @post_attention_norm,
      @post_ffw_norm,
    )
    end

    def explicit_v? : Bool
      !@attn_v_qw.nil?
    end

    def reuse_k_as_v? : Bool
      @attn_v_qw.nil?
    end
  end

  class Gemma4Weights
    getter hparams : Gemma4Hparams
    getter token_embd : QuantWeight
    getter output_norm : Array(Float32)
    getter rope_freqs : Array(Float32)?
    getter layers : Array(Gemma4LayerWeights)

    # Keep the mmap alive for QuantWeight raw slices.
    @gguf : GGUFFile

    def initialize(@gguf : GGUFFile, @hparams : Gemma4Hparams)
      @token_embd = load_qw(@gguf, "token_embd.weight")
      @output_norm = load_f32(@gguf, "output_norm.weight")
      @rope_freqs = @gguf.tensor("rope_freqs.weight") ? load_f32(@gguf, "rope_freqs.weight") : nil
      @layers = Array(Gemma4LayerWeights).new(@hparams.n_layer) do |il|
        load_layer(@gguf, il)
      end
    end

    def self.from_gguf(path : String) : Gemma4Weights
      g = GGUFFile.new(path)
      hp = Gemma4Hparams.new(g)
      Gemma4Weights.new(g, hp)
    end

    private def load_layer(g : GGUFFile, il : Int32) : Gemma4LayerWeights
      p = "blk.#{il}"
      Gemma4LayerWeights.new(
        attn_norm: load_f32(g, "#{p}.attn_norm.weight"),
        attn_q_qw: load_qw(g, "#{p}.attn_q.weight"),
        attn_q_norm: load_f32(g, "#{p}.attn_q_norm.weight"),
        attn_k_qw: load_qw(g, "#{p}.attn_k.weight"),
        attn_k_norm: load_f32(g, "#{p}.attn_k_norm.weight"),
        attn_v_qw: load_qw?(g, "#{p}.attn_v.weight"),
        attn_output_qw: load_qw(g, "#{p}.attn_output.weight"),
        ffn_norm: load_f32(g, "#{p}.ffn_norm.weight"),
        ffn_gate_qw: load_qw(g, "#{p}.ffn_gate.weight"),
        ffn_up_qw: load_qw(g, "#{p}.ffn_up.weight"),
        ffn_down_qw: load_qw(g, "#{p}.ffn_down.weight"),
        layer_output_scale: load_f32(g, "#{p}.layer_output_scale.weight"),
        post_attention_norm: load_f32(g, "#{p}.post_attention_norm.weight"),
        post_ffw_norm: load_f32(g, "#{p}.post_ffw_norm.weight"),
      )
    end

    private def load_qw(g : GGUFFile, name : String) : QuantWeight
      info = g.tensor(name) || raise "gemma4_weights: missing tensor #{name.inspect}"
      raw = g.read_tensor_raw(info)
      in_dim = info.dims[0].to_i32
      out_dim = info.dims.size >= 2 ? info.dims[1].to_i32 : 1
      QuantWeight.new(raw, info.type, out_dim, in_dim)
    end

    private def load_qw?(g : GGUFFile, name : String) : QuantWeight?
      return nil unless g.tensor(name)
      load_qw(g, name)
    end

    private def load_f32(g : GGUFFile, name : String) : Array(Float32)
      info = g.tensor(name) || raise "gemma4_weights: missing tensor #{name.inspect}"
      g.read_tensor_f32(info)
    end
  end
end
