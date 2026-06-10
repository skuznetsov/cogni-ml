require "./reader"
require "./compute"
require "./diffusion_gemma_meta"
{% unless flag?(:cpu_only) %}
  require "./qwen35_metal"
{% end %}

# Structural DiffusionGemma GGUF weight loader.
#
# This keeps tensor data mmap-backed and maps the model-specific tensor names.
# It deliberately does not claim inference support; block-diffusion canvas
# scheduling, self-conditioning, and region-aware masks are separate runtime work.
module ML::GGUF
  struct DiffusionGemmaSelfConditioningWeights
    getter pre_norm : Array(Float32)
    getter gate_qw : QuantWeight
    getter up_qw : QuantWeight
    getter down_qw : QuantWeight

    def initialize(@pre_norm, @gate_qw, @up_qw, @down_qw)
    end
  end

  struct DiffusionGemmaLayerWeights
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
    getter ffn_gate_inp_qw : QuantWeight
    getter ffn_gate_inp_scale : Array(Float32)
    getter ffn_gate_up_exps_qw : QuantWeight?
    getter ffn_gate_exps_qw : QuantWeight?
    getter ffn_up_exps_qw : QuantWeight?
    getter ffn_down_exps_qw : QuantWeight
    getter ffn_down_exps_scale : Array(Float32)

    getter layer_output_scale : Array(Float32)
    getter encoder_layer_output_scale : Array(Float32)
    getter post_attention_norm : Array(Float32)
    getter post_ffw_norm : Array(Float32)
    getter pre_ffw_norm_2 : Array(Float32)
    getter post_ffw_norm_1 : Array(Float32)
    getter post_ffw_norm_2 : Array(Float32)

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
      @ffn_gate_inp_qw,
      @ffn_gate_inp_scale,
      @ffn_gate_up_exps_qw,
      @ffn_gate_exps_qw,
      @ffn_up_exps_qw,
      @ffn_down_exps_qw,
      @ffn_down_exps_scale,
      @layer_output_scale,
      @encoder_layer_output_scale,
      @post_attention_norm,
      @post_ffw_norm,
      @pre_ffw_norm_2,
      @post_ffw_norm_1,
      @post_ffw_norm_2,
    )
    end

    def explicit_v? : Bool
      !@attn_v_qw.nil?
    end

    def reuse_k_as_v? : Bool
      @attn_v_qw.nil?
    end

    def combined_gate_up_experts? : Bool
      !@ffn_gate_up_exps_qw.nil?
    end
  end

  class DiffusionGemmaWeights
    getter hparams : DiffusionGemmaHparams
    getter token_embd : QuantWeight
    getter output_norm : Array(Float32)
    getter rope_freqs : Array(Float32)?
    getter self_conditioning : DiffusionGemmaSelfConditioningWeights
    getter layers : Array(DiffusionGemmaLayerWeights)

    @gguf : GGUFFile

    def initialize(@gguf : GGUFFile, @hparams : DiffusionGemmaHparams)
      @token_embd = load_qw(@gguf, "token_embd.weight")
      @output_norm = load_f32(@gguf, "output_norm.weight")
      @rope_freqs = @gguf.tensor("rope_freqs.weight") ? load_f32(@gguf, "rope_freqs.weight") : nil
      @self_conditioning = DiffusionGemmaSelfConditioningWeights.new(
        pre_norm: load_f32(@gguf, "self_cond_pre_norm.weight"),
        gate_qw: load_qw(@gguf, "self_cond_gate.weight"),
        up_qw: load_qw(@gguf, "self_cond_up.weight"),
        down_qw: load_qw(@gguf, "self_cond_down.weight"),
      )
      @layers = Array(DiffusionGemmaLayerWeights).new(@hparams.n_layer) do |il|
        load_layer(@gguf, il)
      end
      {% unless flag?(:cpu_only) %}
        if Qwen35Metal.available?
          if region = @gguf.mmap_region
            base, size = region
            Qwen35Metal.register_mmap(base, size)
          end
        end
      {% end %}
    end

    def self.from_gguf(path : String) : DiffusionGemmaWeights
      g = GGUFFile.new(path)
      hp = DiffusionGemmaHparams.new(g)
      DiffusionGemmaWeights.new(g, hp)
    end

    private def load_layer(g : GGUFFile, il : Int32) : DiffusionGemmaLayerWeights
      p = "blk.#{il}"
      DiffusionGemmaLayerWeights.new(
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
        ffn_gate_inp_qw: load_qw(g, "#{p}.ffn_gate_inp.weight"),
        ffn_gate_inp_scale: load_f32(g, "#{p}.ffn_gate_inp.scale"),
        ffn_gate_up_exps_qw: load_qw?(g, "#{p}.ffn_gate_up_exps.weight"),
        ffn_gate_exps_qw: load_qw?(g, "#{p}.ffn_gate_exps.weight"),
        ffn_up_exps_qw: load_qw?(g, "#{p}.ffn_up_exps.weight"),
        ffn_down_exps_qw: load_qw(g, "#{p}.ffn_down_exps.weight"),
        ffn_down_exps_scale: load_f32(g, "#{p}.ffn_down_exps.scale"),
        layer_output_scale: load_f32(g, "#{p}.layer_output_scale.weight"),
        encoder_layer_output_scale: load_f32(g, "#{p}.enc_layer_output_scale.weight"),
        post_attention_norm: load_f32(g, "#{p}.post_attention_norm.weight"),
        post_ffw_norm: load_f32(g, "#{p}.post_ffw_norm.weight"),
        pre_ffw_norm_2: load_f32(g, "#{p}.pre_ffw_norm_2.weight"),
        post_ffw_norm_1: load_f32(g, "#{p}.post_ffw_norm_1.weight"),
        post_ffw_norm_2: load_f32(g, "#{p}.post_ffw_norm_2.weight"),
      )
    end

    private def load_qw(g : GGUFFile, name : String) : QuantWeight
      info = g.tensor(name) || raise "diffusion_gemma_weights: missing tensor #{name.inspect}"
      raw = g.read_tensor_raw(info)
      in_dim = info.dims[0].to_i32
      out_dim = flattened_out_dim(info)
      QuantWeight.new(raw, info.type, out_dim, in_dim, "diffusion-gemma:#{name}")
    end

    private def load_qw?(g : GGUFFile, name : String) : QuantWeight?
      return nil unless g.tensor(name)
      load_qw(g, name)
    end

    private def load_f32(g : GGUFFile, name : String) : Array(Float32)
      info = g.tensor(name) || raise "diffusion_gemma_weights: missing tensor #{name.inspect}"
      g.read_tensor_f32(info)
    end

    private def flattened_out_dim(info : TensorInfo) : Int32
      return 1 if info.dims.size < 2
      info.dims[1..].reduce(1_i64) { |a, b| a * b }.to_i32
    end
  end
end
