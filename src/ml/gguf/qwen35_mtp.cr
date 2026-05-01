require "./qwen35_meta"
require "./safetensors"

module ML::GGUF
  struct DenseBF16Weight
    getter name : String
    getter raw : Bytes
    getter out_dim : Int32
    getter in_dim : Int32

    def initialize(@name, @raw, @out_dim, @in_dim)
    end
  end

  # Qwen3.6 built-in MTP (multi-token prediction) sidecar weights.
  #
  # Official HF checkpoints store these as `mtp.*` BF16 tensors. Current GGUF
  # conversions commonly drop them, so we keep them in a separate safetensors
  # sidecar until the native runtime can repack/load them from GGUF directly.
  class Qwen35MTPWeights
    REQUIRED_TENSORS = {
      "mtp.fc.weight",
      "mtp.layers.0.input_layernorm.weight",
      "mtp.layers.0.self_attn.q_proj.weight",
      "mtp.layers.0.self_attn.q_norm.weight",
      "mtp.layers.0.self_attn.k_proj.weight",
      "mtp.layers.0.self_attn.k_norm.weight",
      "mtp.layers.0.self_attn.v_proj.weight",
      "mtp.layers.0.self_attn.o_proj.weight",
      "mtp.layers.0.post_attention_layernorm.weight",
      "mtp.layers.0.mlp.gate_proj.weight",
      "mtp.layers.0.mlp.up_proj.weight",
      "mtp.layers.0.mlp.down_proj.weight",
      "mtp.norm.weight",
      "mtp.pre_fc_norm_embedding.weight",
      "mtp.pre_fc_norm_hidden.weight",
    }

    getter fc : DenseBF16Weight
    getter q_proj : DenseBF16Weight
    getter k_proj : DenseBF16Weight
    getter v_proj : DenseBF16Weight
    getter o_proj : DenseBF16Weight
    getter ffn_gate : DenseBF16Weight
    getter ffn_up : DenseBF16Weight
    getter ffn_down : DenseBF16Weight
    getter input_layernorm : Array(Float32)
    getter q_norm : Array(Float32)
    getter k_norm : Array(Float32)
    getter post_attention_layernorm : Array(Float32)
    getter norm : Array(Float32)
    getter pre_fc_norm_embedding : Array(Float32)
    getter pre_fc_norm_hidden : Array(Float32)

    @safetensors : SafetensorsFile

    def initialize(@safetensors : SafetensorsFile)
      missing = REQUIRED_TENSORS.reject { |name| @safetensors.tensor(name) }
      raise "qwen35_mtp: missing tensors: #{missing.join(", ")}" unless missing.empty?

      @fc = load_matrix("mtp.fc.weight")
      @q_proj = load_matrix("mtp.layers.0.self_attn.q_proj.weight")
      @k_proj = load_matrix("mtp.layers.0.self_attn.k_proj.weight")
      @v_proj = load_matrix("mtp.layers.0.self_attn.v_proj.weight")
      @o_proj = load_matrix("mtp.layers.0.self_attn.o_proj.weight")
      @ffn_gate = load_matrix("mtp.layers.0.mlp.gate_proj.weight")
      @ffn_up = load_matrix("mtp.layers.0.mlp.up_proj.weight")
      @ffn_down = load_matrix("mtp.layers.0.mlp.down_proj.weight")
      @input_layernorm = load_vector("mtp.layers.0.input_layernorm.weight")
      @q_norm = load_vector("mtp.layers.0.self_attn.q_norm.weight")
      @k_norm = load_vector("mtp.layers.0.self_attn.k_norm.weight")
      @post_attention_layernorm = load_vector("mtp.layers.0.post_attention_layernorm.weight")
      @norm = load_vector("mtp.norm.weight")
      @pre_fc_norm_embedding = load_vector("mtp.pre_fc_norm_embedding.weight")
      @pre_fc_norm_hidden = load_vector("mtp.pre_fc_norm_hidden.weight")
    end

    def self.from_safetensors(path : String) : Qwen35MTPWeights
      Qwen35MTPWeights.new(SafetensorsFile.new(path))
    end

    def total_raw_bytes : Int64
      @safetensors.tensors.sum(&.data_bytes)
    end

    def validate_for_qwen35!(hparams : Qwen35Hparams) : Nil
      hidden = hparams.n_embd
      ff = hparams.n_ff
      q_dim = 2 * hparams.head_dim * hparams.n_head
      kv_dim = hparams.head_dim * hparams.n_head_kv
      attn_out_dim = hparams.head_dim * hparams.n_head

      expect_matrix(@fc, hidden, hidden * 2)
      expect_matrix(@q_proj, q_dim, hidden)
      expect_matrix(@k_proj, kv_dim, hidden)
      expect_matrix(@v_proj, kv_dim, hidden)
      expect_matrix(@o_proj, hidden, attn_out_dim)
      expect_matrix(@ffn_gate, ff, hidden)
      expect_matrix(@ffn_up, ff, hidden)
      expect_matrix(@ffn_down, hidden, ff)
      expect_vector("input_layernorm", @input_layernorm, hidden)
      expect_vector("post_attention_layernorm", @post_attention_layernorm, hidden)
      expect_vector("norm", @norm, hidden)
      expect_vector("pre_fc_norm_embedding", @pre_fc_norm_embedding, hidden)
      expect_vector("pre_fc_norm_hidden", @pre_fc_norm_hidden, hidden)
      expect_vector("q_norm", @q_norm, hparams.head_dim)
      expect_vector("k_norm", @k_norm, hparams.head_dim)
    end

    private def load_matrix(name : String) : DenseBF16Weight
      info = @safetensors.tensor(name) || raise "qwen35_mtp: missing tensor #{name}"
      raise "qwen35_mtp: #{name} must be BF16" unless info.dtype.bf16?
      raise "qwen35_mtp: #{name} shape must be 2D, got #{info.shape}" unless info.shape.size == 2
      DenseBF16Weight.new(name, @safetensors.tensor_bytes(info), info.shape[0].to_i32, info.shape[1].to_i32)
    end

    private def load_vector(name : String) : Array(Float32)
      info = @safetensors.tensor(name) || raise "qwen35_mtp: missing tensor #{name}"
      raise "qwen35_mtp: #{name} must be BF16" unless info.dtype.bf16?
      raise "qwen35_mtp: #{name} shape must be 1D, got #{info.shape}" unless info.shape.size == 1
      @safetensors.read_tensor_f32(info)
    end

    private def expect_matrix(w : DenseBF16Weight, out_dim : Int32, in_dim : Int32) : Nil
      return if w.out_dim == out_dim && w.in_dim == in_dim
      raise "qwen35_mtp: #{w.name} shape #{w.out_dim}x#{w.in_dim} != expected #{out_dim}x#{in_dim}"
    end

    private def expect_vector(name : String, values : Array(Float32), dim : Int32) : Nil
      return if values.size == dim
      raise "qwen35_mtp: #{name} size #{values.size} != expected #{dim}"
    end
  end
end
