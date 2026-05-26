require "./qwen35_meta"
require "./qwen35_cpu"
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

      {% if flag?(:qwen35_mtp_metal) %}
        Qwen35Metal.clear_bf16_weight_cache
      {% end %}
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

  # Qwen3.6 GGUF-native MTP block weights.
  #
  # Newer llama.cpp GGUFs keep the MTP predictor as an appended `blk.N` block
  # plus `blk.N.nextn.*` tensors. This loader mirrors llama.cpp's graph_mtp
  # naming and keeps the GGUF mmap alive for QuantWeight slices.
  class Qwen35GGUFMTPWeights
    getter block_index : Int32
    getter nextn_eh_proj_qw : QuantWeight
    getter nextn_enorm : Array(Float32)
    getter nextn_hnorm : Array(Float32)
    getter nextn_embed_tokens_qw : QuantWeight?
    getter nextn_shared_head_qw : QuantWeight?
    getter nextn_shared_head_norm : Array(Float32)?
    getter attn_norm : Array(Float32)
    getter attn_q_qw : QuantWeight
    getter attn_q_norm : Array(Float32)
    getter attn_k_qw : QuantWeight
    getter attn_k_norm : Array(Float32)
    getter attn_v_qw : QuantWeight
    getter attn_output_qw : QuantWeight
    getter post_attention_norm : Array(Float32)
    getter ffn_gate_qw : QuantWeight
    getter ffn_up_qw : QuantWeight
    getter ffn_down_qw : QuantWeight

    @gguf : GGUFFile

    def initialize(@gguf : GGUFFile, hparams : Qwen35Hparams)
      raise "qwen35_gguf_mtp: GGUF has no MTP blocks" unless hparams.nextn_predict_layers > 0
      raise "qwen35_gguf_mtp: only one MTP block is supported, got #{hparams.nextn_predict_layers}" unless hparams.nextn_predict_layers == 1

      @block_index = hparams.n_layer
      p = "blk.#{@block_index}"
      @nextn_eh_proj_qw = load_qw(@gguf, "#{p}.nextn.eh_proj.weight")
      @nextn_enorm = load_f32(@gguf, "#{p}.nextn.enorm.weight")
      @nextn_hnorm = load_f32(@gguf, "#{p}.nextn.hnorm.weight")
      @nextn_embed_tokens_qw = load_qw?(@gguf, "#{p}.nextn.embed_tokens.weight")
      @nextn_shared_head_qw = load_qw?(@gguf, "#{p}.nextn.shared_head_head.weight")
      @nextn_shared_head_norm = load_f32?(@gguf, "#{p}.nextn.shared_head_norm.weight")
      @attn_norm = load_f32(@gguf, "#{p}.attn_norm.weight")
      @attn_q_qw = load_qw(@gguf, "#{p}.attn_q.weight")
      @attn_q_norm = load_f32(@gguf, "#{p}.attn_q_norm.weight")
      @attn_k_qw = load_qw(@gguf, "#{p}.attn_k.weight")
      @attn_k_norm = load_f32(@gguf, "#{p}.attn_k_norm.weight")
      @attn_v_qw = load_qw(@gguf, "#{p}.attn_v.weight")
      @attn_output_qw = load_qw(@gguf, "#{p}.attn_output.weight")
      @post_attention_norm = load_f32(@gguf, "#{p}.post_attention_norm.weight")
      @ffn_gate_qw = load_qw(@gguf, "#{p}.ffn_gate.weight")
      @ffn_up_qw = load_qw(@gguf, "#{p}.ffn_up.weight")
      @ffn_down_qw = load_qw(@gguf, "#{p}.ffn_down.weight")

      validate_for_qwen35!(hparams)

      {% unless flag?(:cpu_only) %}
        if Qwen35Metal.available?
          if region = @gguf.mmap_region
            base, size = region
            Qwen35Metal.register_mmap(base, size)
          end
        end
      {% end %}
    end

    def self.from_gguf(path : String, hparams : Qwen35Hparams) : Qwen35GGUFMTPWeights
      Qwen35GGUFMTPWeights.new(GGUFFile.new(path), hparams)
    end

    def total_raw_bytes : Int64
      qws = [
        @nextn_eh_proj_qw, @attn_q_qw, @attn_k_qw, @attn_v_qw, @attn_output_qw,
        @ffn_gate_qw, @ffn_up_qw, @ffn_down_qw,
      ]
      qws.sum { |qw| qw.raw.size.to_i64 } +
        (@nextn_embed_tokens_qw.try(&.raw.size.to_i64) || 0_i64) +
        (@nextn_shared_head_qw.try(&.raw.size.to_i64) || 0_i64)
    end

    def token_embd_qw(default_qw : QuantWeight) : QuantWeight
      @nextn_embed_tokens_qw || default_qw
    end

    def shared_head_qw(default_qw : QuantWeight) : QuantWeight
      @nextn_shared_head_qw || default_qw
    end

    def shared_head_norm(default_norm : Array(Float32)) : Array(Float32)
      @nextn_shared_head_norm || default_norm
    end

    def validate_for_qwen35!(hparams : Qwen35Hparams) : Nil
      hidden = hparams.n_embd
      ff = hparams.n_ff
      q_dim = 2 * hparams.head_dim * hparams.n_head
      kv_dim = hparams.head_dim * hparams.n_head_kv
      attn_out_dim = hparams.head_dim * hparams.n_head

      expect_matrix(@nextn_eh_proj_qw, hidden, hidden * 2)
      expect_matrix(@attn_q_qw, q_dim, hidden)
      expect_matrix(@attn_k_qw, kv_dim, hidden)
      expect_matrix(@attn_v_qw, kv_dim, hidden)
      expect_matrix(@attn_output_qw, hidden, attn_out_dim)
      expect_matrix(@ffn_gate_qw, ff, hidden)
      expect_matrix(@ffn_up_qw, ff, hidden)
      expect_matrix(@ffn_down_qw, hidden, ff)
      expect_vector("nextn.enorm", @nextn_enorm, hidden)
      expect_vector("nextn.hnorm", @nextn_hnorm, hidden)
      expect_vector("attn_norm", @attn_norm, hidden)
      expect_vector("post_attention_norm", @post_attention_norm, hidden)
      expect_vector("attn_q_norm", @attn_q_norm, hparams.head_dim)
      expect_vector("attn_k_norm", @attn_k_norm, hparams.head_dim)
      if norm = @nextn_shared_head_norm
        expect_vector("nextn.shared_head_norm", norm, hidden)
      end
      if emb = @nextn_embed_tokens_qw
        expect_matrix(emb, hparams.vocab_size > 0 ? hparams.vocab_size : emb.out_dim, hidden)
      end
      if head = @nextn_shared_head_qw
        expect_matrix(head, hparams.vocab_size > 0 ? hparams.vocab_size : head.out_dim, hidden)
      end
    end

    private def load_qw(g : GGUFFile, name : String) : QuantWeight
      info = g.tensor(name) || raise "qwen35_gguf_mtp: missing tensor #{name.inspect}"
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
      info = g.tensor(name) || raise "qwen35_gguf_mtp: missing tensor #{name.inspect}"
      g.read_tensor_f32(info)
    end

    private def load_f32?(g : GGUFFile, name : String) : Array(Float32)?
      info = g.tensor(name)
      info ? g.read_tensor_f32(info) : nil
    end

    private def expect_matrix(w : QuantWeight, out_dim : Int32, in_dim : Int32) : Nil
      return if w.out_dim == out_dim && w.in_dim == in_dim
      raise "qwen35_gguf_mtp: weight shape #{w.out_dim}x#{w.in_dim} != expected #{out_dim}x#{in_dim}"
    end

    private def expect_vector(name : String, values : Array(Float32), dim : Int32) : Nil
      return if values.size == dim
      raise "qwen35_gguf_mtp: #{name} size #{values.size} != expected #{dim}"
    end
  end

  module Qwen35MTP
    extend self

    MTP_METAL_MIN_BYTES = 1_048_576

    @@profile_counts = Hash(String, Int64).new(0_i64)
    @@profile_ns = Hash(String, Int64).new(0_i64)
    @@profile_min_ns = Hash(String, Int64).new(Int64::MAX)
    @@profile_max_ns = Hash(String, Int64).new(0_i64)

    def profile_enabled? : Bool
      ENV["QWEN35_MTP_STAGE_PROFILE"]? == "1"
    end

    def profile_reset : Nil
      @@profile_counts.clear
      @@profile_ns.clear
      @@profile_min_ns.clear
      @@profile_max_ns.clear
    end

    def profile_report : String
      total_ns = @@profile_ns.values.sum(0_i64)
      lines = ["mtp_stage_profile total_ms=#{(total_ns.to_f64 / 1_000_000.0).round(3)} stages=#{@@profile_counts.size}"]
      @@profile_ns.to_a.sort_by { |(_name, ns)| -ns }.each do |(name, ns)|
        count = @@profile_counts[name]
        ms = ns.to_f64 / 1_000_000.0
        avg = count > 0 ? ms / count.to_f64 : 0.0
        min_ms = @@profile_min_ns[name].to_f64 / 1_000_000.0
        max_ms = @@profile_max_ns[name].to_f64 / 1_000_000.0
        pct = total_ns > 0 ? (ns.to_f64 * 100.0 / total_ns.to_f64) : 0.0
        lines << "mtp_stage name=#{name} count=#{count} ms=#{ms.round(3)} avg_ms=#{avg.round(3)} min_ms=#{min_ms.round(3)} max_ms=#{max_ms.round(3)} pct=#{pct.round(2)}"
      end
      lines.join('\n')
    end

    private def profile_stage(name : String, &block : -> T) : T forall T
      unless profile_enabled?
        return yield
      end

      t0 = Time.instant
      result = yield
      elapsed = (Time.instant - t0).total_nanoseconds.to_i64
      @@profile_counts[name] += 1
      @@profile_ns[name] += elapsed
      @@profile_min_ns[name] = elapsed if elapsed < @@profile_min_ns[name]
      @@profile_max_ns[name] = elapsed if elapsed > @@profile_max_ns[name]
      result
    end

    class State
      getter k_cache : Array(Float32)
      getter v_cache : Array(Float32)
      getter kv_dim : Int32
      getter max_seq : Int32
      property length : Int32 = 0

      def initialize(@max_seq : Int32, @kv_dim : Int32)
        @k_cache = Array(Float32).new(@max_seq * @kv_dim, 0.0_f32)
        @v_cache = Array(Float32).new(@max_seq * @kv_dim, 0.0_f32)
      end

      def append!(k : Array(Float32), v : Array(Float32)) : Int32
        raise ArgumentError.new("mtp state full: #{@length} >= #{@max_seq}") if @length >= @max_seq
        raise ArgumentError.new("mtp k size #{k.size} != #{@kv_dim}") unless k.size == @kv_dim
        raise ArgumentError.new("mtp v size #{v.size} != #{@kv_dim}") unless v.size == @kv_dim

        base = @length * @kv_dim
        @kv_dim.times do |i|
          @k_cache[base + i] = k[i]
          @v_cache[base + i] = v[i]
        end
        @length += 1
        @length - 1
      end

      def truncate!(new_length : Int32) : Nil
        raise ArgumentError.new("mtp truncate length #{new_length} outside 0..#{@length}") unless 0 <= new_length <= @length
        @length = new_length
      end

      def fork : State
        copy = State.new(@max_seq, @kv_dim)
        used = @length * @kv_dim
        used.times do |i|
          copy.k_cache[i] = @k_cache[i]
          copy.v_cache[i] = @v_cache[i]
        end
        copy.length = @length
        copy
      end
    end

    def bf16_at(raw : Bytes, i : Int32) : Float32
      off = i * 2
      bits = (raw[off + 1].to_u32 << 24) | (raw[off].to_u32 << 16)
      bits.unsafe_as(Float32)
    end

    private def use_metal_bf16?(w : DenseBF16Weight) : Bool
      {% if flag?(:qwen35_mtp_metal) %}
        return false if ENV["QWEN35_MTP_BF16_METAL_OFF"]? == "1"
        return false unless ENV["QWEN35_MTP_BF16_METAL"]? == "1" || w.raw.size >= MTP_METAL_MIN_BYTES
        Qwen35Metal.available?
      {% else %}
        false
      {% end %}
    end

    def bf16_backend_label : String
      {% if flag?(:qwen35_mtp_metal) %}
        ENV["QWEN35_MTP_BF16_METAL_OFF"]? == "1" ? "cpu_bf16" : "metal_bf16"
      {% else %}
        "cpu_bf16"
      {% end %}
    end

    private def sidecar_norm_weight(w : Array(Float32)) : Array(Float32)
      Array(Float32).new(w.size) { |i| 1.0_f32 + w[i] }
    end

    def matvec_bf16(w : DenseBF16Weight, x : Array(Float32)) : Array(Float32)
      raise ArgumentError.new("#{w.name}: expected input #{w.in_dim}, got #{x.size}") unless x.size == w.in_dim
      expected = w.out_dim.to_i64 * w.in_dim.to_i64 * 2_i64
      raise "#{w.name}: raw size #{w.raw.size} != expected #{expected}" unless w.raw.size.to_i64 == expected

      {% if flag?(:qwen35_mtp_metal) %}
        if use_metal_bf16?(w)
          return Qwen35Metal.bf16_gemv(w.raw, w.in_dim, w.out_dim, x)
        end
      {% end %}

      out = Array(Float32).new(w.out_dim, 0.0_f32)
      w.out_dim.times do |row|
        base = row * w.in_dim
        sum = 0.0_f32
        w.in_dim.times do |col|
          sum += bf16_at(w.raw, base + col) * x[col]
        end
        out[row] = sum
      end
      out
    end

    def matvec_bf16_rows(w : DenseBF16Weight, x : Array(Float32), rows : Array(Int32)) : Array(Float32)
      raise ArgumentError.new("#{w.name}: expected input #{w.in_dim}, got #{x.size}") unless x.size == w.in_dim
      expected = w.out_dim.to_i64 * w.in_dim.to_i64 * 2_i64
      raise "#{w.name}: raw size #{w.raw.size} != expected #{expected}" unless w.raw.size.to_i64 == expected

      out = Array(Float32).new(rows.size, 0.0_f32)
      rows.each_with_index do |row, out_i|
        raise ArgumentError.new("#{w.name}: row #{row} out of range 0...#{w.out_dim}") if row < 0 || row >= w.out_dim
        base = row * w.in_dim
        sum = 0.0_f32
        w.in_dim.times do |col|
          sum += bf16_at(w.raw, base + col) * x[col]
        end
        out[out_i] = sum
      end
      out
    end

    def matvec_bf16_q_gate_rows(w : DenseBF16Weight, x : Array(Float32),
                                n_head : Int32, head_dim : Int32) : Array(Float32)
      q_dim = n_head * head_dim
      raise ArgumentError.new("#{w.name}: expected input #{w.in_dim}, got #{x.size}") unless x.size == w.in_dim
      raise ArgumentError.new("#{w.name}: expected q_proj rows #{q_dim * 2}, got #{w.out_dim}") unless w.out_dim == q_dim * 2
      expected = w.out_dim.to_i64 * w.in_dim.to_i64 * 2_i64
      raise "#{w.name}: raw size #{w.raw.size} != expected #{expected}" unless w.raw.size.to_i64 == expected

      {% if flag?(:qwen35_mtp_metal) %}
        if use_metal_bf16?(w)
          return Qwen35Metal.bf16_q_gate_gemv(w.raw, w.in_dim, q_dim, head_dim, x)
        end
      {% end %}

      out = Array(Float32).new(q_dim, 0.0_f32)
      n_head.times do |h|
        src_base = h * 2 * head_dim + head_dim
        dst_base = h * head_dim
        head_dim.times do |d|
          row_base = (src_base + d) * w.in_dim
          sum = 0.0_f32
          w.in_dim.times do |col|
            sum += bf16_at(w.raw, row_base + col) * x[col]
          end
          out[dst_base + d] = sum
        end
      end
      out
    end

    private def ffn_project_gguf_metal(cur : Array(Float32),
                                       gate_qw : QuantWeight,
                                       up_qw : QuantWeight,
                                       down_qw : QuantWeight) : Array(Float32)?
      {% unless flag?(:cpu_only) %}
        return nil if ENV["QWEN35_MTP_FFN_FUSE_OFF"]? == "1"
        return nil unless Qwen35Metal.available?
        Qwen35Metal.ffn_project(cur, gate_qw, up_qw, down_qw)
      {% else %}
        nil
      {% end %}
    end

    # HF Qwen3.5 RMSNorm stores a learned delta and applies `(1 + weight)`.
    # GGUF target norms are converted for llama.cpp, but the MTP sidecar is raw
    # safetensors, so the sidecar path must apply the +1 here.
    def rms_norm_sidecar(x : Array(Float32), w : Array(Float32), eps : Float32) : Array(Float32)
      dim = x.size
      raise ArgumentError.new("rms_norm_sidecar weight size #{w.size} != #{dim}") unless w.size == dim
      ss = 0.0_f64
      dim.times { |j| ss += x[j].to_f64 * x[j].to_f64 }
      inv_rms = (1.0 / Math.sqrt(ss / dim.to_f64 + eps.to_f64)).to_f32
      Array(Float32).new(dim) { |j| x[j] * inv_rms * (1.0_f32 + w[j]) }
    end

    def rms_norm_sidecar_slice!(x : Array(Float32), offset : Int32, len : Int32,
                                w : Array(Float32), eps : Float32) : Nil
      raise ArgumentError.new("rms_norm_sidecar_slice weight size #{w.size} != #{len}") unless w.size == len
      ss = 0.0_f64
      len.times { |j| ss += x[offset + j].to_f64 * x[offset + j].to_f64 }
      inv_rms = (1.0 / Math.sqrt(ss / len.to_f64 + eps.to_f64)).to_f32
      len.times { |j| x[offset + j] = x[offset + j] * inv_rms * (1.0_f32 + w[j]) }
    end

    def rms_norm_gguf(x : Array(Float32), w : Array(Float32), eps : Float32) : Array(Float32)
      Qwen35CPU.rms_norm(x, w, eps)
    end

    def rms_norm_gguf_slice!(x : Array(Float32), offset : Int32, len : Int32,
                             w : Array(Float32), eps : Float32) : Nil
      Qwen35CPU.rms_norm_slice!(x, offset, len, w, eps)
    end

    # GGUF-native one-token MTP hidden. This is the correctness-first path for
    # llama.cpp-style appended `blk.N.nextn.*` MTP blocks. It intentionally keeps
    # the math explicit; once validated, the same operator sequence can be fused.
    def forward_one_hidden_gguf(weights : Qwen35Weights,
                                mtp : Qwen35GGUFMTPWeights,
                                prev_hidden : Array(Float32),
                                token_id : Int32,
                                pos : Int32,
                                mtp_state : State? = nil) : Array(Float32)
      hp = weights.hparams
      hidden = hp.n_embd
      raise ArgumentError.new("prev_hidden size #{prev_hidden.size} != #{hidden}") unless prev_hidden.size == hidden

      emb = profile_stage("gguf.embed_lookup") { Qwen35CPU.embedding_lookup(mtp.token_embd_qw(weights.token_embd), token_id) }
      emb_norm = profile_stage("gguf.nextn_enorm") { rms_norm_gguf(emb, mtp.nextn_enorm, hp.rms_eps) }
      hidden_norm = profile_stage("gguf.nextn_hnorm") { rms_norm_gguf(prev_hidden, mtp.nextn_hnorm, hp.rms_eps) }

      fc_in = Array(Float32).new(hidden * 2, 0.0_f32)
      hidden.times do |i|
        fc_in[i] = emb_norm[i]
        fc_in[hidden + i] = hidden_norm[i]
      end

      n_head = hp.n_head
      n_head_kv = hp.n_head_kv
      head_dim = hp.head_dim
      q_dim = n_head * head_dim
      kv_dim = n_head_kv * head_dim
      heads_per_group = n_head // n_head_kv

      residual = profile_stage("gguf.eh_proj") { Qwen35CPU.qmatvec_nobias(mtp.nextn_eh_proj_qw, fc_in) }
      cur = profile_stage("gguf.attn_norm") { rms_norm_gguf(residual, mtp.attn_norm, hp.rms_eps) }
      attn_o = Array(Float32).new(q_dim, 0.0_f32)

      if cache = mtp_state
        raise "mtp state kv_dim #{cache.kv_dim} != expected #{kv_dim}" unless cache.kv_dim == kv_dim
        qkv = profile_stage("gguf.attn_qkv") { Qwen35CPU.qmatvec_many([mtp.attn_q_qw, mtp.attn_k_qw, mtp.attn_v_qw], cur) }
        q_full = qkv[0]
        k = qkv[1]
        v = qkv[2]
        raise "mtp q_proj produced #{q_full.size}, expected #{q_dim * 2}" unless q_full.size == q_dim * 2
        raise "mtp k_proj produced #{k.size}, expected #{kv_dim}" unless k.size == kv_dim
        raise "mtp v_proj produced #{v.size}, expected #{kv_dim}" unless v.size == kv_dim

        q = Array(Float32).new(q_dim, 0.0_f32)
        gate = Array(Float32).new(q_dim, 0.0_f32)
        n_head.times do |h|
          src_base = h * 2 * head_dim
          dst_base = h * head_dim
          head_dim.times do |d|
            q[dst_base + d] = q_full[src_base + d]
            gate[dst_base + d] = q_full[src_base + head_dim + d]
          end
        end

        profile_stage("gguf.attn_qk_norm_rope") do
          n_head.times do |h|
            rms_norm_gguf_slice!(q, h * head_dim, head_dim, mtp.attn_q_norm, hp.rms_eps)
            Qwen35CPU.rope_partial!(q, h * head_dim, hp.rope_dim_count, head_dim, pos, hp.rope_freq_base)
          end
          n_head_kv.times do |h|
            rms_norm_gguf_slice!(k, h * head_dim, head_dim, mtp.attn_k_norm, hp.rms_eps)
            Qwen35CPU.rope_partial!(k, h * head_dim, hp.rope_dim_count, head_dim, pos, hp.rope_freq_base)
          end
        end

        profile_stage("gguf.attn_cache_append") do
          cache.append!(k, v)
          nil
        end
        scale = (1.0 / Math.sqrt(head_dim.to_f64)).to_f32
        scores = Array(Float32).new(cache.length, 0.0_f32)
        probs = Array(Float32).new(cache.length, 0.0_f32)

        profile_stage("gguf.attn_softmax_cpu") do
          n_head.times do |h|
            kvh = h // heads_per_group
            q_base = h * head_dim
            max_score = -Float32::INFINITY
            cache.length.times do |t|
              k_base = t * kv_dim + kvh * head_dim
              dot = 0.0_f32
              head_dim.times { |d| dot += q[q_base + d] * cache.k_cache[k_base + d] }
              score = dot * scale
              scores[t] = score
              max_score = score if score > max_score
            end

            denom = 0.0_f32
            cache.length.times do |t|
              p = Math.exp((scores[t] - max_score).to_f64).to_f32
              probs[t] = p
              denom += p
            end
            inv_denom = 1.0_f32 / denom

            head_dim.times do |d|
              sum = 0.0_f32
              cache.length.times do |t|
                v_base = t * kv_dim + kvh * head_dim
                sum += probs[t] * inv_denom * cache.v_cache[v_base + d]
              end
              attn_o[q_base + d] = sum * Qwen35CPU.sigmoid(gate[q_base + d])
            end
          end
        end
      else
        qv = profile_stage("gguf.attn_qv") { Qwen35CPU.qmatvec_many([mtp.attn_q_qw, mtp.attn_v_qw], cur) }
        q_full = qv[0]
        v = qv[1]
        raise "mtp q_proj produced #{q_full.size}, expected #{q_dim * 2}" unless q_full.size == q_dim * 2
        raise "mtp v_proj produced #{v.size}, expected #{kv_dim}" unless v.size == kv_dim

        profile_stage("gguf.attn_one_token_cpu") do
          n_head.times do |h|
            src_base = h * 2 * head_dim + head_dim
            kvh = h // heads_per_group
            q_base = h * head_dim
            kv_base = kvh * head_dim
            head_dim.times do |d|
              attn_o[q_base + d] = v[kv_base + d] * Qwen35CPU.sigmoid(q_full[src_base + d])
            end
          end
        end
      end

      attn_out = profile_stage("gguf.attn_output") { Qwen35CPU.qmatvec_nobias(mtp.attn_output_qw, attn_o) }
      after_attn = profile_stage("gguf.attn_residual") { Array(Float32).new(hidden) { |i| residual[i] + attn_out[i] } }
      cur2 = profile_stage("gguf.post_attn_norm") { rms_norm_gguf(after_attn, mtp.post_attention_norm, hp.rms_eps) }

      ffn_out = profile_stage("gguf.ffn_project_fused") do
        ffn_project_gguf_metal(cur2, mtp.ffn_gate_qw, mtp.ffn_up_qw, mtp.ffn_down_qw)
      end
      unless ffn_out
        gu = profile_stage("gguf.ffn_upgate") { Qwen35CPU.qmatvec_many([mtp.ffn_gate_qw, mtp.ffn_up_qw], cur2) }
        gate_ff = gu[0]
        up_ff = gu[1]
        combined = profile_stage("gguf.ffn_swiglu_cpu") do
          Qwen35CPU.silu!(gate_ff)
          Array(Float32).new(gate_ff.size) { |i| gate_ff[i] * up_ff[i] }
        end
        ffn_out = profile_stage("gguf.ffn_down") { Qwen35CPU.qmatvec_nobias(mtp.ffn_down_qw, combined) }
      end
      profile_stage("gguf.ffn_residual") { Array(Float32).new(hidden) { |i| after_attn[i] + ffn_out[i] } }
    end

    def hidden_top1_gguf(weights : Qwen35Weights,
                         mtp : Qwen35GGUFMTPWeights,
                         hidden_pre_norm : Array(Float32)) : {Int32, Float32}
      profile_stage("gguf.head_top1") do
        Qwen35CPU.hidden_top1_with_norm(
          hidden_pre_norm,
          mtp.shared_head_norm(weights.output_norm),
          mtp.shared_head_qw(weights.output),
          weights.hparams.rms_eps,
        )
      end
    end

    def forward_one_hidden_top1_gguf(weights : Qwen35Weights,
                                     mtp : Qwen35GGUFMTPWeights,
                                     prev_hidden : Array(Float32),
                                     token_id : Int32,
                                     pos : Int32,
                                     mtp_state : State? = nil) : NamedTuple(hidden: Array(Float32), top1: {Int32, Float32})
      unless ENV["QWEN35_MTP_FFN_HEAD_FUSE"]? == "1"
        hidden = forward_one_hidden_gguf(weights, mtp, prev_hidden, token_id, pos, mtp_state)
        return {hidden: hidden, top1: hidden_top1_gguf(weights, mtp, hidden)}
      end

      hp = weights.hparams
      hidden = hp.n_embd
      raise ArgumentError.new("prev_hidden size #{prev_hidden.size} != #{hidden}") unless prev_hidden.size == hidden

      emb = profile_stage("gguf.embed_lookup") { Qwen35CPU.embedding_lookup(mtp.token_embd_qw(weights.token_embd), token_id) }
      emb_norm = profile_stage("gguf.nextn_enorm") { rms_norm_gguf(emb, mtp.nextn_enorm, hp.rms_eps) }
      hidden_norm = profile_stage("gguf.nextn_hnorm") { rms_norm_gguf(prev_hidden, mtp.nextn_hnorm, hp.rms_eps) }

      fc_in = Array(Float32).new(hidden * 2, 0.0_f32)
      hidden.times do |i|
        fc_in[i] = emb_norm[i]
        fc_in[hidden + i] = hidden_norm[i]
      end

      n_head = hp.n_head
      n_head_kv = hp.n_head_kv
      head_dim = hp.head_dim
      q_dim = n_head * head_dim
      kv_dim = n_head_kv * head_dim
      heads_per_group = n_head // n_head_kv

      residual = profile_stage("gguf.eh_proj") { Qwen35CPU.qmatvec_nobias(mtp.nextn_eh_proj_qw, fc_in) }
      cur = profile_stage("gguf.attn_norm") { rms_norm_gguf(residual, mtp.attn_norm, hp.rms_eps) }
      attn_o = Array(Float32).new(q_dim, 0.0_f32)

      if cache = mtp_state
        raise "mtp state kv_dim #{cache.kv_dim} != expected #{kv_dim}" unless cache.kv_dim == kv_dim
        qkv = profile_stage("gguf.attn_qkv") { Qwen35CPU.qmatvec_many([mtp.attn_q_qw, mtp.attn_k_qw, mtp.attn_v_qw], cur) }
        q_full = qkv[0]
        k = qkv[1]
        v = qkv[2]
        raise "mtp q_proj produced #{q_full.size}, expected #{q_dim * 2}" unless q_full.size == q_dim * 2
        raise "mtp k_proj produced #{k.size}, expected #{kv_dim}" unless k.size == kv_dim
        raise "mtp v_proj produced #{v.size}, expected #{kv_dim}" unless v.size == kv_dim

        q = Array(Float32).new(q_dim, 0.0_f32)
        gate = Array(Float32).new(q_dim, 0.0_f32)
        n_head.times do |h|
          src_base = h * 2 * head_dim
          dst_base = h * head_dim
          head_dim.times do |d|
            q[dst_base + d] = q_full[src_base + d]
            gate[dst_base + d] = q_full[src_base + head_dim + d]
          end
        end

        profile_stage("gguf.attn_qk_norm_rope") do
          n_head.times do |h|
            rms_norm_gguf_slice!(q, h * head_dim, head_dim, mtp.attn_q_norm, hp.rms_eps)
            Qwen35CPU.rope_partial!(q, h * head_dim, hp.rope_dim_count, head_dim, pos, hp.rope_freq_base)
          end
          n_head_kv.times do |h|
            rms_norm_gguf_slice!(k, h * head_dim, head_dim, mtp.attn_k_norm, hp.rms_eps)
            Qwen35CPU.rope_partial!(k, h * head_dim, hp.rope_dim_count, head_dim, pos, hp.rope_freq_base)
          end
        end

        profile_stage("gguf.attn_cache_append") do
          cache.append!(k, v)
          nil
        end
        scale = (1.0 / Math.sqrt(head_dim.to_f64)).to_f32
        scores = Array(Float32).new(cache.length, 0.0_f32)
        probs = Array(Float32).new(cache.length, 0.0_f32)

        profile_stage("gguf.attn_softmax_cpu") do
          n_head.times do |h|
            kvh = h // heads_per_group
            q_base = h * head_dim
            max_score = -Float32::INFINITY
            cache.length.times do |t|
              k_base = t * kv_dim + kvh * head_dim
              dot = 0.0_f32
              head_dim.times { |d| dot += q[q_base + d] * cache.k_cache[k_base + d] }
              score = dot * scale
              scores[t] = score
              max_score = score if score > max_score
            end

            denom = 0.0_f32
            cache.length.times do |t|
              p = Math.exp((scores[t] - max_score).to_f64).to_f32
              probs[t] = p
              denom += p
            end
            inv_denom = 1.0_f32 / denom

            head_dim.times do |d|
              sum = 0.0_f32
              cache.length.times do |t|
                v_base = t * kv_dim + kvh * head_dim
                sum += probs[t] * inv_denom * cache.v_cache[v_base + d]
              end
              attn_o[q_base + d] = sum * Qwen35CPU.sigmoid(gate[q_base + d])
            end
          end
        end
      else
        qv = profile_stage("gguf.attn_qv") { Qwen35CPU.qmatvec_many([mtp.attn_q_qw, mtp.attn_v_qw], cur) }
        q_full = qv[0]
        v = qv[1]
        raise "mtp q_proj produced #{q_full.size}, expected #{q_dim * 2}" unless q_full.size == q_dim * 2
        raise "mtp v_proj produced #{v.size}, expected #{kv_dim}" unless v.size == kv_dim

        profile_stage("gguf.attn_one_token_cpu") do
          n_head.times do |h|
            src_base = h * 2 * head_dim + head_dim
            kvh = h // heads_per_group
            q_base = h * head_dim
            kv_base = kvh * head_dim
            head_dim.times do |d|
              attn_o[q_base + d] = v[kv_base + d] * Qwen35CPU.sigmoid(q_full[src_base + d])
            end
          end
        end
      end

      attn_out = profile_stage("gguf.attn_output") { Qwen35CPU.qmatvec_nobias(mtp.attn_output_qw, attn_o) }
      after_attn = profile_stage("gguf.attn_residual") { Array(Float32).new(hidden) { |i| residual[i] + attn_out[i] } }
      cur2 = profile_stage("gguf.post_attn_norm") { rms_norm_gguf(after_attn, mtp.post_attention_norm, hp.rms_eps) }

      if fused = profile_stage("gguf.ffn_head_fused") {
           Qwen35Metal.ffn_project_residual_top1(
             cur2,
             after_attn,
             mtp.ffn_gate_qw,
             mtp.ffn_up_qw,
             mtp.ffn_down_qw,
             mtp.shared_head_norm(weights.output_norm),
             mtp.shared_head_qw(weights.output),
             hp.rms_eps)
         }
        return fused
      end

      ffn_out = profile_stage("gguf.ffn_project_fused") do
        ffn_project_gguf_metal(cur2, mtp.ffn_gate_qw, mtp.ffn_up_qw, mtp.ffn_down_qw)
      end
      unless ffn_out
        gu = profile_stage("gguf.ffn_upgate") { Qwen35CPU.qmatvec_many([mtp.ffn_gate_qw, mtp.ffn_up_qw], cur2) }
        gate_ff = gu[0]
        up_ff = gu[1]
        combined = profile_stage("gguf.ffn_swiglu_cpu") do
          Qwen35CPU.silu!(gate_ff)
          Array(Float32).new(gate_ff.size) { |i| gate_ff[i] * up_ff[i] }
        end
        ffn_out = profile_stage("gguf.ffn_down") { Qwen35CPU.qmatvec_nobias(mtp.ffn_down_qw, combined) }
      end
      hidden_pre_norm = profile_stage("gguf.ffn_residual") { Array(Float32).new(hidden) { |i| after_attn[i] + ffn_out[i] } }
      {hidden: hidden_pre_norm, top1: hidden_top1_gguf(weights, mtp, hidden_pre_norm)}
    end

    def forward_one_hidden_top2_gguf(weights : Qwen35Weights,
                                     mtp : Qwen35GGUFMTPWeights,
                                     prev_hidden : Array(Float32),
                                     token_id : Int32,
                                     pos : Int32,
                                     mtp_state : State? = nil) : NamedTuple(hidden: Array(Float32), top2: Array({Int32, Float32}))
      unless ENV["QWEN35_MTP_FFN_HEAD_FUSE"]? == "1" && ENV["QWEN35_MTP_FFN_HEAD_TOP2_FUSE_OFF"]? != "1"
        hidden = forward_one_hidden_gguf(weights, mtp, prev_hidden, token_id, pos, mtp_state)
        return {hidden: hidden, top2: hidden_top2_gguf(weights, mtp, hidden)}
      end

      hp = weights.hparams
      hidden = hp.n_embd
      raise ArgumentError.new("prev_hidden size #{prev_hidden.size} != #{hidden}") unless prev_hidden.size == hidden

      emb = profile_stage("gguf.embed_lookup") { Qwen35CPU.embedding_lookup(mtp.token_embd_qw(weights.token_embd), token_id) }
      emb_norm = profile_stage("gguf.nextn_enorm") { rms_norm_gguf(emb, mtp.nextn_enorm, hp.rms_eps) }
      hidden_norm = profile_stage("gguf.nextn_hnorm") { rms_norm_gguf(prev_hidden, mtp.nextn_hnorm, hp.rms_eps) }

      fc_in = Array(Float32).new(hidden * 2, 0.0_f32)
      hidden.times do |i|
        fc_in[i] = emb_norm[i]
        fc_in[hidden + i] = hidden_norm[i]
      end

      n_head = hp.n_head
      n_head_kv = hp.n_head_kv
      head_dim = hp.head_dim
      q_dim = n_head * head_dim
      kv_dim = n_head_kv * head_dim
      heads_per_group = n_head // n_head_kv

      residual = profile_stage("gguf.eh_proj") { Qwen35CPU.qmatvec_nobias(mtp.nextn_eh_proj_qw, fc_in) }
      cur = profile_stage("gguf.attn_norm") { rms_norm_gguf(residual, mtp.attn_norm, hp.rms_eps) }
      attn_o = Array(Float32).new(q_dim, 0.0_f32)

      if cache = mtp_state
        raise "mtp state kv_dim #{cache.kv_dim} != expected #{kv_dim}" unless cache.kv_dim == kv_dim
        qkv = profile_stage("gguf.attn_qkv") { Qwen35CPU.qmatvec_many([mtp.attn_q_qw, mtp.attn_k_qw, mtp.attn_v_qw], cur) }
        q_full = qkv[0]
        k = qkv[1]
        v = qkv[2]
        raise "mtp q_proj produced #{q_full.size}, expected #{q_dim * 2}" unless q_full.size == q_dim * 2
        raise "mtp k_proj produced #{k.size}, expected #{kv_dim}" unless k.size == kv_dim
        raise "mtp v_proj produced #{v.size}, expected #{kv_dim}" unless v.size == kv_dim

        q = Array(Float32).new(q_dim, 0.0_f32)
        gate = Array(Float32).new(q_dim, 0.0_f32)
        n_head.times do |h|
          src_base = h * 2 * head_dim
          dst_base = h * head_dim
          head_dim.times do |d|
            q[dst_base + d] = q_full[src_base + d]
            gate[dst_base + d] = q_full[src_base + head_dim + d]
          end
        end

        profile_stage("gguf.attn_qk_norm_rope") do
          n_head.times do |h|
            rms_norm_gguf_slice!(q, h * head_dim, head_dim, mtp.attn_q_norm, hp.rms_eps)
            Qwen35CPU.rope_partial!(q, h * head_dim, hp.rope_dim_count, head_dim, pos, hp.rope_freq_base)
          end
          n_head_kv.times do |h|
            rms_norm_gguf_slice!(k, h * head_dim, head_dim, mtp.attn_k_norm, hp.rms_eps)
            Qwen35CPU.rope_partial!(k, h * head_dim, hp.rope_dim_count, head_dim, pos, hp.rope_freq_base)
          end
        end

        profile_stage("gguf.attn_cache_append") do
          cache.append!(k, v)
          nil
        end
        scale = (1.0 / Math.sqrt(head_dim.to_f64)).to_f32
        scores = Array(Float32).new(cache.length, 0.0_f32)
        probs = Array(Float32).new(cache.length, 0.0_f32)

        profile_stage("gguf.attn_softmax_cpu") do
          n_head.times do |h|
            kvh = h // heads_per_group
            q_base = h * head_dim
            max_score = -Float32::INFINITY
            cache.length.times do |t|
              k_base = t * kv_dim + kvh * head_dim
              dot = 0.0_f32
              head_dim.times { |d| dot += q[q_base + d] * cache.k_cache[k_base + d] }
              score = dot * scale
              scores[t] = score
              max_score = score if score > max_score
            end

            denom = 0.0_f32
            cache.length.times do |t|
              p = Math.exp((scores[t] - max_score).to_f64).to_f32
              probs[t] = p
              denom += p
            end
            inv_denom = 1.0_f32 / denom

            head_dim.times do |d|
              sum = 0.0_f32
              cache.length.times do |t|
                v_base = t * kv_dim + kvh * head_dim
                sum += probs[t] * inv_denom * cache.v_cache[v_base + d]
              end
              attn_o[q_base + d] = sum * Qwen35CPU.sigmoid(gate[q_base + d])
            end
          end
        end
      else
        qv = profile_stage("gguf.attn_qv") { Qwen35CPU.qmatvec_many([mtp.attn_q_qw, mtp.attn_v_qw], cur) }
        q_full = qv[0]
        v = qv[1]
        raise "mtp q_proj produced #{q_full.size}, expected #{q_dim * 2}" unless q_full.size == q_dim * 2
        raise "mtp v_proj produced #{v.size}, expected #{kv_dim}" unless v.size == kv_dim

        profile_stage("gguf.attn_one_token_cpu") do
          n_head.times do |h|
            src_base = h * 2 * head_dim + head_dim
            kvh = h // heads_per_group
            q_base = h * head_dim
            kv_base = kvh * head_dim
            head_dim.times do |d|
              attn_o[q_base + d] = v[kv_base + d] * Qwen35CPU.sigmoid(q_full[src_base + d])
            end
          end
        end
      end

      attn_out = profile_stage("gguf.attn_output") { Qwen35CPU.qmatvec_nobias(mtp.attn_output_qw, attn_o) }
      after_attn = profile_stage("gguf.attn_residual") { Array(Float32).new(hidden) { |i| residual[i] + attn_out[i] } }
      cur2 = profile_stage("gguf.post_attn_norm") { rms_norm_gguf(after_attn, mtp.post_attention_norm, hp.rms_eps) }

      if fused = profile_stage("gguf.ffn_head_top2_fused") {
           Qwen35Metal.ffn_project_residual_top2(
             cur2,
             after_attn,
             mtp.ffn_gate_qw,
             mtp.ffn_up_qw,
             mtp.ffn_down_qw,
             mtp.shared_head_norm(weights.output_norm),
             mtp.shared_head_qw(weights.output),
             hp.rms_eps)
         }
        top2_raw = fused[:top2]
        return {hidden: fused[:hidden], top2: [{top2_raw[0].to_i32, top2_raw[1]}, {top2_raw[2].to_i32, top2_raw[3]}]}
      end

      ffn_out = profile_stage("gguf.ffn_project_fused") do
        ffn_project_gguf_metal(cur2, mtp.ffn_gate_qw, mtp.ffn_up_qw, mtp.ffn_down_qw)
      end
      unless ffn_out
        gu = profile_stage("gguf.ffn_upgate") { Qwen35CPU.qmatvec_many([mtp.ffn_gate_qw, mtp.ffn_up_qw], cur2) }
        gate_ff = gu[0]
        up_ff = gu[1]
        combined = profile_stage("gguf.ffn_swiglu_cpu") do
          Qwen35CPU.silu!(gate_ff)
          Array(Float32).new(gate_ff.size) { |i| gate_ff[i] * up_ff[i] }
        end
        ffn_out = profile_stage("gguf.ffn_down") { Qwen35CPU.qmatvec_nobias(mtp.ffn_down_qw, combined) }
      end
      hidden_pre_norm = profile_stage("gguf.ffn_residual") { Array(Float32).new(hidden) { |i| after_attn[i] + ffn_out[i] } }
      {hidden: hidden_pre_norm, top2: hidden_top2_gguf(weights, mtp, hidden_pre_norm)}
    end

    def hidden_top2_gguf(weights : Qwen35Weights,
                         mtp : Qwen35GGUFMTPWeights,
                         hidden_pre_norm : Array(Float32)) : Array({Int32, Float32})
      project_hidden = profile_stage("gguf.head_norm") { rms_norm_gguf(hidden_pre_norm, mtp.shared_head_norm(weights.output_norm), weights.hparams.rms_eps) }
      head = mtp.shared_head_qw(weights.output)
      {% unless flag?(:cpu_only) %}
        if ENV["QWEN35_MTP_TOP2_METAL_OFF"]? != "1" && Qwen35Metal.available?
          if top2 = profile_stage("gguf.head_top2") { Qwen35Metal.project_top2_no_norm(head, project_hidden) }
            return [{top2[0].to_i32, top2[1]}, {top2[2].to_i32, top2[3]}]
          end
        end
      {% end %}

      logits = profile_stage("gguf.head_logits") { Qwen35CPU.qmatvec_nobias(head, project_hidden) }
      profile_stage("gguf.head_topk_cpu") { top_k(logits, 2) }
    end

    def forward_one_logits_gguf(weights : Qwen35Weights,
                                mtp : Qwen35GGUFMTPWeights,
                                prev_hidden : Array(Float32),
                                token_id : Int32,
                                pos : Int32,
                                mtp_state : State? = nil) : Array(Float32)
      hidden = forward_one_hidden_gguf(weights, mtp, prev_hidden, token_id, pos, mtp_state)
      project_hidden = profile_stage("gguf.head_norm") { rms_norm_gguf(hidden, mtp.shared_head_norm(weights.output_norm), weights.hparams.rms_eps) }
      profile_stage("gguf.head_logits") { Qwen35CPU.qmatvec_nobias(mtp.shared_head_qw(weights.output), project_hidden) }
    end

    def forward_one_top1_gguf(weights : Qwen35Weights,
                              mtp : Qwen35GGUFMTPWeights,
                              prev_hidden : Array(Float32),
                              token_id : Int32,
                              pos : Int32,
                              mtp_state : State? = nil) : {Int32, Float32}
      hidden = forward_one_hidden_gguf(weights, mtp, prev_hidden, token_id, pos, mtp_state)
      hidden_top1_gguf(weights, mtp, hidden)
    end

    # CPU/BF16 formula oracle for one Qwen3.6 MTP layer.
    #
    # This follows the Qwen3.5/Qwen3.6 MTP predictor shape:
    #   fc(cat(pre_fc_norm_embedding(embed(input_id)),
    #          pre_fc_norm_hidden(previous_target_hidden)))
    #   -> full-attention decoder layer -> mtp.norm
    #
    # The current baseline covers num_nextn_predict_layers=1 and a single MTP
    # token, where the MTP attention cache length is one. That means softmax is
    # exactly 1 for each query head and the attention value is the current V
    # head after GQA broadcast. This is enough to validate the first acceptance
    # step before building a resident Metal kernel/cache path.
    def forward_one_hidden(weights : Qwen35Weights,
                           mtp : Qwen35MTPWeights,
                           prev_hidden : Array(Float32),
                           token_id : Int32,
                           pos : Int32,
                           normalized : Bool = true,
                           mtp_state : State? = nil) : Array(Float32)
      hp = weights.hparams
      hidden = hp.n_embd
      raise ArgumentError.new("prev_hidden size #{prev_hidden.size} != #{hidden}") unless prev_hidden.size == hidden

      emb = Qwen35CPU.embedding_lookup(weights.token_embd, token_id)
      emb_norm = rms_norm_sidecar(emb, mtp.pre_fc_norm_embedding, hp.rms_eps)
      hidden_norm = rms_norm_sidecar(prev_hidden, mtp.pre_fc_norm_hidden, hp.rms_eps)

      fc_in = Array(Float32).new(hidden * 2, 0.0_f32)
      hidden.times do |i|
        fc_in[i] = emb_norm[i]
        fc_in[hidden + i] = hidden_norm[i]
      end

      n_head = hp.n_head
      n_head_kv = hp.n_head_kv
      head_dim = hp.head_dim
      q_dim = n_head * head_dim
      kv_dim = n_head_kv * head_dim
      heads_per_group = n_head // n_head_kv

      {% if flag?(:qwen35_mtp_metal) %}
        if ENV["QWEN35_MTP_BODY_METAL"]? == "1" &&
           ENV["QWEN35_MTP_ONE_TOKEN_SHORTCUT_OFF"]? != "1" &&
           normalized &&
           mtp_state.nil? &&
           use_metal_bf16?(mtp.fc)
          if body = Qwen35Metal.mtp_one_token_hidden_from_fc_in(
               fc_in,
               mtp.fc.raw,
               mtp.v_proj.raw,
               mtp.q_proj.raw,
               mtp.o_proj.raw,
               mtp.ffn_gate.raw,
               mtp.ffn_up.raw,
               mtp.ffn_down.raw,
               sidecar_norm_weight(mtp.input_layernorm),
               sidecar_norm_weight(mtp.post_attention_layernorm),
               sidecar_norm_weight(mtp.norm),
               hidden,
               q_dim,
               kv_dim,
               hp.n_ff,
               n_head,
               n_head_kv,
               head_dim,
               hp.rms_eps,
             )
            return body
          end
        end
      {% end %}

      residual = matvec_bf16(mtp.fc, fc_in)
      cur = rms_norm_sidecar(residual, mtp.input_layernorm, hp.rms_eps)

      v = matvec_bf16(mtp.v_proj, cur)
      raise "mtp v_proj produced #{v.size}, expected #{kv_dim}" unless v.size == kv_dim

      attn_o = Array(Float32).new(q_dim, 0.0_f32)

      if cache = mtp_state
        raise "mtp state kv_dim #{cache.kv_dim} != expected #{kv_dim}" unless cache.kv_dim == kv_dim

        q_full = matvec_bf16(mtp.q_proj, cur)
        k = matvec_bf16(mtp.k_proj, cur)
        raise "mtp q_proj produced #{q_full.size}, expected #{q_dim * 2}" unless q_full.size == q_dim * 2
        raise "mtp k_proj produced #{k.size}, expected #{kv_dim}" unless k.size == kv_dim

        q = Array(Float32).new(q_dim, 0.0_f32)
        gate = Array(Float32).new(q_dim, 0.0_f32)
        n_head.times do |h|
          src_base = h * 2 * head_dim
          dst_base = h * head_dim
          head_dim.times do |d|
            q[dst_base + d] = q_full[src_base + d]
            gate[dst_base + d] = q_full[src_base + head_dim + d]
          end
        end

        n_head.times do |h|
          rms_norm_sidecar_slice!(q, h * head_dim, head_dim, mtp.q_norm, hp.rms_eps)
          Qwen35CPU.rope_partial!(q, h * head_dim, hp.rope_dim_count, head_dim, pos, hp.rope_freq_base)
        end
        n_head_kv.times do |h|
          rms_norm_sidecar_slice!(k, h * head_dim, head_dim, mtp.k_norm, hp.rms_eps)
          Qwen35CPU.rope_partial!(k, h * head_dim, hp.rope_dim_count, head_dim, pos, hp.rope_freq_base)
        end

        cache.append!(k, v)
        scale = (1.0 / Math.sqrt(head_dim.to_f64)).to_f32
        scores = Array(Float32).new(cache.length, 0.0_f32)
        probs = Array(Float32).new(cache.length, 0.0_f32)

        n_head.times do |h|
          kvh = h // heads_per_group
          q_base = h * head_dim
          max_score = -Float32::INFINITY
          cache.length.times do |t|
            k_base = t * kv_dim + kvh * head_dim
            dot = 0.0_f32
            head_dim.times { |d| dot += q[q_base + d] * cache.k_cache[k_base + d] }
            score = dot * scale
            scores[t] = score
            max_score = score if score > max_score
          end

          denom = 0.0_f32
          cache.length.times do |t|
            p = Math.exp((scores[t] - max_score).to_f64).to_f32
            probs[t] = p
            denom += p
          end
          inv_denom = 1.0_f32 / denom

          head_dim.times do |d|
            sum = 0.0_f32
            cache.length.times do |t|
              v_base = t * kv_dim + kvh * head_dim
              sum += probs[t] * inv_denom * cache.v_cache[v_base + d]
            end
            attn_o[q_base + d] = sum * Qwen35CPU.sigmoid(gate[q_base + d])
          end
        end
      else
        gate = if ENV["QWEN35_MTP_ONE_TOKEN_SHORTCUT_OFF"]? == "1"
                 # Full formula path for adversary A/B. For a one-token MTP cache,
                 # Q/K and RoPE do not affect softmax, but keeping this branch
                 # makes it easy to catch future multi-token misuse.
                 q_full = matvec_bf16(mtp.q_proj, cur)
                 k = matvec_bf16(mtp.k_proj, cur)
                 raise "mtp q_proj produced #{q_full.size}, expected #{q_dim * 2}" unless q_full.size == q_dim * 2
                 raise "mtp k_proj produced #{k.size}, expected #{kv_dim}" unless k.size == kv_dim

                 q = Array(Float32).new(q_dim, 0.0_f32)
                 gate_full = Array(Float32).new(q_dim, 0.0_f32)
                 n_head.times do |h|
                   src_base = h * 2 * head_dim
                   dst_base = h * head_dim
                   head_dim.times do |d|
                     q[dst_base + d] = q_full[src_base + d]
                     gate_full[dst_base + d] = q_full[src_base + head_dim + d]
                   end
                 end

                 n_head.times do |h|
                   rms_norm_sidecar_slice!(q, h * head_dim, head_dim, mtp.q_norm, hp.rms_eps)
                   Qwen35CPU.rope_partial!(q, h * head_dim, hp.rope_dim_count, head_dim, pos, hp.rope_freq_base)
                 end
                 n_head_kv.times do |h|
                   rms_norm_sidecar_slice!(k, h * head_dim, head_dim, mtp.k_norm, hp.rms_eps)
                   Qwen35CPU.rope_partial!(k, h * head_dim, hp.rope_dim_count, head_dim, pos, hp.rope_freq_base)
                 end
                 gate_full
               else
                 # Exact one-token shortcut: with only the current MTP token in
                 # the MTP KV-cache, attention softmax is 1. Q/K/RoPE cannot
                 # change the output; only V and the output gate are needed.
                 matvec_bf16_q_gate_rows(mtp.q_proj, cur, n_head, head_dim)
               end

        raise "mtp gate produced #{gate.size}, expected #{q_dim}" unless gate.size == q_dim

        n_head.times do |h|
          kvh = h // heads_per_group
          q_base = h * head_dim
          kv_base = kvh * head_dim
          head_dim.times do |d|
            attn_o[q_base + d] = v[kv_base + d] * Qwen35CPU.sigmoid(gate[q_base + d])
          end
        end
      end

      attn_out = matvec_bf16(mtp.o_proj, attn_o)
      after_attn = Array(Float32).new(hidden) { |i| residual[i] + attn_out[i] }
      cur2 = rms_norm_sidecar(after_attn, mtp.post_attention_layernorm, hp.rms_eps)

      gate_ff = matvec_bf16(mtp.ffn_gate, cur2)
      up_ff = matvec_bf16(mtp.ffn_up, cur2)
      Qwen35CPU.silu!(gate_ff)
      combined = Array(Float32).new(gate_ff.size) { |i| gate_ff[i] * up_ff[i] }
      ffn_out = matvec_bf16(mtp.ffn_down, combined)
      after_ffn = Array(Float32).new(hidden) { |i| after_attn[i] + ffn_out[i] }

      return after_ffn unless normalized

      rms_norm_sidecar(after_ffn, mtp.norm, hp.rms_eps)
    end

    def forward_one_logits(weights : Qwen35Weights,
                           mtp : Qwen35MTPWeights,
                           prev_hidden : Array(Float32),
                           token_id : Int32,
                           pos : Int32) : Array(Float32)
      hidden = forward_one_hidden(weights, mtp, prev_hidden, token_id, pos)
      Qwen35CPU.qmatvec_nobias(weights.output, hidden)
    end

    # Project an already MTP-normalized hidden through the shared output head.
    # Unlike target hidden projection, MTP hidden has already passed `mtp.norm`,
    # so applying the target output RMSNorm here would double-normalize it.
    def hidden_top1(weights : Qwen35Weights, hidden : Array(Float32)) : {Int32, Float32}
      {% if flag?(:qwen35_mtp_metal) %}
        if ENV["QWEN35_MTP_TOP1_METAL_OFF"]? != "1" && Qwen35Metal.available?
          if top1 = Qwen35Metal.project_top1_no_norm(weights.output, hidden)
            return {top1[0].to_i32, top1[1]}
          end
        end
      {% end %}

      logits = Qwen35CPU.qmatvec_nobias(weights.output, hidden)
      top_k(logits, 1)[0]
    end

    def hidden_top2(weights : Qwen35Weights, hidden : Array(Float32)) : Array({Int32, Float32})
      {% if flag?(:qwen35_mtp_metal) %}
        if ENV["QWEN35_MTP_TOP2_METAL_OFF"]? != "1" && Qwen35Metal.available?
          if top2 = Qwen35Metal.project_top2_no_norm(weights.output, hidden)
            return [{top2[0].to_i32, top2[1]}, {top2[2].to_i32, top2[3]}]
          end
        end
      {% end %}

      logits = Qwen35CPU.qmatvec_nobias(weights.output, hidden)
      top_k(logits, 2)
    end

    def top_k(logits : Array(Float32), k : Int32) : Array({Int32, Float32})
      raise ArgumentError.new("top_k must be positive") unless k > 0
      best = [] of {Int32, Float32}
      logits.each_with_index do |v, id|
        id32 = id.to_i32
        if best.size < k || v > best[-1][1] || (v == best[-1][1] && id32 < best[-1][0])
          best << {id32, v}
          best.sort! do |a, b|
            cmp = b[1] <=> a[1]
            cmp == 0 ? (a[0] <=> b[0]) : cmp
          end
          best.pop if best.size > k
        end
      end
      best
    end

    def forward_one_top1(weights : Qwen35Weights,
                         mtp : Qwen35MTPWeights,
                         prev_hidden : Array(Float32),
                         token_id : Int32,
                         pos : Int32) : {Int32, Float32}
      hidden = forward_one_hidden(weights, mtp, prev_hidden, token_id, pos)
      hidden_top1(weights, hidden)
    end
  end
end
