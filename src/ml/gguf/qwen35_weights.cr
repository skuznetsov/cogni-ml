require "./reader"
require "./compute" # for QuantWeight
require "./qwen35_meta"
require "./qwen35_mmap_views"
{% unless flag?(:cpu_only) %}
  require "./qwen35_metal"
{% end %}

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
    getter gguf_file_type : Int64?

    # Kept alive so the mmap region backing every QuantWeight.raw stays
    # mapped until this weight set is explicitly closed. Closing it invalidates
    # both the heap-free slices and any whole-mmap Metal buffer built on top of
    # them, so callers must quiesce all inference first.
    @gguf : GGUFFile
    @closed : Bool
    @close_mutex : Mutex
    @q4_gemv_x16_capability : Q4GemvX16Capability
    {% unless flag?(:cpu_only) %}
      @mmap_base : Pointer(UInt8)?
    {% end %}

    def initialize(@gguf : GGUFFile, @hparams : Qwen35Hparams)
      @closed = false
      @close_mutex = Mutex.new
      @gguf_file_type = @gguf.get_int("general.file_type")
      @q4_gemv_x16_capability = self.class.q4_gemv_x16_capability_for(
        @gguf.get_string("general.name"),
        @gguf.get_string("general.basename")
      )
      coarse_mmap_views = self.class.coarse_mmap_views_enabled?
      {% if flag?(:cpu_only) %}
        if coarse_mmap_views
          raise "QWEN35_COARSE_WEIGHT_VIEWS=1 requires a Metal-enabled build"
        end
      {% end %}
      {% unless flag?(:cpu_only) %}
        @mmap_base = nil
      {% end %}
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

      # Register mmap-backed quantized weights as zero-copy Metal buffers.
      # The established path keeps one whole-file view. The opt-in coarse path
      # uses dense command-group views that remain alive for the model lifetime,
      # avoiding both the whole-file first-touch and unsafe wrapper rotation.
      {% unless flag?(:cpu_only) %}
        metal_available = Qwen35Metal.available?
        if coarse_mmap_views && !metal_available
          raise "QWEN35_COARSE_WEIGHT_VIEWS=1 requires an available Metal device"
        end
        if metal_available
          if region = @gguf.mmap_region
            base, size = region
            if coarse_mmap_views
              groups = build_mmap_weight_groups(base, size)
              views = groups.flat_map(&.views).uniq { |view| {view.address, view.length} }
              validate_mmap_weight_coverage!(views)
              registered = false
              begin
                Qwen35Metal.register_mmap_views(base, size, views, strict: true)
                registered = true
                @mmap_base = base
                if ENV["QWEN35_COARSE_WEIGHT_VIEW_TRACE"]? == "1"
                  stats = Qwen35Metal.mmap_registration_stats
                  STDERR.puts(
                    "qwen35_coarse_weight_views groups=#{groups.size} " \
                    "views=#{stats[:views]} view_bytes=#{stats[:view_bytes]} " \
                    "unique_view_bytes=#{stats[:unique_view_bytes]} " \
                    "owner_bytes=#{stats[:owner_bytes]} strict=#{stats[:strict]}",
                  )
                end
              rescue ex
                Qwen35Metal.unregister_mmap(base) if registered
                @mmap_base = nil
                raise ex
              end
            else
              Qwen35Metal.register_mmap(base, size)
              @mmap_base = base
            end
          elsif coarse_mmap_views
            raise "QWEN35_COARSE_WEIGHT_VIEWS=1 requires mmap-backed weights"
          end
        end
      {% end %}
    end

    # Performance defaults are issued only to the exact GGUF model identity
    # measured by the corresponding gate. Unknown converters and future models
    # fail closed to the established kernels while explicit env experiments
    # remain available.
    def self.q4_gemv_x16_capability_for(model_name : String?, model_basename : String?) : Q4GemvX16Capability
      if model_name == "Qwen_Qwen3.8 27B" && model_basename == "Qwen_Qwen3.8"
        Q4GemvX16Capability::Qwen38
      else
        Q4GemvX16Capability::Unknown
      end
    end

    def self.from_gguf(path : String) : Qwen35Weights
      g = GGUFFile.new(path)
      begin
        hp = Qwen35Hparams.new(g)
        # Do NOT close `g` on success — the mmap backs every QuantWeight.raw;
        # Qwen35Weights keeps a reference to it for its lifetime.
        Qwen35Weights.new(g, hp)
      rescue ex
        g.close
        raise ex
      end
    end

    # Release any process-global no-copy Metal wrapper before unmapping the
    # GGUF file that backs its bytes. Callers must quiesce in-flight inference,
    # including explicit/lane Metal queues, before closing a weight set; the
    # wrapper cannot protect concurrent users after this method returns.
    def close : Nil
      @close_mutex.synchronize do
        return if @closed

        {% unless flag?(:cpu_only) %}
          if base = @mmap_base
            Qwen35Metal.unregister_mmap(base)
            @mmap_base = nil
          end
        {% end %}
        @gguf.close
        @closed = true
      end
    end

    def finalize
      close
    end

    def self.coarse_mmap_views_enabled?(
      configured : String? = ENV["QWEN35_COARSE_WEIGHT_VIEWS"]?,
    ) : Bool
      case configured
      when nil, "0" then false
      when "1"      then true
      else
        raise ArgumentError.new("QWEN35_COARSE_WEIGHT_VIEWS must be 0 or 1")
      end
    end

    private def build_mmap_weight_groups(base : Pointer(UInt8), size : UInt64) : Array(Qwen35MmapWeightGroup)
      full_layers = @hparams.full_attention_layers
      raise "coarse mmap views require at least one full-attention layer" if full_layers.empty?

      ranges = [] of {Int32, Int32}
      if full_layers.size == 1
        ranges << {0_i32, (@layers.size - 1).to_i32}
      else
        ranges << {0_i32, (full_layers[1] - 1).to_i32}
        (1...full_layers.size).each do |index|
          first_layer = full_layers[index]
          last_layer = index + 1 < full_layers.size ? full_layers[index + 1] - 1 : @layers.size - 1
          ranges << {first_layer.to_i32, last_layer.to_i32}
        end
      end

      groups = ranges.map_with_index do |range, index|
        first_layer, last_layer = range
        weights = [] of QuantWeight
        weights << @token_embd if index == 0
        (first_layer..last_layer).each do |layer|
          weights.concat(quant_weights_for_layer(@layers[layer]))
        end
        view = Qwen35MmapWeightView.for_spans(
          base.address,
          size,
          weights.map { |weight| mmap_span_for(weight) },
        )
        Qwen35MmapWeightGroup.new(first_layer, last_layer, [view])
      end

      unless @output.same?(@token_embd)
        output_view = Qwen35MmapWeightView.for_spans(
          base.address,
          size,
          [mmap_span_for(@output)],
        )
        last = groups.last
        groups[-1] = Qwen35MmapWeightGroup.new(
          last.first_layer,
          last.last_layer,
          last.views + [output_view],
        )
      end
      groups
    end

    private def validate_mmap_weight_coverage!(views : Array(Qwen35MmapWeightView)) : Nil
      weights = [@token_embd, @output]
      @layers.each { |layer| weights.concat(quant_weights_for_layer(layer)) }
      weights.uniq(&.object_id).each do |weight|
        span = mmap_span_for(weight)
        unless views.any?(&.contains?(span))
          raise "coarse mmap views do not cover #{weight.route_tag || weight.object_id}"
        end
      end
    end

    private def mmap_span_for(weight : QuantWeight) : Qwen35MmapWeightSpan
      Qwen35MmapWeightSpan.new(weight.raw.to_unsafe.address, weight.raw.size.to_i64)
    end

    private def quant_weights_for_layer(layer : Qwen35LayerWeights) : Array(QuantWeight)
      case layer
      in Qwen35FullAttnWeights
        [
          layer.attn_q_qw,
          layer.attn_k_qw,
          layer.attn_v_qw,
          layer.attn_output_qw,
          layer.ffn_gate_qw,
          layer.ffn_up_qw,
          layer.ffn_down_qw,
        ]
      in Qwen35RecurrentWeights
        [
          layer.attn_qkv_qw,
          layer.attn_gate_qw,
          layer.ssm_alpha_qw,
          layer.ssm_beta_qw,
          layer.ssm_out_qw,
          layer.ffn_gate_qw,
          layer.ffn_up_qw,
          layer.ffn_down_qw,
        ]
      end
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
      QuantWeight.new(raw, info.type, out_dim, in_dim,
        route_tag: "qwen35:#{name}",
        q4_gemv_x16_capability: @q4_gemv_x16_capability)
    end

    private def load_f32(g : GGUFFile, name : String) : Array(Float32)
      info = g.tensor(name) || raise "qwen35_weights: missing tensor #{name.inspect}"
      g.read_tensor_f32(info)
    end
  end
end
