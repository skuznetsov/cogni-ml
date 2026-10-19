require "./metal_ops"
require "./checkpoint"

module ML::Video
  # Copyright 2024-2025 The Alibaba Wan Team Authors.
  # Apache-2.0. Adapted for cogni-ml from Wan2.2 S2V at
  # 1ea34ff48f87168174e12956e200b1d908b1c5ff.
  private module WanS2VShape
    def self.count(label : String, *factors : Int32) : Int32
      result = 1_i64
      factors.each do |factor|
        unless factor > 0 && result <= Int32::MAX.to_i64 // factor
          raise ArgumentError.new("Wan S2V #{label} exceeds supported element count")
        end
        result *= factor
      end
      result.to_i32
    end
  end

  # Tiny and production shapes share the exact Wan S2V block topology. `split`
  # is the e[1] segment boundary from upstream; positions are [B,S,(t,h,w)].
  struct WanS2VConfig
    getter batches : Int32
    getter seq : Int32
    getter context_tokens : Int32
    getter dim : Int32
    getter heads : Int32
    getter ffn_dim : Int32
    getter split : Int32
    getter theta : Float32
    getter video_frames : Int32
    getter spatial_tokens : Int32
    getter head_dim : Int32
    getter rope_axis_dims : {Int32, Int32, Int32}

    def initialize(@batches : Int32, @seq : Int32, @context_tokens : Int32,
                   @dim : Int32, @heads : Int32, @ffn_dim : Int32,
                   @split : Int32, @theta : Float32, @video_frames : Int32,
                   @spatial_tokens : Int32)
      unless @batches > 0 && @seq > 0 && @context_tokens > 0 && @dim > 0 &&
             @heads > 0 && @ffn_dim > 0 && @video_frames > 0 && @spatial_tokens > 0
        raise ArgumentError.new("Wan S2V dimensions must be positive")
      end
      raise ArgumentError.new("Wan S2V dim must divide evenly across heads") unless @dim % @heads == 0
      @head_dim = @dim // @heads
      raise ArgumentError.new("Wan S2V head dimension must be even and <= 256") if @head_dim < 6 || @head_dim > 256 || @head_dim.odd?
      raise ArgumentError.new("Wan S2V RoPE theta must be finite and positive") unless @theta.finite? && @theta > 0
      raise ArgumentError.new("Wan S2V split must be inside the sequence") unless @split > 0 && @split < @seq
      WanS2VShape.count("hidden state", @batches, @seq, @dim)
      WanS2VShape.count("context", @batches, @context_tokens, @dim)
      WanS2VShape.count("modulation", @batches, 12, @dim)
      WanS2VShape.count("positions", @batches, @seq, 3)
      WanS2VShape.count("feed-forward activations", @batches, @seq, @ffn_dim)
      WanS2VShape.count("projection weights", @dim, @dim)
      WanS2VShape.count("feed-forward weights", @dim, @ffn_dim)
      raise ArgumentError.new("Video prefix exceeds the Wan sequence") if video_tokens > @seq
      pairs = @head_dim // 2
      thirds = pairs // 3
      @rope_axis_dims = {pairs - 2 * thirds, thirds, thirds}
    end

    def video_tokens : Int32
      @video_frames * @spatial_tokens
    end

    def rows : Int32
      @batches * @seq
    end

    def context_rows : Int32
      @batches * @context_tokens
    end

    def values : Int32
      rows * @dim
    end
  end

  # Audio encoder outputs are external resident buffers. Local audio is exactly
  # five tokens per frame: four MotionEncoder tokens plus upstream's learned pad.
  struct WanS2VAudioConfig
    getter batches : Int32
    getter seq : Int32
    getter dim : Int32
    getter heads : Int32
    getter video_frames : Int32
    getter spatial_tokens : Int32
    getter audio_tokens : Int32
    getter head_dim : Int32

    def initialize(@batches : Int32, @seq : Int32, @dim : Int32,
                   @heads : Int32, @video_frames : Int32,
                   @spatial_tokens : Int32, @audio_tokens : Int32)
      unless @batches > 0 && @seq > 0 && @dim > 0 && @heads > 0 &&
             @video_frames > 0 && @spatial_tokens > 0
        raise ArgumentError.new("Wan S2V audio dimensions must be positive")
      end
      raise ArgumentError.new("Wan S2V audio dim must divide evenly across heads") unless @dim % @heads == 0
      @head_dim = @dim // @heads
      raise ArgumentError.new("Wan S2V audio head dimension must be even and <= 256") if @head_dim < 2 || @head_dim > 256 || @head_dim.odd?
      raise ArgumentError.new("Wan S2V uses four local audio tokens plus one learned pad") unless @audio_tokens == 5
      WanS2VShape.count("audio hidden state", @batches, @seq, @dim)
      WanS2VShape.count("audio video prefix", @batches, @video_frames, @spatial_tokens, @dim)
      WanS2VShape.count("local audio embeddings", @batches, @video_frames, @audio_tokens, @dim)
      WanS2VShape.count("global audio embeddings", @batches, @video_frames, @dim)
      WanS2VShape.count("AdaLayerNorm weights", @dim, 2, @dim)
      WanS2VShape.count("audio projections", @dim, @dim)
      raise ArgumentError.new("Audio video prefix exceeds the Wan sequence") if video_tokens > @seq
    end

    def video_tokens : Int32
      @video_frames * @spatial_tokens
    end

    def gathered_rows : Int32
      @batches * video_tokens
    end

    def audio_rows : Int32
      @batches * @video_frames * @audio_tokens
    end

    def frame_rows : Int32
      @batches * @video_frames
    end

    def values : Int32
      @batches * @seq * @dim
    end
  end

  module WanS2VBlock
    SOURCE = {{ read_file("#{__DIR__}/kernels/wan_s2v.metal") }}
    EPS    = 1e-6_f32

    # Map the upstream WanModel_S2V block-0 and audio-injector state-dict keys
    # to the Arena names consumed by encode_with_audio. Shapes preserve the
    # PyTorch state-dict layout; Metal linear weights remain [out, in].
    def self.checkpoint_bindings(cfg : WanS2VConfig,
                                 dtype : ML::GGUF::SafeTensorDType? = nil) : Array(WeightBinding)
      dim = cfg.dim.to_i64
      ffn = cfg.ffn_dim.to_i64
      bindings = [] of WeightBinding
      bindings << WeightBinding.new("block_modulation", "blocks.0.modulation", [1_i64, 6_i64, dim])
      bindings << WeightBinding.new("norm3_weight", "blocks.0.norm3.weight", [dim])
      bindings << WeightBinding.new("norm3_bias", "blocks.0.norm3.bias", [dim])
      {"self", "cross"}.each do |prefix|
        source_prefix = "blocks.0.#{prefix}_attn"
        {"q", "k", "v", "o"}.each do |name|
          bindings << WeightBinding.new("#{prefix}_#{name}_weight", "#{source_prefix}.#{name}.weight", [dim, dim])
          bindings << WeightBinding.new("#{prefix}_#{name}_bias", "#{source_prefix}.#{name}.bias", [dim])
        end
        bindings << WeightBinding.new("#{prefix}_norm_q_weight", "#{source_prefix}.norm_q.weight", [dim])
        bindings << WeightBinding.new("#{prefix}_norm_k_weight", "#{source_prefix}.norm_k.weight", [dim])
      end
      bindings << WeightBinding.new("ffn_in_weight", "blocks.0.ffn.0.weight", [ffn, dim])
      bindings << WeightBinding.new("ffn_in_bias", "blocks.0.ffn.0.bias", [ffn])
      bindings << WeightBinding.new("ffn_out_weight", "blocks.0.ffn.2.weight", [dim, ffn])
      bindings << WeightBinding.new("ffn_out_bias", "blocks.0.ffn.2.bias", [dim])
      bindings << WeightBinding.new("adain_weight", "audio_injector.injector_adain_layers.0.linear.weight", [2_i64 * dim, dim])
      bindings << WeightBinding.new("adain_bias", "audio_injector.injector_adain_layers.0.linear.bias", [2_i64 * dim])
      {"q", "k", "v", "o"}.each do |name|
        source_prefix = "audio_injector.injector.0"
        bindings << WeightBinding.new("audio_#{name}_weight", "#{source_prefix}.#{name}.weight", [dim, dim])
        bindings << WeightBinding.new("audio_#{name}_bias", "#{source_prefix}.#{name}.bias", [dim])
      end
      bindings << WeightBinding.new("audio_q_norm_weight", "audio_injector.injector.0.norm_q.weight", [dim])
      bindings << WeightBinding.new("audio_k_norm_weight", "audio_injector.injector.0.norm_k.weight", [dim])
      if expected_dtype = dtype
        bindings.map do |binding|
          WeightBinding.new(binding.target, binding.source, binding.shape, expected_dtype)
        end
      else
        bindings
      end
    end

    private def self.pipeline(name : String) : ML::Metal::ComputePipeline
      ML::Metal::PipelineCache.get(name) { ML::Metal::ComputePipeline.new(name, SOURCE) }
    end

    # Call during model setup. Compilation is intentionally kept out of encode.
    def self.prepare! : Nil
      MetalOps.prepare!
      {"wan_modulate_segments", "wan_segment_add", "wan_rope_f32"}.each do |name|
        pipeline(name)
      end
      {"video_linear_f32", "video_norm_f32", "video_modulate_f32", "video_add_f32",
       "video_gelu_f32", "video_attention_f32", "video_copy_f32"}.each do |name|
        MetalOps.pipeline(name)
      end
    end

    # Allocate every temporary before command encoding. The same Arena can be
    # reused across serial invocations after the previous command has completed.
    def self.prepare_scratch!(arena : Arena, cfg : WanS2VConfig) : Nil
      n = cfg.values
      ff = cfg.rows * cfg.ffn_dim
      ctx = cfg.context_rows
      arena.upload("wan_b_zero_bias", Array(Float32).new(cfg.dim, 0.0_f32))
      frequency_hi, frequency_lo = rope_frequencies(cfg)
      frequency_hi_name, frequency_lo_name = rope_frequency_names(cfg)
      arena.upload(frequency_hi_name, frequency_hi)
      arena.upload(frequency_lo_name, frequency_lo)
      {
        {"wan_b_norm1", n}, {"wan_b_mod1", n},
        {"wan_b_self_q", n}, {"wan_b_self_k", n}, {"wan_b_self_v", n},
        {"wan_b_self_qn", n}, {"wan_b_self_kn", n}, {"wan_b_self_qr", n}, {"wan_b_self_kr", n},
        {"wan_b_self_attn", n}, {"wan_b_self_proj", n}, {"wan_b_self_res", n},
        {"wan_b_cross_norm", n}, {"wan_b_cross_q", n}, {"wan_b_cross_k", ctx * cfg.dim},
        {"wan_b_cross_v", ctx * cfg.dim}, {"wan_b_cross_qn", n},
        {"wan_b_cross_kn", ctx * cfg.dim}, {"wan_b_cross_attn", n},
        {"wan_b_cross_proj", n}, {"wan_b_cross_res", n},
        {"wan_b_ffn_norm", n}, {"wan_b_ffn_mod", n}, {"wan_b_ffn_in", ff},
        {"wan_b_ffn_gelu", ff}, {"wan_b_ffn_out", n},
      }.each { |name, count| arena.alloc(name, count) }
    end

    # Checkpoints are optional preallocated destinations. Accepted names are
    # documented here so the oracle and trace contract remain stable.
    CHECKPOINTS = {
      "norm1" => :n, "self_mod" => :n,
      "self_q" => :n, "self_k" => :n, "self_v" => :n,
      "self_q_norm" => :n,
      "self_q_rope" => :n, "self_k_rope" => :n,
      "self_attention" => :n, "self_projection" => :n, "self_residual" => :n,
      "cross_norm" => :n, "cross_q" => :n, "cross_k" => :ctx,
      "cross_v" => :ctx, "cross_attention" => :n, "cross_projection" => :n,
      "cross_residual" => :n, "ffn_norm" => :n, "ffn_mod" => :n,
      "ffn_in" => :ff, "ffn_gelu" => :ff, "ffn_out" => :n, "output" => :n,
    }

    # Encode one upstream WanS2VAttentionBlock. Inputs, activations and scratch
    # are contiguous F32 Metal buffers; learned parameters may be F32 or BF16.
    # `modulation` is [B,6,2,C], `positions` is
    # integer-valued F32 [B,S,3], and context is [B,L,C]. No host math, buffer
    # creation, upload, commit, wait, or read occurs in this method.
    def self.encode(e : ML::Metal::ComputeEncoder, x : ML::MetalBuffer,
                    modulation : ML::MetalBuffer, positions : ML::MetalBuffer,
                    context : ML::MetalBuffer, y : ML::MetalBuffer,
                    weights : Arena, scratch : Arena, cfg : WanS2VConfig,
                    trace : Hash(String, ML::MetalBuffer)? = nil) : Nil
      n = cfg.values.to_i64
      rows = cfg.rows
      ctx_rows = cfg.context_rows
      dim = cfg.dim
      ff_count = (rows.to_i64 * cfg.ffn_dim)
      validate_inputs!(x, modulation, positions, context, y, weights, scratch, cfg)
      validate_trace!(trace, x, modulation, positions, context, y, weights, scratch, cfg)

      norm1 = scratch["wan_b_norm1"]
      mod1 = scratch["wan_b_mod1"]
      self_q = scratch["wan_b_self_q"]
      self_k = scratch["wan_b_self_k"]
      self_v = scratch["wan_b_self_v"]
      self_qn = scratch["wan_b_self_qn"]
      self_kn = scratch["wan_b_self_kn"]
      self_qr = scratch["wan_b_self_qr"]
      self_kr = scratch["wan_b_self_kr"]
      self_attn = scratch["wan_b_self_attn"]
      self_proj = scratch["wan_b_self_proj"]
      self_res = scratch["wan_b_self_res"]
      cross_norm = scratch["wan_b_cross_norm"]
      cross_q = scratch["wan_b_cross_q"]
      cross_k = scratch["wan_b_cross_k"]
      cross_v = scratch["wan_b_cross_v"]
      cross_qn = scratch["wan_b_cross_qn"]
      cross_kn = scratch["wan_b_cross_kn"]
      cross_attn = scratch["wan_b_cross_attn"]
      cross_proj = scratch["wan_b_cross_proj"]
      cross_res = scratch["wan_b_cross_res"]
      ffn_norm = scratch["wan_b_ffn_norm"]
      ffn_mod = scratch["wan_b_ffn_mod"]
      ffn_in = scratch["wan_b_ffn_in"]
      ffn_gelu = scratch["wan_b_ffn_gelu"]
      ffn_out = scratch["wan_b_ffn_out"]

      # self-attention branch (norm1 is non-affine in upstream)
      MetalOps.norm(e, x, weights["norm3_weight"], weights["norm3_bias"], norm1,
        rows, dim, EPS, affine: false)
      trace_copy(e, trace, "norm1", norm1, n)
      encode_modulate(e, norm1, modulation, weights["block_modulation"], mod1,
        cfg, 0, 1, weights.dtype("block_modulation"))
      trace_copy(e, trace, "self_mod", mod1, n)
      encode_qkv(e, mod1, self_q, self_k, self_v, weights, rows, dim)
      trace_copy(e, trace, "self_q", self_q, n)
      trace_copy(e, trace, "self_k", self_k, n)
      trace_copy(e, trace, "self_v", self_v, n)
      norm_rms(e, self_q, weights["self_norm_q_weight"], weights.dtype("self_norm_q_weight"),
        scratch["wan_b_zero_bias"], self_qn, rows, dim)
      trace_copy(e, trace, "self_q_norm", self_qn, n)
      norm_rms(e, self_k, weights["self_norm_k_weight"], weights.dtype("self_norm_k_weight"),
        scratch["wan_b_zero_bias"], self_kn, rows, dim)
      encode_rope(e, self_qn, positions, self_qr, cfg, scratch)
      encode_rope(e, self_kn, positions, self_kr, cfg, scratch)
      trace_copy(e, trace, "self_q_rope", self_qr, n)
      trace_copy(e, trace, "self_k_rope", self_kr, n)
      MetalOps.attention(e, self_qr, self_kr, self_v, self_attn,
        cfg.seq, cfg.seq, cfg.heads, cfg.head_dim, cfg.batches)
      trace_copy(e, trace, "self_attention", self_attn, n)
      linear(e, self_attn, weights, "self_o_weight", "self_o_bias", self_proj, rows, dim, dim)
      trace_copy(e, trace, "self_projection", self_proj, n)
      encode_gated_residual(e, x, self_proj, modulation, weights["block_modulation"], self_res,
        cfg, 2, weights.dtype("block_modulation"))
      trace_copy(e, trace, "self_residual", self_res, n)

      # text/context cross attention (norm3 is affine; no RoPE)
      MetalOps.norm(e, self_res, weights["norm3_weight"], weights["norm3_bias"], cross_norm,
        rows, dim, EPS, affine: true, weight_dtype: weights.dtype("norm3_weight"),
        bias_dtype: weights.dtype("norm3_bias"))
      trace_copy(e, trace, "cross_norm", cross_norm, n)
      linear(e, cross_norm, weights, "cross_q_weight", "cross_q_bias", cross_q, rows, dim, dim)
      linear(e, context, weights, "cross_k_weight", "cross_k_bias", cross_k, ctx_rows, dim, dim)
      linear(e, context, weights, "cross_v_weight", "cross_v_bias", cross_v, ctx_rows, dim, dim)
      trace_copy(e, trace, "cross_q", cross_q, n)
      trace_copy(e, trace, "cross_k", cross_k, ctx_rows.to_i64 * dim)
      trace_copy(e, trace, "cross_v", cross_v, ctx_rows.to_i64 * dim)
      norm_rms(e, cross_q, weights["cross_norm_q_weight"], weights.dtype("cross_norm_q_weight"),
        scratch["wan_b_zero_bias"], cross_qn, rows, dim)
      norm_rms(e, cross_k, weights["cross_norm_k_weight"], weights.dtype("cross_norm_k_weight"),
        scratch["wan_b_zero_bias"], cross_kn, ctx_rows, dim)
      MetalOps.attention(e, cross_qn, cross_kn, cross_v, cross_attn,
        cfg.seq, cfg.context_tokens, cfg.heads, cfg.head_dim, cfg.batches)
      trace_copy(e, trace, "cross_attention", cross_attn, n)
      linear(e, cross_attn, weights, "cross_o_weight", "cross_o_bias", cross_proj, rows, dim, dim)
      trace_copy(e, trace, "cross_projection", cross_proj, n)
      MetalOps.add(e, self_res, cross_proj, cross_res, n.to_i32)
      trace_copy(e, trace, "cross_residual", cross_res, n)

      # feed-forward branch: norm2, segmented AdaLN, tanh GELU, gated residual
      MetalOps.norm(e, cross_res, weights["norm3_weight"], weights["norm3_bias"], ffn_norm,
        rows, dim, EPS, affine: false)
      trace_copy(e, trace, "ffn_norm", ffn_norm, n)
      encode_modulate(e, ffn_norm, modulation, weights["block_modulation"], ffn_mod,
        cfg, 3, 4, weights.dtype("block_modulation"))
      trace_copy(e, trace, "ffn_mod", ffn_mod, n)
      linear(e, ffn_mod, weights, "ffn_in_weight", "ffn_in_bias", ffn_in, rows, dim, cfg.ffn_dim)
      trace_copy(e, trace, "ffn_in", ffn_in, ff_count)
      MetalOps.gelu(e, ffn_in, ffn_gelu, ff_count.to_i32, approximate: true)
      trace_copy(e, trace, "ffn_gelu", ffn_gelu, ff_count)
      linear(e, ffn_gelu, weights, "ffn_out_weight", "ffn_out_bias", ffn_out, rows, cfg.ffn_dim, dim)
      trace_copy(e, trace, "ffn_out", ffn_out, n)
      encode_gated_residual(e, cross_res, ffn_out, modulation, weights["block_modulation"], y,
        cfg, 5, weights.dtype("block_modulation"))
      trace_copy(e, trace, "output", y, n)
    end

    # Encode the block followed by its audio injector using a caller-owned,
    # preallocated `block_output` buffer. Validate the shared video layout
    # before either operator appends work to the command encoder.
    def self.encode_with_audio(e : ML::Metal::ComputeEncoder,
                               x : ML::MetalBuffer, modulation : ML::MetalBuffer,
                               positions : ML::MetalBuffer, context : ML::MetalBuffer,
                               local_audio : ML::MetalBuffer, global_audio : ML::MetalBuffer,
                               block_output : ML::MetalBuffer, y : ML::MetalBuffer,
                               weights : Arena, scratch : Arena,
                               cfg : WanS2VConfig, audio_cfg : WanS2VAudioConfig,
                               block_trace : Hash(String, ML::MetalBuffer)? = nil,
                               audio_trace : Hash(String, ML::MetalBuffer)? = nil) : Nil
      validate_audio_pair!(cfg, audio_cfg)
      encode(e, x, modulation, positions, context, block_output, weights, scratch, cfg, block_trace)
      WanS2VAudioInjector.encode(e, block_output, local_audio, global_audio, y,
        weights, scratch, audio_cfg, audio_trace)
    end

    def self.validate_audio_pair!(cfg : WanS2VConfig, audio_cfg : WanS2VAudioConfig) : Nil
      unless cfg.batches == audio_cfg.batches && cfg.seq == audio_cfg.seq &&
             cfg.dim == audio_cfg.dim && cfg.heads == audio_cfg.heads &&
             cfg.video_frames == audio_cfg.video_frames &&
             cfg.spatial_tokens == audio_cfg.spatial_tokens
        raise ArgumentError.new("Wan S2V block/audio configurations must share batches, sequence, channels, heads, and video grid")
      end
      unless cfg.split == cfg.video_tokens && audio_cfg.video_tokens == cfg.video_tokens
        raise ArgumentError.new("Wan S2V block split must equal the shared video prefix")
      end
    end

    private def self.encode_qkv(e : ML::Metal::ComputeEncoder, x : ML::MetalBuffer,
                                q : ML::MetalBuffer, k : ML::MetalBuffer, v : ML::MetalBuffer,
                                w : Arena, rows : Int32, dim : Int32) : Nil
      linear(e, x, w, "self_q_weight", "self_q_bias", q, rows, dim, dim)
      linear(e, x, w, "self_k_weight", "self_k_bias", k, rows, dim, dim)
      linear(e, x, w, "self_v_weight", "self_v_bias", v, rows, dim, dim)
    end

    private def self.linear(e : ML::Metal::ComputeEncoder, x : ML::MetalBuffer,
                            weights : Arena, weight_name : String, bias_name : String,
                            y : ML::MetalBuffer, rows : Int32, in_dim : Int32,
                            out_dim : Int32) : Nil
      MetalOps.linear(e, x, weights[weight_name], weights[bias_name], y,
        rows, in_dim, out_dim, weight_dtype: weights.dtype(weight_name),
        bias_dtype: weights.dtype(bias_name))
    end

    private def self.norm_rms(e : ML::Metal::ComputeEncoder, x : ML::MetalBuffer,
                              weight : ML::MetalBuffer, weight_dtype : WeightDType,
                              bias : ML::MetalBuffer, y : ML::MetalBuffer,
                              rows : Int32, dim : Int32) : Nil
      MetalOps.norm(e, x, weight, bias, y, rows, dim, EPS, rms: true, affine: true,
        weight_dtype: weight_dtype)
    end

    private def self.encode_modulate(e : ML::Metal::ComputeEncoder, x : ML::MetalBuffer,
                                     modulation : ML::MetalBuffer, base : ML::MetalBuffer,
                                     y : ML::MetalBuffer, cfg : WanS2VConfig,
                                     shift_field : Int32, scale_field : Int32,
                                     base_dtype : WeightDType) : Nil
      e.set_pipeline(pipeline("wan_modulate_segments"))
      e.set_buffer(x, 0).set_buffer(modulation, 1).set_buffer(base, 2)
      e.set_buffer(base, 10).set_value(base_dtype.bf16? ? 1_u32 : 0_u32, 11)
      e.set_buffer(y, 3, ML::Metal::BufferAccess::Write)
      e.set_value(cfg.batches.to_u32, 4).set_value(cfg.seq.to_u32, 5).set_value(cfg.dim.to_u32, 6)
      e.set_value(cfg.split.to_u32, 7).set_value(shift_field.to_u32, 8).set_value(scale_field.to_u32, 9)
      e.dispatch_1d(cfg.values)
    end

    private def self.encode_gated_residual(e : ML::Metal::ComputeEncoder, x : ML::MetalBuffer,
                                           delta : ML::MetalBuffer, modulation : ML::MetalBuffer,
                                           base : ML::MetalBuffer, y : ML::MetalBuffer,
                                           cfg : WanS2VConfig, gate_field : Int32,
                                           base_dtype : WeightDType) : Nil
      e.set_pipeline(pipeline("wan_segment_add"))
      e.set_buffer(x, 0).set_buffer(delta, 1).set_buffer(modulation, 2).set_buffer(base, 3)
      e.set_buffer(base, 10).set_value(base_dtype.bf16? ? 1_u32 : 0_u32, 11)
      e.set_buffer(y, 4, ML::Metal::BufferAccess::Write)
      e.set_value(cfg.batches.to_u32, 5).set_value(cfg.seq.to_u32, 6).set_value(cfg.dim.to_u32, 7)
      e.set_value(cfg.split.to_u32, 8).set_value(gate_field.to_u32, 9)
      e.dispatch_1d(cfg.values)
    end

    private def self.encode_rope(e : ML::Metal::ComputeEncoder, x : ML::MetalBuffer,
                                 positions : ML::MetalBuffer, y : ML::MetalBuffer,
                                 cfg : WanS2VConfig, scratch : Arena) : Nil
      frequency_hi_name, frequency_lo_name = rope_frequency_names(cfg)
      e.set_pipeline(pipeline("wan_rope_f32"))
      e.set_buffer(x, 0).set_buffer(positions, 1).set_buffer(y, 2, ML::Metal::BufferAccess::Write)
      e.set_buffer(scratch[frequency_hi_name], 3).set_buffer(scratch[frequency_lo_name], 4)
      e.set_value(cfg.batches.to_u32, 5).set_value(cfg.seq.to_u32, 6)
      e.set_value(cfg.heads.to_u32, 7).set_value(cfg.head_dim.to_u32, 8)
      e.dispatch_1d(cfg.batches * cfg.seq * cfg.heads * (cfg.head_dim // 2))
    end

    # Frequency constants depend on theta and the per-axis pair partitions.
    # Keying setup buffers by that identity makes a reused Arena fail closed if
    # an encoder is accidentally given a different RoPE configuration.
    private def self.rope_frequency_names(cfg : WanS2VConfig) : {String, String}
      axis0, axis1, axis2 = cfg.rope_axis_dims
      identity = "#{cfg.theta}_#{cfg.head_dim}_#{axis0}_#{axis1}_#{axis2}"
      {"wan_b_rope_freq_hi_#{identity}", "wan_b_rope_freq_lo_#{identity}"}
    end

    # Wan builds these inverse frequencies in FP64 before constructing its
    # complex RoPE table. Keep a two-F32 expansion in resident setup buffers;
    # runtime phase construction and rotation remain entirely on the GPU.
    private def self.rope_frequencies(cfg : WanS2VConfig) : {Array(Float32), Array(Float32)}
      pairs = cfg.head_dim // 2
      frequency_hi = Array(Float32).new(pairs, 0.0_f32)
      frequency_lo = Array(Float32).new(pairs, 0.0_f32)
      theta = cfg.theta.to_f64
      pair = 0
      {cfg.rope_axis_dims[0], cfg.rope_axis_dims[1], cfg.rope_axis_dims[2]}.each do |axis_dim|
        axis_dim.times do |local_pair|
          frequency = 1.0_f64 / (theta ** (local_pair.to_f64 / axis_dim.to_f64))
          high = frequency.to_f32
          frequency_hi[pair] = high
          frequency_lo[pair] = (frequency - high.to_f64).to_f32
          pair += 1
        end
      end
      {frequency_hi, frequency_lo}
    end

    private def self.trace_copy(e : ML::Metal::ComputeEncoder,
                                trace : Hash(String, ML::MetalBuffer)?, name : String,
                                source : ML::MetalBuffer, count : Int64) : Nil
      if destinations = trace
        if target = destinations[name]?
          MetalOps.copy(e, source, target, count.to_i32)
        end
      end
    end

    private def self.validate_inputs!(x : ML::MetalBuffer, modulation : ML::MetalBuffer,
                                      positions : ML::MetalBuffer, context : ML::MetalBuffer,
                                      y : ML::MetalBuffer, weights : Arena, scratch : Arena,
                                      cfg : WanS2VConfig) : Nil
      n = cfg.values.to_i64
      dim = cfg.dim.to_i64
      ctx = cfg.context_rows.to_i64 * dim
      ff = cfg.rows.to_i64 * cfg.ffn_dim
      MetalOps.check(x, n)
      MetalOps.check(y, n)
      MetalOps.check(modulation, cfg.batches.to_i64 * 12 * cfg.dim)
      MetalOps.check(positions, cfg.batches.to_i64 * cfg.seq * 3)
      MetalOps.check(context, ctx)
      MetalOps.distinct(y, x, modulation, positions, context)
      weight_count = dim * dim
      weights.check_weight("block_modulation", 6_i64 * dim)
      weights.check_weight("norm3_weight", dim)
      weights.check_weight("norm3_bias", dim)
      MetalOps.check(scratch["wan_b_zero_bias"], dim)
      frequency_hi_name, frequency_lo_name = rope_frequency_names(cfg)
      MetalOps.check(scratch[frequency_hi_name], cfg.head_dim // 2)
      MetalOps.check(scratch[frequency_lo_name], cfg.head_dim // 2)
      weights.check_weight("self_q_weight", weight_count)
      weights.check_weight("self_q_bias", dim)
      weights.check_weight("self_k_weight", weight_count)
      weights.check_weight("self_k_bias", dim)
      weights.check_weight("self_v_weight", weight_count)
      weights.check_weight("self_v_bias", dim)
      weights.check_weight("self_o_weight", weight_count)
      weights.check_weight("self_o_bias", dim)
      weights.check_weight("self_norm_q_weight", dim)
      weights.check_weight("self_norm_k_weight", dim)
      weights.check_weight("cross_q_weight", weight_count)
      weights.check_weight("cross_q_bias", dim)
      weights.check_weight("cross_k_weight", weight_count)
      weights.check_weight("cross_k_bias", dim)
      weights.check_weight("cross_v_weight", weight_count)
      weights.check_weight("cross_v_bias", dim)
      weights.check_weight("cross_o_weight", weight_count)
      weights.check_weight("cross_o_bias", dim)
      weights.check_weight("cross_norm_q_weight", dim)
      weights.check_weight("cross_norm_k_weight", dim)
      weights.check_weight("ffn_in_weight", cfg.ffn_dim.to_i64 * dim)
      weights.check_weight("ffn_in_bias", cfg.ffn_dim.to_i64)
      weights.check_weight("ffn_out_weight", dim * cfg.ffn_dim)
      weights.check_weight("ffn_out_bias", dim)
      MetalOps.check(scratch["wan_b_norm1"], n)
      MetalOps.check(scratch["wan_b_mod1"], n)
      MetalOps.check(scratch["wan_b_self_q"], n)
      MetalOps.check(scratch["wan_b_self_k"], n)
      MetalOps.check(scratch["wan_b_self_v"], n)
      MetalOps.check(scratch["wan_b_self_qn"], n)
      MetalOps.check(scratch["wan_b_self_kn"], n)
      MetalOps.check(scratch["wan_b_self_qr"], n)
      MetalOps.check(scratch["wan_b_self_kr"], n)
      MetalOps.check(scratch["wan_b_self_attn"], n)
      MetalOps.check(scratch["wan_b_self_proj"], n)
      MetalOps.check(scratch["wan_b_self_res"], n)
      MetalOps.check(scratch["wan_b_cross_norm"], n)
      MetalOps.check(scratch["wan_b_cross_q"], n)
      MetalOps.check(scratch["wan_b_cross_k"], ctx)
      MetalOps.check(scratch["wan_b_cross_v"], ctx)
      MetalOps.check(scratch["wan_b_cross_qn"], n)
      MetalOps.check(scratch["wan_b_cross_kn"], ctx)
      MetalOps.check(scratch["wan_b_cross_attn"], n)
      MetalOps.check(scratch["wan_b_cross_proj"], n)
      MetalOps.check(scratch["wan_b_cross_res"], n)
      MetalOps.check(scratch["wan_b_ffn_norm"], n)
      MetalOps.check(scratch["wan_b_ffn_mod"], n)
      MetalOps.check(scratch["wan_b_ffn_in"], ff)
      MetalOps.check(scratch["wan_b_ffn_gelu"], ff)
      MetalOps.check(scratch["wan_b_ffn_out"], n)
      ensure_disjoint!(y, x, modulation, positions, context, weights, scratch)
    end

    private def self.ensure_disjoint!(y : ML::MetalBuffer, x : ML::MetalBuffer,
                                      modulation : ML::MetalBuffer, positions : ML::MetalBuffer,
                                      context : ML::MetalBuffer, weights : Arena,
                                      scratch : Arena) : Nil
      scratch.buffers.each do |name, buffer|
        reject_overlap!(buffer, y)
        reject_overlap!(buffer, x)
        reject_overlap!(buffer, modulation)
        reject_overlap!(buffer, positions)
        reject_overlap!(buffer, context)
        weights.buffers.each_value { |weight| reject_overlap!(buffer, weight) }
        scratch.buffers.each do |other_name, other|
          reject_overlap!(buffer, other) if name < other_name
        end
      end
      weights.buffers.each_value do |buffer|
        reject_overlap!(buffer, y)
        reject_overlap!(buffer, x)
        reject_overlap!(buffer, modulation)
        reject_overlap!(buffer, positions)
        reject_overlap!(buffer, context)
      end
    end

    private def self.reject_overlap!(a : ML::MetalBuffer, b : ML::MetalBuffer) : Nil
      raise ArgumentError.new("Wan S2V buffers overlap unsafely") if MetalOps.overlap?(a, b)
    end

    private def self.validate_trace!(trace : Hash(String, ML::MetalBuffer)?,
                                     x : ML::MetalBuffer, modulation : ML::MetalBuffer,
                                     positions : ML::MetalBuffer, context : ML::MetalBuffer,
                                     y : ML::MetalBuffer, weights : Arena, scratch : Arena,
                                     cfg : WanS2VConfig) : Nil
      return unless destinations = trace
      n = cfg.values.to_i64
      ctx = cfg.context_rows.to_i64 * cfg.dim
      ff = cfg.rows.to_i64 * cfg.ffn_dim
      destinations.each do |name, buffer|
        size = case CHECKPOINTS[name]?
               when :ctx then ctx
               when :ff  then ff
               when :n   then n
               else           raise ArgumentError.new("Unknown Wan S2V checkpoint #{name}")
               end
        MetalOps.check(buffer, size)
        reject_overlap!(buffer, x)
        reject_overlap!(buffer, modulation)
        reject_overlap!(buffer, positions)
        reject_overlap!(buffer, context)
        reject_overlap!(buffer, y)
        weights.buffers.each do |weight_name, weight|
          reject_overlap!(buffer, weight)
        end
        scratch.buffers.each do |scratch_name, temporary|
          reject_overlap!(buffer, temporary)
        end
        destinations.each do |other_name, other|
          reject_overlap!(buffer, other) if name != other_name
        end
      end
    end
  end

  module WanS2VAudioInjector
    SOURCE    = WanS2VBlock::SOURCE
    EPS       = 1e-6_f32
    ADALN_EPS = 1e-5_f32

    private def self.pipeline(name : String) : ML::Metal::ComputePipeline
      ML::Metal::PipelineCache.get(name) { ML::Metal::ComputePipeline.new(name, SOURCE) }
    end

    def self.prepare! : Nil
      MetalOps.prepare!
      {"wan_audio_gather", "wan_audio_silu", "wan_audio_adain", "wan_audio_scatter"}.each do |name|
        pipeline(name)
      end
      {"video_linear_f32", "video_norm_f32", "video_attention_f32", "video_copy_f32"}.each do |name|
        MetalOps.pipeline(name)
      end
    end

    def self.prepare_scratch!(arena : Arena, cfg : WanS2VAudioConfig) : Nil
      gathered = cfg.gathered_rows.to_i64 * cfg.dim
      frames = cfg.frame_rows.to_i64 * cfg.dim
      audio = cfg.audio_rows.to_i64 * cfg.dim
      arena.upload("wan_a_zero_bias", Array(Float32).new(cfg.dim, 0.0_f32))
      {
        {"wan_a_gather", gathered}, {"wan_a_norm", gathered},
        {"wan_a_global_silu", frames}, {"wan_a_adain_proj", cfg.frame_rows.to_i64 * cfg.dim * 2},
        {"wan_a_adain", gathered}, {"wan_a_q", gathered}, {"wan_a_qn", gathered},
        {"wan_a_k", audio}, {"wan_a_kn", audio}, {"wan_a_v", audio},
        {"wan_a_attn", gathered}, {"wan_a_proj", gathered},
      }.each { |name, count| arena.alloc(name, count.to_i32) }
    end

    CHECKPOINTS = {
      "gathered_video" => :video, "audio_norm" => :video, "global_silu" => :global,
      "adain_projection" => :adain, "adain_video" => :video, "audio_q" => :video,
      "audio_k" => :audio, "audio_v" => :audio, "audio_attention" => :video,
      "audio_projection" => :video, "audio_output" => :all,
    }

    # Local audio is [B,F,5,C]; global audio is [B,F,1,C], both produced by the
    # upstream audio/MotionEncoder path. The output copies the reference/motion
    # suffix byte-for-byte and adds cross-attention only to F*spatial video rows.
    def self.encode(e : ML::Metal::ComputeEncoder, x : ML::MetalBuffer,
                    local_audio : ML::MetalBuffer, global_audio : ML::MetalBuffer,
                    y : ML::MetalBuffer, weights : Arena, scratch : Arena,
                    cfg : WanS2VAudioConfig,
                    trace : Hash(String, ML::MetalBuffer)? = nil) : Nil
      validate_inputs!(x, local_audio, global_audio, y, weights, scratch, cfg)
      validate_trace!(trace, x, local_audio, global_audio, y, weights, scratch, cfg)
      gathered = scratch["wan_a_gather"]
      normed = scratch["wan_a_norm"]
      global_silu = scratch["wan_a_global_silu"]
      adain_proj = scratch["wan_a_adain_proj"]
      adain = scratch["wan_a_adain"]
      q = scratch["wan_a_q"]
      qn = scratch["wan_a_qn"]
      k = scratch["wan_a_k"]
      kn = scratch["wan_a_kn"]
      v = scratch["wan_a_v"]
      attn = scratch["wan_a_attn"]
      proj = scratch["wan_a_proj"]
      video_count = cfg.gathered_rows.to_i64 * cfg.dim
      global_count = cfg.frame_rows.to_i64 * cfg.dim
      audio_count = cfg.audio_rows.to_i64 * cfg.dim
      all_count = cfg.values

      e.set_pipeline(pipeline("wan_audio_gather"))
      e.set_buffer(x, 0).set_buffer(gathered, 1, ML::Metal::BufferAccess::Write)
      e.set_value(cfg.batches.to_u32, 2).set_value(cfg.seq.to_u32, 3)
      e.set_value(cfg.video_frames.to_u32, 4).set_value(cfg.spatial_tokens.to_u32, 5).set_value(cfg.dim.to_u32, 6)
      e.dispatch_1d(video_count.to_i32)
      trace_copy(e, trace, "gathered_video", gathered, video_count)

      MetalOps.norm(e, gathered, scratch["wan_a_zero_bias"], scratch["wan_a_zero_bias"], normed,
        cfg.gathered_rows, cfg.dim, ADALN_EPS, affine: false)
      trace_copy(e, trace, "audio_norm", normed, video_count)
      e.set_pipeline(pipeline("wan_audio_silu"))
      e.set_buffer(global_audio, 0).set_buffer(global_silu, 1, ML::Metal::BufferAccess::Write)
      e.set_value(global_count.to_u32, 2)
      e.dispatch_1d(global_count.to_i32)
      trace_copy(e, trace, "global_silu", global_silu, global_count)
      linear(e, global_silu, weights, "adain_weight", "adain_bias",
        adain_proj, cfg.frame_rows, cfg.dim, 2 * cfg.dim)
      trace_copy(e, trace, "adain_projection", adain_proj, cfg.frame_rows.to_i64 * 2 * cfg.dim)
      e.set_pipeline(pipeline("wan_audio_adain"))
      e.set_buffer(normed, 0).set_buffer(adain_proj, 1).set_buffer(adain, 2, ML::Metal::BufferAccess::Write)
      e.set_value(cfg.batches.to_u32, 3).set_value(cfg.video_frames.to_u32, 4)
      e.set_value(cfg.spatial_tokens.to_u32, 5).set_value(cfg.dim.to_u32, 6)
      e.dispatch_1d(video_count.to_i32)
      trace_copy(e, trace, "adain_video", adain, video_count)

      linear(e, adain, weights, "audio_q_weight", "audio_q_bias", q,
        cfg.gathered_rows, cfg.dim, cfg.dim)
      linear(e, local_audio, weights, "audio_k_weight", "audio_k_bias", k,
        cfg.audio_rows, cfg.dim, cfg.dim)
      linear(e, local_audio, weights, "audio_v_weight", "audio_v_bias", v,
        cfg.audio_rows, cfg.dim, cfg.dim)
      trace_copy(e, trace, "audio_q", q, video_count)
      trace_copy(e, trace, "audio_k", k, audio_count)
      trace_copy(e, trace, "audio_v", v, audio_count)
      MetalOps.norm(e, q, weights["audio_q_norm_weight"], scratch["wan_a_zero_bias"], qn,
        cfg.gathered_rows, cfg.dim, EPS, rms: true, affine: true,
        weight_dtype: weights.dtype("audio_q_norm_weight"), bias_dtype: scratch.dtype("wan_a_zero_bias"))
      MetalOps.norm(e, k, weights["audio_k_norm_weight"], scratch["wan_a_zero_bias"], kn,
        cfg.audio_rows, cfg.dim, EPS, rms: true, affine: true,
        weight_dtype: weights.dtype("audio_k_norm_weight"), bias_dtype: scratch.dtype("wan_a_zero_bias"))
      MetalOps.attention(e, qn, kn, v, attn, cfg.spatial_tokens, cfg.audio_tokens,
        cfg.heads, cfg.head_dim, cfg.frame_rows)
      trace_copy(e, trace, "audio_attention", attn, video_count)
      linear(e, attn, weights, "audio_o_weight", "audio_o_bias", proj,
        cfg.gathered_rows, cfg.dim, cfg.dim)
      trace_copy(e, trace, "audio_projection", proj, video_count)
      e.set_pipeline(pipeline("wan_audio_scatter"))
      e.set_buffer(x, 0).set_buffer(proj, 1).set_buffer(y, 2, ML::Metal::BufferAccess::Write)
      e.set_value(cfg.batches.to_u32, 3).set_value(cfg.seq.to_u32, 4)
      e.set_value(cfg.video_tokens.to_u32, 5).set_value(cfg.dim.to_u32, 6)
      e.dispatch_1d(all_count)
      trace_copy(e, trace, "audio_output", y, all_count.to_i64)
    end

    private def self.linear(e : ML::Metal::ComputeEncoder, x : ML::MetalBuffer,
                            weights : Arena, weight_name : String, bias_name : String,
                            y : ML::MetalBuffer, rows : Int32, in_dim : Int32,
                            out_dim : Int32) : Nil
      MetalOps.linear(e, x, weights[weight_name], weights[bias_name], y,
        rows, in_dim, out_dim, weight_dtype: weights.dtype(weight_name),
        bias_dtype: weights.dtype(bias_name))
    end

    private def self.trace_copy(e : ML::Metal::ComputeEncoder,
                                trace : Hash(String, ML::MetalBuffer)?, name : String,
                                source : ML::MetalBuffer, count : Int64) : Nil
      if destinations = trace
        if target = destinations[name]?
          MetalOps.copy(e, source, target, count.to_i32)
        end
      end
    end

    private def self.validate_inputs!(x : ML::MetalBuffer, local_audio : ML::MetalBuffer,
                                      global_audio : ML::MetalBuffer, y : ML::MetalBuffer,
                                      weights : Arena, scratch : Arena,
                                      cfg : WanS2VAudioConfig) : Nil
      n = cfg.values.to_i64
      dim = cfg.dim.to_i64
      gathered = cfg.gathered_rows.to_i64 * dim
      frames = cfg.frame_rows.to_i64 * dim
      audio = cfg.audio_rows.to_i64 * dim
      MetalOps.check(x, n)
      MetalOps.check(y, n)
      MetalOps.check(local_audio, audio)
      MetalOps.check(global_audio, frames)
      MetalOps.distinct(y, x, local_audio, global_audio)
      {
        {"adain_weight", dim * 2 * dim}, {"adain_bias", dim * 2},
        {"audio_q_weight", dim * dim}, {"audio_q_bias", dim},
        {"audio_k_weight", dim * dim}, {"audio_k_bias", dim},
        {"audio_v_weight", dim * dim}, {"audio_v_bias", dim},
        {"audio_o_weight", dim * dim}, {"audio_o_bias", dim},
        {"audio_q_norm_weight", dim}, {"audio_k_norm_weight", dim},
      }.each { |name, count| weights.check_weight(name, count) }
      MetalOps.check(scratch["wan_a_zero_bias"], dim)
      {
        {"wan_a_gather", gathered}, {"wan_a_norm", gathered},
        {"wan_a_global_silu", frames}, {"wan_a_adain_proj", cfg.frame_rows.to_i64 * dim * 2},
        {"wan_a_adain", gathered}, {"wan_a_q", gathered}, {"wan_a_qn", gathered},
        {"wan_a_k", audio}, {"wan_a_kn", audio}, {"wan_a_v", audio},
        {"wan_a_attn", gathered}, {"wan_a_proj", gathered},
      }.each { |name, count| MetalOps.check(scratch[name], count) }
      scratch.buffers.each do |name, buffer|
        if MetalOps.overlap?(buffer, y) || MetalOps.overlap?(buffer, x) || MetalOps.overlap?(buffer, local_audio) || MetalOps.overlap?(buffer, global_audio)
          raise ArgumentError.new("Wan audio input/output aliases scratch buffer #{name}")
        end
        weights.buffers.each do |weight_name, weight|
          raise ArgumentError.new("Wan audio scratch aliases weight #{weight_name}") if MetalOps.overlap?(buffer, weight)
        end
        scratch.buffers.each do |other_name, other|
          raise ArgumentError.new("Wan audio scratch buffers overlap") if name < other_name && MetalOps.overlap?(buffer, other)
        end
      end
      weights.buffers.each do |name, buffer|
        if MetalOps.overlap?(buffer, y) || MetalOps.overlap?(buffer, x) || MetalOps.overlap?(buffer, local_audio) || MetalOps.overlap?(buffer, global_audio)
          raise ArgumentError.new("Wan audio input/output aliases weight #{name}")
        end
      end
      weights.buffers.each do |name, buffer|
        weights.buffers.each do |other_name, other|
          raise ArgumentError.new("Wan audio weights overlap") if name < other_name && MetalOps.overlap?(buffer, other)
        end
      end
    end

    private def self.validate_trace!(trace : Hash(String, ML::MetalBuffer)?,
                                     x : ML::MetalBuffer, local_audio : ML::MetalBuffer,
                                     global_audio : ML::MetalBuffer, y : ML::MetalBuffer,
                                     weights : Arena, scratch : Arena,
                                     cfg : WanS2VAudioConfig) : Nil
      return unless destinations = trace
      video = cfg.gathered_rows.to_i64 * cfg.dim
      global = cfg.frame_rows.to_i64 * cfg.dim
      audio = cfg.audio_rows.to_i64 * cfg.dim
      adain = cfg.frame_rows.to_i64 * cfg.dim * 2
      destinations.each do |name, buffer|
        count = case name
                when "gathered_video", "audio_norm", "adain_video", "audio_q", "audio_attention", "audio_projection" then video
                when "audio_output"                                                                                  then cfg.values.to_i64
                when "global_silu"                                                                                   then global
                when "adain_projection"                                                                              then adain
                when "audio_k", "audio_v"                                                                            then audio
                else                                                                                                      raise ArgumentError.new("Unknown Wan audio checkpoint #{name}")
                end
        MetalOps.check(buffer, count)
        {x, local_audio, global_audio, y}.each do |input|
          raise ArgumentError.new("Wan audio checkpoint aliases a live input/output") if MetalOps.overlap?(buffer, input)
        end
        weights.buffers.each do |weight_name, weight|
          raise ArgumentError.new("Wan audio checkpoint aliases weight #{weight_name}") if MetalOps.overlap?(buffer, weight)
        end
        scratch.buffers.each do |scratch_name, temporary|
          raise ArgumentError.new("Wan audio checkpoint aliases scratch #{scratch_name}") if MetalOps.overlap?(buffer, temporary)
        end
        destinations.each do |other_name, other|
          if name != other_name && MetalOps.overlap?(buffer, other)
            raise ArgumentError.new("Wan audio checkpoints must use distinct destinations")
          end
        end
      end
    end
  end
end
