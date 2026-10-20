require "./wan_s2v_block"

module ML::Video
  # Production Wan2.2 S2V topology with fixed sequence lengths shared by the
  # batch. Inputs to the layer runner are resident token sequences; patching,
  # text/time projection and the audio MotionEncoder are separate entry points.
  # Bucketed XLSR audio features are supplied by the conditioning boundary.
  struct WanS2VTransformerConfig
    INJECTED_BLOCK_IDS = [0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39]

    getter batches : Int32
    getter context_tokens : Int32
    getter dim : Int32
    getter heads : Int32
    getter ffn_dim : Int32
    getter layers : Int32
    getter video_frames : Int32
    getter spatial_tokens : Int32
    getter latent_height : Int32
    getter latent_width : Int32
    getter injection_ids : Array(Int32)
    getter theta : Float32
    getter text_dim : Int32
    getter text_tokens : Int32
    getter freq_dim : Int32
    getter in_dim : Int32
    getter out_dim : Int32
    getter audio_dim : Int32
    getter patch_tile_rows : Int32
    getter drop_motion_frames : Bool

    def initialize(@batches : Int32 = 1, @context_tokens : Int32 = 512,
                   @dim : Int32 = 5120, @heads : Int32 = 40,
                   @ffn_dim : Int32 = 13824, @layers : Int32 = 40,
                   @video_frames : Int32 = 21, @spatial_tokens : Int32 = 1560,
                   @latent_height : Int32 = 60, @latent_width : Int32 = 104,
                   injection_ids : Array(Int32) = INJECTED_BLOCK_IDS,
                   @theta : Float32 = 10000.0_f32,
                   @text_dim : Int32 = 4096, @text_tokens : Int32 = 512,
                   @freq_dim : Int32 = 256, @in_dim : Int32 = 16,
                   @out_dim : Int32 = 16, @audio_dim : Int32 = 1024,
                   @patch_tile_rows : Int32 = 128,
                   @drop_motion_frames : Bool = false)
      @injection_ids = injection_ids.dup
      validate!
    end

    def validate! : Nil
      values = {batches, context_tokens, dim, heads, ffn_dim, layers,
                video_frames, spatial_tokens, latent_height, latent_width,
                text_dim, text_tokens, freq_dim, in_dim, out_dim, audio_dim,
                patch_tile_rows}
      raise ArgumentError.new("Wan S2V transformer dimensions must be positive") unless values.all? { |v| v > 0 }
      raise ArgumentError.new("Wan S2V RoPE theta must be finite and positive") unless theta.finite? && theta > 0
      raise ArgumentError.new("Wan S2V dim must divide evenly across heads") unless dim % heads == 0
      head_dim = dim // heads
      raise ArgumentError.new("Wan S2V head dimension must be even and <= 256") if head_dim < 6 || head_dim > 256 || head_dim.odd?
      raise ArgumentError.new("Wan S2V timestep frequency width must be even") if freq_dim.odd?
      raise ArgumentError.new("Wan S2V timestep activation is too large") if (batches.to_i64 + 1) * freq_dim > Int32::MAX
      raise ArgumentError.new("Wan S2V FramePack requires latent dimensions >= 8") if latent_height < 8 || latent_width < 8
      unless (latent_height // 2) * (latent_width // 2) == spatial_tokens && latent_height.even? && latent_width.even?
        raise ArgumentError.new("Wan S2V latent grid must match the 1x2x2 patch grid")
      end
      raise ArgumentError.new("Wan S2V sequence is too large") if sequence_tokens.to_i64 * batches * dim > Int32::MAX
      raise ArgumentError.new("Wan S2V FFN activation is too large") if sequence_tokens.to_i64 * batches * ffn_dim > Int32::MAX
      raise ArgumentError.new("Wan S2V text activation is too large") if batches.to_i64 * text_tokens * Math.max(text_dim, dim) > Int32::MAX
      raise ArgumentError.new("Wan S2V time activation is too large") if (batches.to_i64 + 1) * 6 * dim > Int32::MAX
      raise ArgumentError.new("Wan S2V patch tile is too large") if patch_tile_rows.to_i64 * dim > Int32::MAX || patch_tile_rows.to_i64 * in_dim * 256 > Int32::MAX
      raise ArgumentError.new("Wan S2V requires unique in-range audio injection layers") unless injection_ids.uniq == injection_ids && injection_ids.all? { |i| i >= 0 && i < layers }
      block_config # validate the shared block geometry
      audio_config # validate audio prefix geometry
    end

    def video_tokens : Int32
      checked_count("video tokens", video_frames, spatial_tokens)
    end

    def sequence_tokens : Int32
      value = video_tokens.to_i64 + reference_tokens + framepack_tokens
      raise ArgumentError.new("Wan S2V sequence tokens exceeds Int32") unless value > 0 && value <= Int32::MAX
      value.to_i32
    end

    def reference_tokens : Int32
      (latent_height // 2) * (latent_width // 2)
    end

    # FramePackMotioner emits one recent 1x2x2 bucket, one 2x4x4 bucket,
    # then the 16x8x8 far bucket (all valid Conv3d output sizes floor).
    def framepack_tokens : Int32
      return 0 if drop_motion_frames
      recent = (latent_height // 2) * (latent_width // 2)
      middle = (latent_height // 4) * (latent_width // 4)
      far = 4 * (latent_height // 8) * (latent_width // 8)
      recent + middle + far
    end

    def suffix_tokens : Int32
      reference_tokens + framepack_tokens
    end

    def rows : Int32
      checked_count("transformer rows", batches, sequence_tokens)
    end

    def block_config : WanS2VConfig
      WanS2VConfig.new(batches, sequence_tokens, context_tokens, dim, heads,
        ffn_dim, video_tokens, theta, video_frames, spatial_tokens)
    end

    def audio_config : WanS2VAudioConfig
      WanS2VAudioConfig.new(batches, sequence_tokens, dim, heads,
        video_frames, spatial_tokens, 5)
    end

    private def checked_count(label : String, first : Int32, second : Int32) : Int32
      value = first.to_i64 * second
      raise ArgumentError.new("Wan S2V #{label} exceeds Int32") unless value > 0 && value <= Int32::MAX
      value.to_i32
    end
  end

  # Native GPU forward through all 40 distinct WanS2VAttentionBlocks, with the
  # twelve official local-audio injections. Every layer consumes its own
  # resident BF16 parameters and uses the admitted block implementation.
  module WanS2VTransformer
    SOURCE             = {{ read_file("#{__DIR__}/kernels/wan_s2v_transformer.metal") }}
    INJECTED_BLOCK_IDS = WanS2VTransformerConfig::INJECTED_BLOCK_IDS
    # Non-block/non-injector transformer weights. The causal audio encoder's
    # twelve source tensors are bound by WanS2VCausalAudioEncoder.
    EXTRA_WEIGHT_COUNT =       24
    S2V_EPS            = 1e-6_f32

    def self.checkpoint_bindings(cfg : WanS2VTransformerConfig,
                                 dtype : ML::GGUF::SafeTensorDType = ML::GGUF::SafeTensorDType::BF16) : Array(WeightBinding)
      cfg.validate!
      result = [] of WeightBinding
      cfg.layers.times do |layer|
        injector = -1
        cfg.injection_ids.each_with_index do |id, index|
          injector = index if id == layer
        end
        block_prefix = "block.#{layer}."
        audio_prefix = injector >= 0 ? "audio.#{injector}." : "unused.#{layer}."
        WanS2VBlock.checkpoint_bindings(cfg.block_config, dtype, layer.to_i32,
          Math.max(injector, 0).to_i32, block_prefix, audio_prefix).each do |binding|
          if binding.source.starts_with?("blocks.") || (injector >= 0 && binding.source.starts_with?("audio_injector."))
            result << binding
          end
        end
      end

      add_extra_bindings(result, cfg, dtype)
      expected = cfg.layers * 27 + cfg.injection_ids.size * 12 + EXTRA_WEIGHT_COUNT
      unless result.size == expected
        raise ArgumentError.new("Wan S2V checkpoint binding count mismatch: expected #{expected}, got #{result.size}")
      end
      targets = Set(String).new
      sources = Set(String).new
      result.each do |binding|
        raise ArgumentError.new("Duplicate Wan S2V checkpoint target #{binding.target}") unless targets.add?(binding.target)
        raise ArgumentError.new("Duplicate Wan S2V checkpoint source #{binding.source}") unless sources.add?(binding.source)
        unless binding.dtype == dtype
          raise ArgumentError.new("Wan S2V checkpoint dtype mismatch for #{binding.source}")
        end
      end
      result
    end

    # Allocate reusable block/audio scratch sets before encoding. The two
    # arenas stay separate because their temporary names and live ranges differ.
    def self.prepare_scratch!(block_scratch : Arena, audio_scratch : Arena,
                              cfg : WanS2VTransformerConfig) : Nil
      cfg.validate!
      WanS2VBlock.prepare_scratch!(block_scratch, cfg.block_config)
      WanS2VAudioInjector.prepare_scratch!(audio_scratch, cfg.audio_config)
    end

    # Allocate the shared time-embedding and patch-projection tiles before
    # encoding. The 4x FramePack Conv3d patch is the largest input row here:
    # `in_dim * 4 * 8 * 8` values per tile row.
    def self.transformer_scratch_plan(cfg : WanS2VTransformerConfig) : Array(Tuple(String, Int32))
      time_rows = cfg.batches + 1
      patch_features = cfg.in_dim * 4 * 8 * 8
      [
        {"wan_t_time_sinusoid", time_rows * cfg.freq_dim},
        {"wan_t_time_hidden", time_rows * cfg.dim},
        {"wan_t_time_embedding_all", time_rows * cfg.dim},
        {"wan_t_time_projection_input", time_rows * cfg.dim},
        {"wan_t_time_projection_all", time_rows * 6 * cfg.dim},
        {"wan_t_patch_input", cfg.patch_tile_rows * patch_features},
        {"wan_t_patch_projected", cfg.patch_tile_rows * cfg.dim},
      ]
    end

    def self.transformer_scratch_bytes(cfg : WanS2VTransformerConfig) : Int64
      values = transformer_scratch_plan(cfg).sum { |_, count| count.to_i64 }
      (values + cfg.freq_dim) * sizeof(Float32)
    end

    # Setup-only constants are computed from the same FP64 formula as
    # `sinusoidal_embedding_1d`; runtime timestep trigonometry stays on Metal.
    def self.prepare_transformer_scratch!(scratch : Arena,
                                          cfg : WanS2VTransformerConfig) : Nil
      cfg.validate!
      frequencies_hi = Array(Float32).new(cfg.freq_dim // 2, 0.0_f32)
      frequencies_lo = Array(Float32).new(cfg.freq_dim // 2, 0.0_f32)
      half = cfg.freq_dim // 2
      half.times do |index|
        exact = 10000.0_f64 ** (-index.to_f64 / half.to_f64)
        high = exact.to_f32
        frequencies_hi[index] = high
        frequencies_lo[index] = (exact - high.to_f64).to_f32
      end
      scratch.upload(time_frequency_hi_name(cfg), frequencies_hi)
      scratch.upload(time_frequency_lo_name(cfg), frequencies_lo)
      transformer_scratch_plan(cfg).each { |name, count| scratch.alloc(name, count) }
    end

    def self.prepare! : Nil
      WanS2VBlock.prepare!
      WanS2VAudioInjector.prepare!
      {"wan_s2v_sinusoidal", "wan_s2v_silu", "wan_s2v_time_expand",
       "wan_s2v_patchify", "wan_s2v_scatter_tokens", "wan_s2v_head_norm",
       "wan_s2v_head_modulate", "wan_s2v_head_linear", "wan_s2v_unpatchify"}.each do |name|
        pipeline(name)
      end
      {"video_linear_f32", "video_norm_f32"}.each { |name| MetalOps.pipeline(name) }
    end

    # Encode the layer stack into an existing serial command encoder. `x` and
    # `alternate` are caller-owned [B,S,C] F32 buffers; all other activations
    # are resident F32. No buffer is created, uploaded, waited, or read here.
    # The optional execution chain submits separate asynchronous phases and
    # invalidates borrowed encoders. The returned buffer is the final sequence. Optional
    # trace destinations are caller-owned and copy only the selected layer.
    def self.encode_layers(e : ML::Metal::ComputeEncoder,
                           x : ML::MetalBuffer, alternate : ML::MetalBuffer,
                           modulation : ML::MetalBuffer, positions : ML::MetalBuffer,
                           context : ML::MetalBuffer, local_audio : ML::MetalBuffer,
                           global_audio : ML::MetalBuffer, weights : Arena,
                           block_scratch : Arena, audio_scratch : Arena,
                           cfg : WanS2VTransformerConfig,
                           trace_layer : Int32? = nil,
                           block_trace : Hash(String, ML::MetalBuffer)? = nil,
                           audio_trace : Hash(String, ML::MetalBuffer)? = nil,
                           tiled_attention : Bool = false,
                           execution_chain : ML::Metal::MPSCommandChain? = nil,
                           mps_workspace : WanS2VMPSLinearWorkspace? = nil,
                           attention_workspace : WanAttentionMPSChunkWorkspace? = nil) : ML::MetalBuffer
      cfg.validate!
      WanS2VBlock.validate_attention_backend!(cfg.dim // cfg.heads, tiled_attention)
      validate_attention_workspace!(cfg, block_scratch, execution_chain, attention_workspace)
      if active = execution_chain
        e = active.encoder
      end
      block_cfg = cfg.block_config
      audio_cfg = cfg.audio_config
      # Per-block checks are prefix-scoped. Protect every resident parameter
      # before any layer can overwrite storage belonging to a later block.
      validate_weight_aliases!(weights, x, alternate, modulation, positions,
        context, local_audio, global_audio)
      {block_scratch, audio_scratch}.each do |arena|
        arena.buffers.each_value do |buffer|
          validate_weight_aliases!(weights, buffer)
        end
      end
      {block_trace, audio_trace}.each do |destinations|
        destinations.try &.each_value do |buffer|
          validate_weight_aliases!(weights, buffer)
        end
      end
      count = cfg.rows.to_i64 * cfg.dim
      MetalOps.check(x, count)
      MetalOps.check(alternate, count)
      MetalOps.distinct(x, alternate)
      MetalOps.check(modulation, cfg.batches.to_i64 * 12 * cfg.dim)
      MetalOps.check(positions, cfg.batches.to_i64 * cfg.sequence_tokens * 3)
      MetalOps.check(context, cfg.batches.to_i64 * cfg.context_tokens * cfg.dim)
      MetalOps.check(local_audio, cfg.audio_config.audio_rows.to_i64 * cfg.dim)
      MetalOps.check(global_audio, cfg.audio_config.frame_rows.to_i64 * cfg.dim)

      current = x
      spare = alternate
      injector_index = 0
      cfg.layers.times do |layer|
        has_audio = injector_index < cfg.injection_ids.size && cfg.injection_ids[injector_index] == layer
        layer_trace = trace_layer == layer
        if has_audio
          # The block consumes current and writes spare. Once that dispatch is
          # ordered, its input is dead, so the injector can write current while
          # reading spare. Keep the two operators' independently sized scratch
          # arenas separate and preserve the upstream block-then-inject order.
          WanS2VBlock.encode(e, current, modulation, positions, context, spare,
            weights, block_scratch, block_cfg,
            layer_trace ? block_trace : nil, "block.#{layer}.", tiled_attention,
            execution_chain, mps_workspace, attention_workspace)
          if active = execution_chain
            e = active.encoder
          end
          WanS2VAudioInjector.encode(e, spare, local_audio, global_audio, current,
            weights, audio_scratch, audio_cfg,
            layer_trace ? audio_trace : nil, "audio.#{injector_index}.", tiled_attention,
            mps_workspace)
          injector_index += 1
        else
          WanS2VBlock.encode(e, current, modulation, positions, context, spare,
            weights, block_scratch, block_cfg,
            layer_trace ? block_trace : nil, "block.#{layer}.", tiled_attention,
            execution_chain, mps_workspace, attention_workspace)
          if active = execution_chain
            e = active.encoder
          end
          current, spare = spare, current
        end
        if active = execution_chain
          active.pause!
          active.continue_async!
          active.resume!
          e = active.encoder
        end
      end
      raise ArgumentError.new("Wan S2V did not consume every audio injection") unless injector_index == cfg.injection_ids.size
      current
    end

    # Complete resident-buffer transformer forward. All inputs, output
    # storage, weights, and scratch arenas are caller-owned and must remain
    # alive through encoder completion. This method only enqueues operations;
    # it does not allocate Metal buffers, upload, wait, or read back. The
    # optional execution chain permits separate asynchronous submissions;
    # callers must refresh their borrowed encoder after this method returns.
    # `local_audio` and `global_audio` are the already transformed outputs of
    # the native CausalAudioEncoder/FramePack conditioning stage. A selected
    # trace layer is intended for bounded fixture diagnostics.
    def self.encode(e : ML::Metal::ComputeEncoder,
                    video_latents : ML::MetalBuffer,
                    condition_latents : ML::MetalBuffer,
                    reference_latents : ML::MetalBuffer,
                    motion_latents : ML::MetalBuffer,
                    raw_context : ML::MetalBuffer,
                    text_hidden : ML::MetalBuffer,
                    projected_context : ML::MetalBuffer,
                    timestep : ML::MetalBuffer,
                    time_embedding : ML::MetalBuffer,
                    modulation : ML::MetalBuffer,
                    positions : ML::MetalBuffer,
                    local_audio : ML::MetalBuffer,
                    global_audio : ML::MetalBuffer,
                    sequence : ML::MetalBuffer,
                    alternate : ML::MetalBuffer,
                    prediction : ML::MetalBuffer,
                    latent_output : ML::MetalBuffer,
                    weights : Arena,
                    transformer_scratch : Arena,
                    block_scratch : Arena,
                    audio_scratch : Arena,
                    cfg : WanS2VTransformerConfig,
                    motion_frames : Int32,
                    trace_layer : Int32? = nil,
                    block_trace : Hash(String, ML::MetalBuffer)? = nil,
                    audio_trace : Hash(String, ML::MetalBuffer)? = nil,
                    tiled_attention : Bool = false,
                    execution_chain : ML::Metal::MPSCommandChain? = nil,
                    mps_workspace : WanS2VMPSLinearWorkspace? = nil,
                    attention_workspace : WanAttentionMPSChunkWorkspace? = nil) : ML::MetalBuffer
      cfg.validate!
      WanS2VBlock.validate_attention_backend!(cfg.dim // cfg.heads, tiled_attention)
      validate_attention_workspace!(cfg, block_scratch, execution_chain, attention_workspace)
      if active = execution_chain
        e = active.encoder
      end
      encode_patch_inputs(e, video_latents, condition_latents, reference_latents,
        motion_latents, sequence, weights, transformer_scratch, cfg, motion_frames)
      encode_text(e, raw_context, projected_context, text_hidden, weights, cfg)
      encode_time(e, timestep, time_embedding, modulation, weights, transformer_scratch, cfg)
      hidden = encode_layers(e, sequence, alternate, modulation, positions,
        projected_context, local_audio, global_audio, weights, block_scratch,
        audio_scratch, cfg, trace_layer, block_trace, audio_trace, tiled_attention,
        execution_chain, mps_workspace, attention_workspace)
      if active = execution_chain
        e = active.encoder
      end
      encode_head(e, hidden, time_embedding, prediction, latent_output, weights,
        cfg, cfg.video_frames, cfg.latent_height, cfg.latent_width)
      latent_output
    end

    private def self.validate_attention_workspace!(cfg : WanS2VTransformerConfig,
                                                   block_scratch : Arena,
                                                   chain : ML::Metal::MPSCommandChain?,
                                                   workspace : WanAttentionMPSChunkWorkspace?) : Nil
      return unless workspace
      raise ArgumentError.new("Prepared Wan attention requires an execution chain") unless chain
      workspace.validate_for!(cfg.batches, cfg.sequence_tokens, cfg.sequence_tokens,
        cfg.heads, cfg.dim // cfg.heads, block_scratch["wan_b_self_q"],
        block_scratch["wan_b_self_k"], block_scratch["wan_b_self_v"],
        block_scratch["wan_b_self_attn"])
    end

    # The first official request has no pose video. Wan's host path still
    # VAE-encodes its fallback COND and then evaluates `COND[0] * 0` before
    # passing it through the learned condition encoder. This in-place GPU
    # zeroing preserves that input exactly while avoiding a second latent-sized
    # buffer. The caller must have finished every other use of the VAE output.
    def self.zero_no_pose_condition(e : ML::Metal::ComputeEncoder,
                                    condition_latents : ML::MetalBuffer,
                                    cfg : WanS2VTransformerConfig) : Nil
      cfg.validate!
      count = cfg.batches.to_i64 * cfg.in_dim * cfg.video_frames *
              cfg.latent_height * cfg.latent_width
      raise ArgumentError.new("Wan S2V no-pose condition exceeds Int32 dispatch range") if count > Int32::MAX
      MetalOps.check(condition_latents, count)
      e.set_pipeline(pipeline("wan_s2v_zero_condition"))
      e.set_buffer(condition_latents, 0, ML::Metal::BufferAccess::Write)
      e.set_value(count.to_u32, 1)
      e.dispatch_1d(count.to_i32)
    end

    # Project a fixed-width, already padded UMT5 context. Tokenization and
    # sequence padding are caller-owned; neural projection remains on Metal.
    def self.encode_text(e : ML::Metal::ComputeEncoder, raw_context : ML::MetalBuffer,
                         projected_context : ML::MetalBuffer, text_hidden : ML::MetalBuffer,
                         weights : Arena, cfg : WanS2VTransformerConfig) : Nil
      rows = cfg.batches * cfg.text_tokens
      MetalOps.check(raw_context, rows.to_i64 * cfg.text_dim)
      MetalOps.check(projected_context, rows.to_i64 * cfg.dim)
      MetalOps.check(text_hidden, rows.to_i64 * cfg.dim)
      MetalOps.distinct(text_hidden, raw_context)
      MetalOps.distinct(projected_context, raw_context, text_hidden)
      validate_weight_aliases!(weights, raw_context, text_hidden, projected_context)
      prefix = "model."
      MetalOps.linear(e, raw_context, weights["#{prefix}text_embedding.0.weight"],
        weights["#{prefix}text_embedding.0.bias"], text_hidden, rows,
        cfg.text_dim, cfg.dim, weight_dtype: weights.dtype("#{prefix}text_embedding.0.weight"),
        bias_dtype: weights.dtype("#{prefix}text_embedding.0.bias"))
      MetalOps.gelu(e, text_hidden, text_hidden, rows * cfg.dim, approximate: true)
      MetalOps.linear(e, text_hidden, weights["#{prefix}text_embedding.2.weight"],
        weights["#{prefix}text_embedding.2.bias"], projected_context, rows,
        cfg.dim, cfg.dim, weight_dtype: weights.dtype("#{prefix}text_embedding.2.weight"),
        bias_dtype: weights.dtype("#{prefix}text_embedding.2.bias"))
    end

    # Upstream sinusoidal timestep embedding, MLP, and 6-way projection.
    # `time_embedding` is [B,C]; `modulation` is [B,6,2,C], with the zero-time
    # embedding assigned to each sample's reference/motion suffix.
    def self.encode_time(e : ML::Metal::ComputeEncoder, timestep : ML::MetalBuffer,
                         time_embedding : ML::MetalBuffer, modulation : ML::MetalBuffer,
                         weights : Arena, scratch : Arena,
                         cfg : WanS2VTransformerConfig) : Nil
      cfg.validate!
      time_rows = cfg.batches + 1
      output_modulation = cfg.batches.to_i64 * 12 * cfg.dim
      time_values = cfg.batches.to_i64 * cfg.dim
      MetalOps.check(timestep, cfg.batches)
      MetalOps.check(time_embedding, time_values)
      MetalOps.check(modulation, output_modulation)
      MetalOps.distinct(time_embedding, timestep, modulation)
      MetalOps.distinct(modulation, timestep)
      validate_transformer_scratch!(scratch, cfg)
      validate_transformer_scratch_aliases!(scratch, weights, timestep,
        time_embedding, modulation)
      validate_weight_aliases!(weights, timestep, time_embedding, modulation)
      {"time_embedding.0.weight"  => cfg.dim.to_i64 * cfg.freq_dim,
       "time_embedding.0.bias"    => cfg.dim.to_i64,
       "time_embedding.2.weight"  => cfg.dim.to_i64 * cfg.dim,
       "time_embedding.2.bias"    => cfg.dim.to_i64,
       "time_projection.1.weight" => cfg.dim.to_i64 * 6 * cfg.dim,
       "time_projection.1.bias"   => cfg.dim.to_i64 * 6}.each do |name, count|
        weights.check_weight("model.#{name}", count)
      end

      e.set_pipeline(pipeline("wan_s2v_sinusoidal"))
      e.set_buffer(timestep, 0).set_buffer(scratch[time_frequency_hi_name(cfg)], 1)
      e.set_buffer(scratch[time_frequency_lo_name(cfg)], 2)
      e.set_buffer(scratch["wan_t_time_sinusoid"], 3, ML::Metal::BufferAccess::Write)
      e.set_value(cfg.batches.to_u32, 4).set_value(cfg.freq_dim.to_u32, 5)
      e.dispatch_1d(time_rows * cfg.freq_dim)

      linear(e, scratch["wan_t_time_sinusoid"], weights, "model.time_embedding.0",
        scratch["wan_t_time_hidden"], time_rows, cfg.freq_dim, cfg.dim)
      silu(e, scratch["wan_t_time_hidden"], scratch["wan_t_time_hidden"], time_rows * cfg.dim)
      linear(e, scratch["wan_t_time_hidden"], weights, "model.time_embedding.2",
        scratch["wan_t_time_embedding_all"], time_rows, cfg.dim, cfg.dim)
      MetalOps.copy(e, scratch["wan_t_time_embedding_all"],
        scratch["wan_t_time_projection_input"], time_rows * cfg.dim)
      silu(e, scratch["wan_t_time_projection_input"], scratch["wan_t_time_projection_input"], time_rows * cfg.dim)
      linear(e, scratch["wan_t_time_projection_input"], weights, "model.time_projection.1",
        scratch["wan_t_time_projection_all"], time_rows, cfg.dim, cfg.dim * 6)

      e.set_pipeline(pipeline("wan_s2v_time_expand"))
      e.set_buffer(scratch["wan_t_time_embedding_all"], 0)
      e.set_buffer(scratch["wan_t_time_projection_all"], 1)
      e.set_buffer(time_embedding, 2, ML::Metal::BufferAccess::Write)
      e.set_buffer(modulation, 3, ML::Metal::BufferAccess::Write)
      e.set_value(cfg.batches.to_u32, 4).set_value(cfg.dim.to_u32, 5)
      e.dispatch_1d(output_modulation.to_i32)
    end

    # Apply the upstream S2V prediction head to only the noisy-token prefix.
    # `time_embedding` is the nonzero-timestep [B,C] output from encode_time.
    def self.encode_head(e : ML::Metal::ComputeEncoder, hidden : ML::MetalBuffer,
                         time_embedding : ML::MetalBuffer, prediction : ML::MetalBuffer,
                         latent_output : ML::MetalBuffer, weights : Arena,
                         cfg : WanS2VTransformerConfig,
                         output_frames : Int32, latent_height : Int32,
                         latent_width : Int32) : Nil
      prefix_count = cfg.batches.to_i64 * cfg.video_tokens * cfg.dim
      rows = cfg.batches * cfg.video_tokens
      output_values = cfg.batches.to_i64 * cfg.video_tokens * cfg.out_dim * 4
      MetalOps.check(hidden, cfg.rows.to_i64 * cfg.dim)
      MetalOps.check(time_embedding, cfg.batches.to_i64 * cfg.dim)
      MetalOps.check(prediction, output_values)
      latent_values = cfg.batches.to_i64 * cfg.out_dim * output_frames * latent_height * latent_width
      MetalOps.check(latent_output, latent_values)
      validate_weight_aliases!(weights, hidden, time_embedding, prediction,
        latent_output)
      MetalOps.distinct(prediction, hidden, time_embedding, latent_output)
      MetalOps.distinct(latent_output, hidden, time_embedding)
      unless output_frames == cfg.video_frames && latent_height == cfg.latent_height &&
             latent_width == cfg.latent_width
        raise ArgumentError.new("Wan S2V unpatchify geometry does not match the noisy token grid")
      end
      raise ArgumentError.new("Wan S2V latent dimensions must be even") if latent_height.odd? || latent_width.odd?
      unless output_frames.to_i64 * (latent_height // 2) * (latent_width // 2) == cfg.video_tokens
        raise ArgumentError.new("Wan S2V latent grid does not match noisy token count")
      end
      # The head norm is non-affine WanLayerNorm. Reuse the consumed hidden
      # prefix in place, then apply Head_S2V's learned shift/scale table.
      weights.check_weight("model.head.modulation", 2_i64 * cfg.dim)
      weights.check_weight("model.head.head.weight", cfg.out_dim.to_i64 * 4 * cfg.dim)
      weights.check_weight("model.head.head.bias", cfg.out_dim.to_i64 * 4)
      head_modulation = weights["model.head.modulation"]
      head_weight = weights["model.head.head.weight"]
      head_bias = weights["model.head.head.bias"]
      MetalOps.distinct(hidden, time_embedding, head_modulation)
      MetalOps.distinct(prediction, hidden, time_embedding, head_modulation, head_weight, head_bias)
      MetalOps.distinct(latent_output, hidden, time_embedding, prediction,
        head_modulation, head_weight, head_bias)
      e.set_pipeline(pipeline("wan_s2v_head_norm"))
      e.set_buffer(hidden, 0, ML::Metal::BufferAccess::ReadWrite)
      e.set_value(cfg.batches.to_u32, 1).set_value(cfg.sequence_tokens.to_u32, 2)
      e.set_value(cfg.video_tokens.to_u32, 3).set_value(cfg.dim.to_u32, 4)
      e.set_value(S2V_EPS, 5)
      threads = Math.min(256, ((cfg.dim + 31) // 32) * 32)
      e.dispatch_threadgroups({(cfg.batches * cfg.video_tokens).to_i32, 1, 1}, {threads, 1, 1})
      e.set_pipeline(pipeline("wan_s2v_head_modulate"))
      e.set_buffer(hidden, 0, ML::Metal::BufferAccess::ReadWrite).set_buffer(time_embedding, 1)
      e.set_buffer(weights["model.head.modulation"], 2)
      e.set_buffer(weights["model.head.modulation"], 3)
      e.set_value(cfg.sequence_tokens.to_u32, 4).set_value(cfg.video_tokens.to_u32, 5)
      e.set_value(cfg.dim.to_u32, 6)
      e.set_value(weights.dtype("model.head.modulation").bf16? ? 1_u32 : 0_u32, 7)
      e.set_value(prefix_count.to_u32, 8)
      e.dispatch_1d(prefix_count.to_i32)

      e.set_pipeline(pipeline("wan_s2v_head_linear"))
      e.set_buffer(hidden, 0).set_buffer(weights["model.head.head.weight"], 1)
      e.set_buffer(weights["model.head.head.bias"], 2)
      e.set_buffer(prediction, 3, ML::Metal::BufferAccess::Write)
      e.set_value(cfg.batches.to_u32, 4).set_value(cfg.sequence_tokens.to_u32, 5)
      e.set_value(cfg.video_tokens.to_u32, 6).set_value(cfg.dim.to_u32, 7)
      e.set_value((cfg.out_dim * 4).to_u32, 8)
      e.set_buffer(weights["model.head.head.weight"], 9)
      e.set_buffer(weights["model.head.head.bias"], 10)
      e.set_value(weights.dtype("model.head.head.weight").bf16? ? 1_u32 : 0_u32, 11)
      e.set_value(weights.dtype("model.head.head.bias").bf16? ? 1_u32 : 0_u32, 12)
      e.dispatch_1d(output_values.to_i32)
      e.set_pipeline(pipeline("wan_s2v_unpatchify"))
      e.set_buffer(prediction, 0).set_buffer(latent_output, 1, ML::Metal::BufferAccess::Write)
      e.set_value(cfg.batches.to_u32, 2).set_value(output_frames.to_u32, 3)
      e.set_value((latent_height // 2).to_u32, 4).set_value((latent_width // 2).to_u32, 5)
      e.set_value(cfg.out_dim.to_u32, 6)
      e.set_value(latent_height.to_u32, 7).set_value(latent_width.to_u32, 8)
      e.dispatch_1d(latent_values.to_i32)
    end

    # Embed raw noisy, condition, reference, and FramePack latents into packed
    # [B,S,C]. Inputs are contiguous [B,in_dim,T,H,W] F32 resident buffers.
    # Motion is right-aligned in upstream's 19-frame FramePack window.
    def self.encode_patch_inputs(e : ML::Metal::ComputeEncoder,
                                 video_latents : ML::MetalBuffer,
                                 condition_latents : ML::MetalBuffer,
                                 reference_latents : ML::MetalBuffer,
                                 motion_latents : ML::MetalBuffer,
                                 sequence : ML::MetalBuffer,
                                 weights : Arena, scratch : Arena,
                                 cfg : WanS2VTransformerConfig,
                                 motion_frames : Int32) : Nil
      cfg.validate!
      raise ArgumentError.new("Wan S2V FramePack input must contain 1..19 frames") unless motion_frames > 0 && motion_frames <= 19
      sequence_values = cfg.rows.to_i64 * cfg.dim
      video_values = cfg.batches.to_i64 * cfg.in_dim * cfg.video_frames * cfg.latent_height * cfg.latent_width
      reference_values = cfg.batches.to_i64 * cfg.in_dim * cfg.latent_height * cfg.latent_width
      motion_values = cfg.batches.to_i64 * cfg.in_dim * motion_frames * cfg.latent_height * cfg.latent_width
      MetalOps.check(video_latents, video_values)
      MetalOps.check(condition_latents, video_values)
      MetalOps.check(reference_latents, reference_values)
      MetalOps.check(motion_latents, motion_values)
      MetalOps.check(sequence, sequence_values)
      MetalOps.distinct(sequence, video_latents, condition_latents, reference_latents, motion_latents)

      patch_input = scratch["wan_t_patch_input"]
      patch_projected = scratch["wan_t_patch_projected"]
      patch_features = cfg.in_dim.to_i64 * 256
      MetalOps.check(patch_input, cfg.patch_tile_rows.to_i64 * patch_features)
      MetalOps.check(patch_projected, cfg.patch_tile_rows.to_i64 * cfg.dim)
      MetalOps.distinct(patch_input, patch_projected, sequence, video_latents,
        condition_latents, reference_latents, motion_latents)
      MetalOps.distinct(patch_projected, sequence, video_latents,
        condition_latents, reference_latents, motion_latents)
      validate_transformer_scratch_aliases!(scratch, weights, video_latents,
        condition_latents, reference_latents, motion_latents, sequence)
      validate_weight_aliases!(weights, video_latents, condition_latents,
        reference_latents, motion_latents, sequence)
      {
        "model.patch_embedding.weight"      => cfg.dim.to_i64 * patch_features // 64,
        "model.patch_embedding.bias"        => cfg.dim.to_i64,
        "model.cond_encoder.weight"         => cfg.dim.to_i64 * cfg.in_dim * 4,
        "model.cond_encoder.bias"           => cfg.dim.to_i64,
        "model.frame_packer.proj.weight"    => cfg.dim.to_i64 * cfg.in_dim * 4,
        "model.frame_packer.proj.bias"      => cfg.dim.to_i64,
        "model.frame_packer.proj_2x.weight" => cfg.dim.to_i64 * cfg.in_dim * 32,
        "model.frame_packer.proj_2x.bias"   => cfg.dim.to_i64,
        "model.frame_packer.proj_4x.weight" => cfg.dim.to_i64 * cfg.in_dim * 256,
        "model.frame_packer.proj_4x.bias"   => cfg.dim.to_i64,
        "model.trainable_cond_mask.weight"  => 3_i64 * cfg.dim,
      }.each { |name, count| weights.check_weight(name, count) }
      class_weight = weights["model.trainable_cond_mask.weight"]
      class_dtype = weights.dtype("model.trainable_cond_mask.weight")
      encode_patch_segment(e, video_latents, sequence, patch_input, patch_projected,
        weights, class_weight, class_dtype, cfg, "model.patch_embedding",
        cfg.video_frames, cfg.video_frames, cfg.video_frames,
        cfg.latent_height // 2, cfg.latent_width // 2, 1, 2, 2,
        0, 0, false, 0)
      encode_patch_segment(e, reference_latents, sequence, patch_input, patch_projected,
        weights, class_weight, class_dtype, cfg, "model.patch_embedding",
        1, 1, 1, cfg.latent_height // 2, cfg.latent_width // 2,
        1, 2, 2, 0, cfg.video_tokens, false, 1)

      motion_offset = cfg.video_tokens + cfg.reference_tokens
      unless cfg.drop_motion_frames
        encode_patch_segment(e, motion_latents, sequence, patch_input, patch_projected,
          weights, class_weight, class_dtype, cfg, "model.frame_packer.proj",
          motion_frames, 19, 1, cfg.latent_height // 2, cfg.latent_width // 2,
          1, 2, 2, 18, motion_offset, false, 2)
        encode_patch_segment(e, motion_latents, sequence, patch_input, patch_projected,
          weights, class_weight, class_dtype, cfg, "model.frame_packer.proj_2x",
          motion_frames, 19, 1, cfg.latent_height // 4, cfg.latent_width // 4,
          2, 4, 4, 8, motion_offset + cfg.reference_tokens, false, 2)
        middle_tokens = (cfg.latent_height // 4) * (cfg.latent_width // 4)
        encode_patch_segment(e, motion_latents, sequence, patch_input, patch_projected,
          weights, class_weight, class_dtype, cfg, "model.frame_packer.proj_4x",
          motion_frames, 19, 4, cfg.latent_height // 8, cfg.latent_width // 8,
          4, 8, 8, 0, motion_offset + cfg.reference_tokens + middle_tokens,
          false, 2)
      end
      encode_patch_segment(e, condition_latents, sequence, patch_input, patch_projected,
        weights, class_weight, class_dtype, cfg, "model.cond_encoder",
        cfg.video_frames, cfg.video_frames, cfg.video_frames,
        cfg.latent_height // 2, cfg.latent_width // 2, 1, 2, 2,
        0, 0, true, 3)
    end

    # Integer-valued RoPE metadata is setup input, not a neural activation.
    # np.linspace(...).astype(int) in upstream is reproduced with exact integer
    # floor division, including the coarse FramePack spatial coordinates.
    def self.position_metadata(cfg : WanS2VTransformerConfig) : Array(Float32)
      sample_values = cfg.sequence_tokens * 3
      values = Array(Float32).new(cfg.batches * sample_values, 0.0_f32)
      token = 0
      half_h, half_w = cfg.latent_height // 2, cfg.latent_width // 2
      cfg.video_frames.times do |frame|
        half_h.times do |h|
          half_w.times do |w|
            write_position(values, token, frame, h, w)
            token += 1
          end
        end
      end
      half_h.times do |h|
        half_w.times do |w|
          write_position(values, token, 30, h, w)
          token += 1
        end
      end
      unless cfg.drop_motion_frames
        half_h.times do |h|
          half_w.times do |w|
            write_position(values, token, -1, h, w)
            token += 1
          end
        end
        sample_indices(half_h, cfg.latent_height // 4).each do |h|
          sample_indices(half_w, cfg.latent_width // 4).each do |w|
            write_position(values, token, -3, h, w)
            token += 1
          end
        end
        far_h = sample_indices(half_h, cfg.latent_height // 8)
        far_w = sample_indices(half_w, cfg.latent_width // 8)
        {-19, -14, -9, -4}.each do |frame|
          far_h.each do |h|
            far_w.each do |w|
              write_position(values, token, frame, h, w)
              token += 1
            end
          end
        end
      end
      raise ArgumentError.new("Wan S2V position metadata has the wrong sequence length") unless token == cfg.sequence_tokens
      cfg.batches.times do |batch|
        next if batch == 0
        source = values[0, sample_values]
        source.each_with_index { |value, index| values[batch * sample_values + index] = value }
      end
      values
    end

    private def self.pipeline(name : String) : ML::Metal::ComputePipeline
      ML::Metal::PipelineCache.get(name) { ML::Metal::ComputePipeline.new(name, SOURCE) }
    end

    private def self.encode_patch_segment(e : ML::Metal::ComputeEncoder,
                                          latent : ML::MetalBuffer,
                                          sequence : ML::MetalBuffer,
                                          patch_input : ML::MetalBuffer,
                                          patch_projected : ML::MetalBuffer,
                                          weights : Arena, class_weight : ML::MetalBuffer,
                                          class_dtype : WeightDType,
                                          cfg : WanS2VTransformerConfig,
                                          prefix : String, source_frames : Int32,
                                          virtual_frames : Int32, output_frames : Int32,
                                          grid_h : Int32, grid_w : Int32,
                                          patch_t : Int32, patch_h : Int32, patch_w : Int32,
                                          source_frame_start : Int32, token_offset : Int32,
                                          add_to_sequence : Bool, class_id : Int32) : Nil
      rows = output_frames * grid_h * grid_w
      features = cfg.in_dim * patch_t * patch_h * patch_w
      weight_name = "#{prefix}.weight"
      bias_name = "#{prefix}.bias"
      weights.check_weight(weight_name, cfg.dim.to_i64 * features)
      weights.check_weight(bias_name, cfg.dim)
      current = 0
      cfg.batches.times do |batch|
        while current < rows
          tile_rows = Math.min(cfg.patch_tile_rows, rows - current)
          e.set_pipeline(pipeline("wan_s2v_patchify"))
          e.set_buffer(latent, 0).set_buffer(patch_input, 1, ML::Metal::BufferAccess::Write)
          e.set_value(cfg.in_dim.to_u32, 2).set_value(source_frames.to_u32, 3)
          e.set_value(cfg.latent_height.to_u32, 4).set_value(cfg.latent_width.to_u32, 5)
          e.set_value(virtual_frames.to_u32, 6).set_value(patch_t.to_u32, 7)
          e.set_value(patch_h.to_u32, 8).set_value(patch_w.to_u32, 9)
          e.set_value(grid_h.to_u32, 10).set_value(grid_w.to_u32, 11)
          e.set_value(source_frame_start.to_u32, 12).set_value(current.to_u32, 13)
          e.set_value(tile_rows.to_u32, 14).set_value(batch.to_u32, 15)
          e.dispatch_1d(tile_rows * features)
          MetalOps.linear(e, patch_input, weights[weight_name], weights[bias_name],
            patch_projected, tile_rows, features, cfg.dim,
            weight_dtype: weights.dtype(weight_name), bias_dtype: weights.dtype(bias_name))
          e.set_pipeline(pipeline("wan_s2v_scatter_tokens"))
          e.set_buffer(patch_projected, 0).set_buffer(class_weight, 1)
          e.set_buffer(class_weight, 2).set_buffer(sequence, 3,
            add_to_sequence ? ML::Metal::BufferAccess::ReadWrite : ML::Metal::BufferAccess::Write)
          e.set_value(batch.to_u32, 4).set_value(cfg.sequence_tokens.to_u32, 5)
          e.set_value(cfg.dim.to_u32, 6).set_value(token_offset.to_u32, 7)
          e.set_value(current.to_u32, 8).set_value(tile_rows.to_u32, 9)
          e.set_value(class_id.to_u32, 10).set_value(add_to_sequence ? 1_u32 : 0_u32, 11)
          e.set_value(class_dtype.bf16? ? 1_u32 : 0_u32, 12)
          e.dispatch_1d(tile_rows * cfg.dim)
          current += tile_rows
        end
      end
    end

    private def self.write_position(values : Array(Float32), token : Int32,
                                    frame : Int32, height : Int32, width : Int32) : Nil
      at = token * 3
      values[at] = frame.to_f32
      values[at + 1] = height.to_f32
      values[at + 2] = width.to_f32
    end

    private def self.sample_indices(extent : Int32, count : Int32) : Array(Int32)
      raise ArgumentError.new("Wan S2V RoPE sample grid must be positive") unless extent > 0 && count > 0
      indices = Array(Int32).new(count, 0)
      return indices if count == 1
      (0...count).each do |index|
        indices[index] = ((extent - 1).to_i64 * index // (count - 1)).to_i32
      end
      indices
    end

    private def self.time_frequency_hi_name(cfg : WanS2VTransformerConfig) : String
      "wan_t_time_freq_hi_#{cfg.freq_dim}"
    end

    private def self.time_frequency_lo_name(cfg : WanS2VTransformerConfig) : String
      "wan_t_time_freq_lo_#{cfg.freq_dim}"
    end

    private def self.validate_transformer_scratch!(scratch : Arena,
                                                   cfg : WanS2VTransformerConfig) : Nil
      scratch_plan = transformer_scratch_plan(cfg)
      scratch_plan.each { |name, count| MetalOps.check(scratch[name], count) }
      MetalOps.check(scratch[time_frequency_hi_name(cfg)], cfg.freq_dim // 2)
      MetalOps.check(scratch[time_frequency_lo_name(cfg)], cfg.freq_dim // 2)
    end

    private def self.validate_transformer_scratch_aliases!(scratch : Arena,
                                                           weights : Arena,
                                                           *inputs : ML::MetalBuffer) : Nil
      scratch.buffers.each do |name, buffer|
        inputs.each do |input|
          if MetalOps.overlap?(buffer, input)
            raise ArgumentError.new("Wan S2V scratch buffer #{name} aliases a live input/output")
          end
        end
        scratch.buffers.each do |other_name, other|
          if name < other_name && MetalOps.overlap?(buffer, other)
            raise ArgumentError.new("Wan S2V transformer scratch buffers overlap")
          end
        end
        weights.buffers.each do |weight_name, weight|
          if MetalOps.overlap?(buffer, weight)
            raise ArgumentError.new("Wan S2V transformer scratch buffer #{name} aliases weight #{weight_name}")
          end
        end
      end
    end

    private def self.validate_weight_aliases!(weights : Arena,
                                              *buffers : ML::MetalBuffer) : Nil
      weights.buffers.each do |name, weight|
        buffers.each do |buffer|
          if MetalOps.overlap?(weight, buffer)
            raise ArgumentError.new("Wan S2V buffer aliases weight #{name}")
          end
        end
      end
    end

    private def self.linear(e : ML::Metal::ComputeEncoder, x : ML::MetalBuffer,
                            weights : Arena, prefix : String, y : ML::MetalBuffer,
                            rows : Int32, in_dim : Int32, out_dim : Int32) : Nil
      weight_name = "#{prefix}.weight"
      bias_name = "#{prefix}.bias"
      MetalOps.linear(e, x, weights[weight_name], weights[bias_name], y,
        rows, in_dim, out_dim, weight_dtype: weights.dtype(weight_name),
        bias_dtype: weights.dtype(bias_name))
    end

    private def self.silu(e : ML::Metal::ComputeEncoder, x : ML::MetalBuffer,
                          y : ML::MetalBuffer, count : Int32) : Nil
      MetalOps.check(x, count.to_i64)
      MetalOps.check(y, count.to_i64)
      MetalOps.inplace_or_distinct(x, y)
      e.set_pipeline(pipeline("wan_s2v_silu"))
      e.set_buffer(x, 0, ML::Metal::BufferAccess::ReadWrite)
      e.set_buffer(y, 1, ML::Metal::BufferAccess::ReadWrite)
      e.set_value(count.to_u32, 2)
      e.dispatch_1d(count)
    end

    private def self.add_extra_bindings(result : Array(WeightBinding),
                                        cfg : WanS2VTransformerConfig,
                                        dtype : ML::GGUF::SafeTensorDType) : Nil
      dim = cfg.dim.to_i64
      text_dim = cfg.text_dim.to_i64
      freq = cfg.freq_dim.to_i64
      add = ->(source : String, shape : Array(Int64)) do
        result << WeightBinding.new("model.#{source}", source, shape, dtype)
      end
      add.call("patch_embedding.weight", [dim, cfg.in_dim.to_i64, 1_i64, 2_i64, 2_i64])
      add.call("patch_embedding.bias", [dim])
      add.call("text_embedding.0.weight", [dim, text_dim])
      add.call("text_embedding.0.bias", [dim])
      add.call("text_embedding.2.weight", [dim, dim])
      add.call("text_embedding.2.bias", [dim])
      add.call("time_embedding.0.weight", [dim, freq])
      add.call("time_embedding.0.bias", [dim])
      add.call("time_embedding.2.weight", [dim, dim])
      add.call("time_embedding.2.bias", [dim])
      add.call("time_projection.1.weight", [6_i64 * dim, dim])
      add.call("time_projection.1.bias", [6_i64 * dim])
      add.call("cond_encoder.weight", [dim, cfg.in_dim.to_i64, 1_i64, 2_i64, 2_i64])
      add.call("cond_encoder.bias", [dim])
      add.call("head.head.weight", [cfg.out_dim.to_i64 * 4, dim])
      add.call("head.head.bias", [cfg.out_dim.to_i64 * 4])
      add.call("head.modulation", [1_i64, 2_i64, dim])
      add.call("frame_packer.proj.weight", [dim, cfg.in_dim.to_i64, 1_i64, 2_i64, 2_i64])
      add.call("frame_packer.proj.bias", [dim])
      add.call("frame_packer.proj_2x.weight", [dim, cfg.in_dim.to_i64, 2_i64, 4_i64, 4_i64])
      add.call("frame_packer.proj_2x.bias", [dim])
      add.call("frame_packer.proj_4x.weight", [dim, cfg.in_dim.to_i64, 4_i64, 8_i64, 8_i64])
      add.call("frame_packer.proj_4x.bias", [dim])
      add.call("trainable_cond_mask.weight", [3_i64, dim])
    end
  end
end
