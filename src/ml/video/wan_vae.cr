require "./metal_ops"
require "./checkpoint"

module ML::Video
  # Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
  # Apache-2.0. Native F32 Metal execution adapted from Wan2.1_VAE at
  # Wan2.2 commit 1ea34ff48f87168174e12956e200b1d908b1c5ff.

  private module WanVAEShape
    def self.count(label : String, *factors : Int32) : Int32
      result = 1_i64
      factors.each do |factor|
        unless factor > 0 && result <= Int32::MAX.to_i64 // factor
          raise ArgumentError.new("Wan VAE #{label} exceeds supported element count")
        end
        result *= factor
      end
      result.to_i32
    end

    def self.bytes(label : String, count : Int64) : Int64
      unless count >= 0 && count <= Int64::MAX // sizeof(Float32)
        raise ArgumentError.new("Wan VAE #{label} byte count overflows")
      end
      count * sizeof(Float32)
    end

    def self.add_bytes(label : String, total : Int64, additional : Int64) : Int64
      unless total >= 0 && additional >= 0 && additional <= Int64::MAX - total
        raise ArgumentError.new("Wan VAE #{label} byte count overflows")
      end
      total + additional
    end
  end

  struct WanVAEConfig
    getter batches : Int32
    getter height : Int32
    getter width : Int32
    getter dim : Int32
    getter z_dim : Int32
    @dim_mult : Array(Int32)
    getter num_res_blocks : Int32
    @temporal_downsample : Array(Bool)

    def initialize(@batches : Int32, @height : Int32, @width : Int32,
                   @dim : Int32 = 96, @z_dim : Int32 = 16,
                   @dim_mult : Array(Int32) = [1, 2, 4, 4],
                   @num_res_blocks : Int32 = 2,
                   @temporal_downsample : Array(Bool) = [false, true, true])
      @dim_mult = @dim_mult.dup
      @temporal_downsample = @temporal_downsample.dup
      unless @batches > 0 && @height > 0 && @width > 0 && @dim > 0 && @z_dim > 0
        raise ArgumentError.new("Wan VAE dimensions must be positive")
      end
      unless @height % 8 == 0 && @width % 8 == 0
        raise ArgumentError.new("Wan VAE image dimensions must be divisible by eight")
      end
      unless @dim_mult.size == 4 && @dim_mult.all? { |value| value > 0 }
        raise ArgumentError.new("Wan VAE requires four positive dim_mult entries")
      end
      unless @dim_mult == [1, 2, 4, 4]
        raise ArgumentError.new("Wan VAE currently supports the pinned [1, 2, 4, 4] channel geometry")
      end
      unless @dim <= 512 // @dim_mult.max
        raise ArgumentError.new("Wan VAE attention channels exceed the supported threadgroup width")
      end
      unless @temporal_downsample.size == @dim_mult.size - 1
        raise ArgumentError.new("Wan VAE temporal_downsample must match the resample count")
      end
      unless @temporal_downsample == [false, true, true]
        raise ArgumentError.new("Wan VAE currently supports the pinned temporal downsample geometry")
      end
      unless @num_res_blocks >= 0
        raise ArgumentError.new("Wan VAE residual-block count cannot be negative")
      end
      WanVAEShape.count("image input", @batches, 3, 1, @height, @width)
      WanVAEShape.count("latent image", @batches, @z_dim, 1, latent_height, latent_width)
      WanVAEShape.count("decoder chunk", @batches, temporal_factor, output_channels, @height, @width)
    end

    def dim_mult : Array(Int32)
      @dim_mult.dup
    end

    def temporal_downsample : Array(Bool)
      @temporal_downsample.dup
    end

    def latent_height : Int32
      @height // 8
    end

    def latent_width : Int32
      @width // 8
    end

    def output_channels : Int32
      @dim * @dim_mult.first
    end

    def temporal_levels : Int32
      @temporal_downsample.count(true)
    end

    def temporal_factor : Int32
      1 << temporal_levels
    end

    def decoded_frames(latent_frames : Int32) : Int32
      raise ArgumentError.new("Wan VAE latent frame count must be positive") unless latent_frames > 0
      WanVAEShape.count("decoded frames", 1 + (latent_frames - 1) * temporal_factor)
    end

    def decoder_dims : Array(Int32)
      [@dim * @dim_mult.last] + @dim_mult.reverse.map { |value| @dim * value }
    end

    def encoder_dims : Array(Int32)
      [@dim] + @dim_mult.map { |value| @dim * value }
    end
  end

  private class WanVAECachedState
    getter first : ML::MetalBuffer
    getter second : ML::MetalBuffer
    getter frame_plane : Int32
    property active_slot : Int32
    property history : Int32
    property replicate : Bool

    def initialize(@first : ML::MetalBuffer, @second : ML::MetalBuffer,
                   @frame_plane : Int32)
      @active_slot = 0
      @history = 0
      @replicate = false
    end

    def current : ML::MetalBuffer
      @active_slot == 0 ? @first : @second
    end

    def spare : ML::MetalBuffer
      @active_slot == 0 ? @second : @first
    end

    def advance!(history : Int32, replicate : Bool = false) : Nil
      @active_slot = 1 - @active_slot
      @history = history
      @replicate = replicate
    end

    def reset! : Nil
      @active_slot = 0
      @history = 0
      @replicate = false
    end
  end

  # Owns preallocated F32 activation scratch and double-buffered decoder
  # histories. Build once per serial request shape; reset only after the
  # previous encoder has completed.
  class WanVAEWorkspace
    getter arena : Arena
    getter config : WanVAEConfig
    getter scratch_values : Int32
    getter cache_values : Int64
    getter planned_bytes : Int64
    getter max_latent_frames : Int32
    @cache_states = {} of String => WanVAECachedState

    def initialize(@arena : Arena, @config : WanVAEConfig,
                   @scratch_values : Int32, @cache_values : Int64,
                   @planned_bytes : Int64, @max_latent_frames : Int32)
    end

    def buffer(name : String) : ML::MetalBuffer
      @arena[name]
    end

    def cache(name : String) : WanVAECachedState
      @cache_states[name]
    end

    def reset_cache_state! : Nil
      @cache_states.each_value(&.reset!)
    end

    def cache_names : Array(String)
      @cache_states.keys
    end

    def temp(except_a : ML::MetalBuffer? = nil,
             except_b : ML::MetalBuffer? = nil,
             except_c : ML::MetalBuffer? = nil) : ML::MetalBuffer
      {"wan_vae_tmp_0", "wan_vae_tmp_1", "wan_vae_tmp_2", "wan_vae_tmp_3"}.each do |name|
        candidate = @arena[name]
        next if except_a && candidate.same?(except_a)
        next if except_b && candidate.same?(except_b)
        next if except_c && candidate.same?(except_c)
        return candidate
      end
      raise ArgumentError.new("Wan VAE scratch alias exhaustion")
    end
  end

  module WanVAE
    SOURCE     = {{ read_file("#{__DIR__}/kernels/wan_vae.metal") }}
    SCALE_MEAN = [-0.7571_f32, -0.7089_f32, -0.9113_f32, 0.1075_f32,
                  -0.1745_f32, 0.9653_f32, -0.1517_f32, 1.5508_f32,
                  0.4134_f32, -0.0715_f32, 0.5517_f32, -0.3632_f32,
                  -0.1922_f32, -0.9497_f32, 0.2503_f32, -0.2921_f32]
    SCALE_INV_STD = [0.354811251_f32, 0.687710583_f32, 0.429645509_f32, 0.376534373_f32,
                     0.819940984_f32, 0.564716518_f32, 0.383847684_f32, 0.482090324_f32,
                     0.305932045_f32, 0.464554489_f32, 0.349015772_f32, 0.641889751_f32,
                     0.610426068_f32, 0.888651848_f32, 0.353969783_f32, 0.521920681_f32]

    private record CacheSpec, name : String, channels : Int32, height : Int32, width : Int32
    private record Activation, buffer : ML::MetalBuffer, batches : Int32, channels : Int32,
      frames : Int32, height : Int32, width : Int32

    def self.pipeline(name : String) : ML::Metal::ComputePipeline
      ML::Metal::PipelineCache.get(name) { ML::Metal::ComputePipeline.new(name, SOURCE) }
    end

    def self.prepare! : Nil
      MetalOps.prepare!
      {"wan_vae_conv3d_f32", "wan_vae_rms_norm_f32", "wan_vae_silu_f32",
       "wan_vae_add_f32", "wan_vae_cache_last2_f32", "wan_vae_nearest2x_f32",
       "wan_vae_temporal_shuffle_f32", "wan_vae_attention_f32", "wan_vae_scale_f32",
       "wan_vae_split_scale_f32", "wan_vae_copy_clamp_chunk_f32",
       "wan_vae_time_slice_f32", "wan_vae_copy_frames_f32",
       "wan_vae_crop_frames_f32"}.each { |name| pipeline(name) }
    end

    # Exact official state-dict names are the Arena keys. All learned VAE
    # tensors, including residual norms and Resample submodules, are included.
    def self.checkpoint_bindings(cfg : WanVAEConfig,
                                 dtype : ML::GGUF::SafeTensorDType? = nil) : Array(WeightBinding)
      result = [] of WeightBinding
      dims = cfg.encoder_dims
      add_conv3d(result, "encoder.conv1", 3, dims[0], 3, 3, 3)
      index = 0
      (0...cfg.dim_mult.size).each do |stage|
        in_channels = dims[stage]
        out_channels = dims[stage + 1]
        cfg.num_res_blocks.times do |block|
          prefix = "encoder.downsamples.#{index}"
          add_residual_bindings(result, prefix, in_channels, out_channels)
          in_channels = out_channels
          index += 1
        end
        if stage < cfg.dim_mult.size - 1
          prefix = "encoder.downsamples.#{index}"
          add_conv2d(result, "#{prefix}.resample.1", out_channels, out_channels, 3, 3)
          if cfg.temporal_downsample[stage]
            add_conv3d(result, "#{prefix}.time_conv", out_channels, out_channels, 3, 1, 1)
          end
          index += 1
        end
      end
      add_residual_bindings(result, "encoder.middle.0", dims.last, dims.last)
      add_attention_bindings(result, "encoder.middle.1", dims.last)
      add_residual_bindings(result, "encoder.middle.2", dims.last, dims.last)
      add_norm_binding(result, "encoder.head.0.gamma", dims.last, 4)
      add_conv3d(result, "encoder.head.2", dims.last, cfg.z_dim * 2, 3, 3, 3)

      decoder_dims = cfg.decoder_dims
      add_conv3d(result, "decoder.conv1", cfg.z_dim, decoder_dims.first, 3, 3, 3)
      add_residual_bindings(result, "decoder.middle.0", decoder_dims.first, decoder_dims.first)
      add_attention_bindings(result, "decoder.middle.1", decoder_dims.first)
      add_residual_bindings(result, "decoder.middle.2", decoder_dims.first, decoder_dims.first)
      index = 0
      current_height = cfg.latent_height
      current_width = cfg.latent_width
      (0...cfg.dim_mult.size).each do |stage|
        source_in = decoder_dims[stage]
        out_channels = decoder_dims[stage + 1]
        in_channels = stage >= 1 ? source_in // 2 : source_in
        (cfg.num_res_blocks + 1).times do |block|
          prefix = "decoder.upsamples.#{index}"
          add_residual_bindings(result, prefix, in_channels, out_channels)
          in_channels = out_channels
          index += 1
        end
        if stage < cfg.dim_mult.size - 1
          prefix = "decoder.upsamples.#{index}"
          if cfg.temporal_downsample.reverse[stage]
            add_conv3d(result, "#{prefix}.time_conv", out_channels, out_channels * 2, 3, 1, 1)
          end
          add_conv2d(result, "#{prefix}.resample.1", out_channels, out_channels // 2, 3, 3)
          index += 1
          current_height *= 2
          current_width *= 2
        end
      end
      final_dim = decoder_dims.last
      add_norm_binding(result, "decoder.head.0.gamma", final_dim, 4)
      add_conv3d(result, "decoder.head.2", final_dim, 3, 3, 3, 3)
      add_conv3d(result, "conv1", cfg.z_dim * 2, cfg.z_dim * 2, 1, 1, 1)
      add_conv3d(result, "conv2", cfg.z_dim, cfg.z_dim, 1, 1, 1)
      if dtype
        result.each_with_index do |binding, i|
          result[i] = WeightBinding.new(binding.target, binding.source, binding.shape, dtype)
        end
      end
      result
    end

    # Plan cache/scratch/output bytes before allocating. Both budget and reserve
    # are mandatory request inputs: reserve must include concurrent residency
    # and any checkpoint staging held outside `weights`.
    def self.prepare!(workspace_arena : Arena, weights : Arena, cfg : WanVAEConfig,
                      budget_bytes : Int64, reserve_bytes : Int64,
                      max_latent_frames : Int32) : WanVAEWorkspace
      raise ArgumentError.new("Wan VAE workspace must start empty") unless workspace_arena.resident_bytes == 0
      raise ArgumentError.new("Invalid Wan VAE residency budget") unless budget_bytes > 0 && reserve_bytes >= 0
      raise ArgumentError.new("Wan VAE needs a positive latent frame count") unless max_latent_frames > 0
      specs = encoder_cache_specs(cfg) + decoder_cache_specs(cfg)
      chunk_frames = cfg.temporal_factor
      activation_count = WanVAEShape.count("activation scratch", cfg.batches, chunk_frames,
        cfg.output_channels, cfg.height, cfg.width)
      cache_bytes = specs.reduce(0_i64) do |total, spec|
        # Each cache has two frames and two buffers so a queued update can
        # alternate without mutating the history still read by the command.
        frame_plane = WanVAEShape.count("cache frame", cfg.batches,
          spec.channels, spec.height, spec.width)
        cache_values = WanVAEShape.count("double-buffered cache", 4, frame_plane)
        bytes = WanVAEShape.bytes("double-buffered cache", cache_values.to_i64)
        WanVAEShape.add_bytes("cache", total, bytes)
      end
      scratch_bytes = WanVAEShape.bytes("activation scratch", activation_count.to_i64 * 5)
      constants_bytes = WanVAEShape.bytes("scale constants", 2_i64 * cfg.z_dim)
      output_frames = cfg.decoded_frames(max_latent_frames)
      output_count = WanVAEShape.count("decoded output", cfg.batches, 3, output_frames, cfg.height, cfg.width)
      output_bytes = WanVAEShape.bytes("decoded output", output_count.to_i64)
      planned = WanVAEShape.add_bytes("workspace plan", scratch_bytes, cache_bytes)
      planned = WanVAEShape.add_bytes("workspace plan", planned, constants_bytes)
      planned = WanVAEShape.add_bytes("workspace plan", planned, output_bytes)
      planned = WanVAEShape.add_bytes("workspace plan", planned, weights.resident_bytes)
      if reserve_bytes > budget_bytes || planned > budget_bytes - reserve_bytes
        raise ArgumentError.new("Wan VAE activation/cache plan exceeds caller memory budget")
      end
      workspace_arena.upload("wan_vae_scale_mean", SCALE_MEAN.to_a)
      workspace_arena.upload("wan_vae_scale_inv_std", SCALE_INV_STD.to_a)
      5.times { |index| workspace_arena.alloc("wan_vae_tmp_#{index}", activation_count) }
      # Four compute temporaries are consumed by the alias-safe residual paths;
      # the fifth is reserved for a one-frame decoder input slice.
      cache_before = workspace_arena.resident_bytes
      specs.each do |spec|
        frame_plane = WanVAEShape.count("cache frame", cfg.batches, spec.channels, spec.height, spec.width)
        count = WanVAEShape.count("two-frame cache", 2, frame_plane)
        workspace_arena.alloc("wan_vae_cache_#{spec.name}_0", count)
        workspace_arena.alloc("wan_vae_cache_#{spec.name}_1", count)
      end
      actual_workspace_bytes = workspace_arena.resident_bytes
      actual_plan = WanVAEShape.add_bytes("actual workspace plan", actual_workspace_bytes, output_bytes)
      actual_plan = WanVAEShape.add_bytes("actual workspace plan", actual_plan, weights.resident_bytes)
      if reserve_bytes > budget_bytes || actual_plan > budget_bytes - reserve_bytes
        workspace_arena.release
        raise ArgumentError.new("Wan VAE workspace allocation exceeds caller memory budget")
      end
      workspace = WanVAEWorkspace.new(workspace_arena, cfg, activation_count,
        actual_workspace_bytes - cache_before,
        WanVAEShape.add_bytes("workspace reserve", actual_plan, reserve_bytes),
        max_latent_frames)
      specs.each do |spec|
        first = workspace_arena["wan_vae_cache_#{spec.name}_0"]
        second = workspace_arena["wan_vae_cache_#{spec.name}_1"]
        frame_plane = WanVAEShape.count("cache frame", cfg.batches, spec.channels, spec.height, spec.width)
        workspace.add_cache_state(spec.name, first, second, frame_plane)
      end
      workspace
    rescue ex
      workspace_arena.release if workspace_arena.resident_bytes > 0
      raise ex
    end

    # Encode one reference image. Input is contiguous F32 BCTHW with T=1;
    # output is contiguous F32 B,Cz,1,H/8,W/8.
    def self.encode_image(e : ML::Metal::ComputeEncoder, image : ML::MetalBuffer,
                          latent : ML::MetalBuffer, weights : Arena,
                          workspace : WanVAEWorkspace, cfg : WanVAEConfig) : Nil
      encode_video(e, image, latent, weights, workspace, cfg, 1)
      nil
    end

    # The pinned motion-reference path uses a first-frame chunk followed by
    # four-frame chunks. Its 73-frame input produces 19 latent frames.
    def self.encode_motion(e : ML::Metal::ComputeEncoder, video : ML::MetalBuffer,
                           latent : ML::MetalBuffer, weights : Arena,
                           workspace : WanVAEWorkspace, cfg : WanVAEConfig,
                           frames : Int32 = 73) : Int32
      encode_video(e, video, latent, weights, workspace, cfg, frames)
    end

    # Stream an input whose frame count is 1+4n. Cache updates remain ordered
    # inside the caller's command encoder; this path does no host activation
    # math, allocation, submission, waiting or reading.
    def self.encode_video(e : ML::Metal::ComputeEncoder, video : ML::MetalBuffer,
                          latent : ML::MetalBuffer, weights : Arena,
                          workspace : WanVAEWorkspace, cfg : WanVAEConfig,
                          frames : Int32) : Int32
      validate_workspace!(workspace, cfg)
      validate_weights!(weights, cfg)
      validate_disjoint_residency!(weights, workspace)
      unless frames > 0 && (frames == 1 || (frames - 1) % 4 == 0)
        raise ArgumentError.new("Wan VAE encoder frame count must be 1+4n")
      end
      latent_frames = 1 + (frames - 1) // 4
      input_count = WanVAEShape.count("video input", cfg.batches, 3, frames, cfg.height, cfg.width)
      latent_count = WanVAEShape.count("video latent", cfg.batches, cfg.z_dim, latent_frames,
        cfg.latent_height, cfg.latent_width)
      MetalOps.check(video, input_count.to_i64)
      MetalOps.check(latent, latent_count.to_i64)
      MetalOps.distinct(latent, video)
      validate_external_buffer!(video, weights, workspace)
      validate_external_buffer!(latent, weights, workspace)
      workspace.reset_cache_state!
      frame_offset = 0
      latent_offset = 0
      chunk_index = 0
      while frame_offset < frames
        chunk_frames = chunk_index == 0 ? 1 : 4
        source = workspace.buffer("wan_vae_tmp_4")
        dispatch_copy_frames(e, video, source, cfg.batches, 3, frames, frame_offset,
          cfg.height, cfg.width, chunk_frames, 0, chunk_frames)
        value = Activation.new(source, cfg.batches, 3, chunk_frames, cfg.height, cfg.width)
        value = encode_encoder(e, value, weights, workspace, cfg)
        value = conv_auto(e, value, weights, workspace, "conv1", cfg.z_dim * 2,
          1, 1, 1, 1, 1, 1, 0, 0, 0)
        scaled = workspace.temp(value.buffer, source)
        dispatch_split_scale(e, value.buffer, scaled, workspace, cfg,
          value.frames, value.height, value.width)
        dispatch_copy_frames(e, scaled, latent, cfg.batches, cfg.z_dim, value.frames, 0,
          cfg.latent_height, cfg.latent_width, latent_frames, latent_offset, value.frames)
        frame_offset += chunk_frames
        latent_offset += value.frames
        chunk_index += 1
      end
      unless frame_offset == frames && latent_offset == latent_frames
        raise ArgumentError.new("Wan VAE encoder cache produced #{latent_offset} latents, expected #{latent_frames}")
      end
      latent_frames
    end

    # Decode one latent sequence frame-by-frame. Wan's first upsample3d call is
    # `Rep` and bypasses temporal conv/shuffle. With two temporal upsamplers,
    # 22 latent frames write exactly 85 frames. `decoded` is the raw output;
    # use crop_frames! for the caller's explicit S2V window/drop policy. No
    # command submission, wait, or activation read occurs here.
    def self.decode(e : ML::Metal::ComputeEncoder, latent : ML::MetalBuffer,
                    decoded : ML::MetalBuffer, weights : Arena,
                    workspace : WanVAEWorkspace, cfg : WanVAEConfig,
                    latent_frames : Int32 = 21) : Int32
      validate_workspace!(workspace, cfg)
      validate_weights!(weights, cfg)
      validate_disjoint_residency!(weights, workspace)
      unless latent_frames > 0 && latent_frames <= workspace.max_latent_frames
        raise ArgumentError.new("Wan VAE decode exceeds its planned latent-frame capacity")
      end
      output_frames = cfg.decoded_frames(latent_frames)
      latent_count = WanVAEShape.count("latent input", cfg.batches, cfg.z_dim,
        latent_frames, cfg.latent_height, cfg.latent_width)
      output_count = WanVAEShape.count("decoded output", cfg.batches, 3,
        output_frames, cfg.height, cfg.width)
      MetalOps.check(latent, latent_count.to_i64)
      MetalOps.check(decoded, output_count.to_i64)
      MetalOps.distinct(decoded, latent)
      validate_external_buffer!(latent, weights, workspace)
      validate_external_buffer!(decoded, weights, workspace)
      workspace.reset_cache_state!
      offset = 0
      latent_frames.times do |frame|
        source = workspace.buffer("wan_vae_tmp_4")
        dispatch_time_slice(e, latent, source, cfg, latent_frames, frame)
        value = Activation.new(source, cfg.batches, cfg.z_dim, 1, cfg.latent_height, cfg.latent_width)
        unscaled = workspace.temp(value.buffer)
        dispatch_scale(e, value.buffer, unscaled, workspace, cfg, 1,
          cfg.latent_height, cfg.latent_width, true)
        value = Activation.new(unscaled, cfg.batches, cfg.z_dim, 1, cfg.latent_height, cfg.latent_width)
        value = conv_auto(e, value, weights, workspace, "conv2", cfg.z_dim,
          1, 1, 1, 1, 1, 1, 0, 0, 0)
        value = decode_decoder(e, value, weights, workspace, cfg)
        chunk_frames = value.frames
        dispatch_copy_clamp(e, value, decoded, cfg, output_frames, offset)
        offset += chunk_frames
      end
      unless offset == output_frames
        raise ArgumentError.new("Wan VAE decoder temporal cache produced #{offset} frames, expected #{output_frames}")
      end
      output_frames
    end

    def self.crop_frames!(e : ML::Metal::ComputeEncoder, source : ML::MetalBuffer,
                          target : ML::MetalBuffer, cfg : WanVAEConfig,
                          source_frames : Int32, source_offset : Int32,
                          target_frames : Int32) : Nil
      unless source_offset >= 0 && target_frames > 0 && source_offset <= source_frames &&
             target_frames <= source_frames - source_offset
        raise ArgumentError.new("Invalid Wan VAE output crop")
      end
      source_count = WanVAEShape.count("crop source", cfg.batches, 3, source_frames,
        cfg.height, cfg.width)
      target_count = WanVAEShape.count("crop target", cfg.batches, 3, target_frames,
        cfg.height, cfg.width)
      MetalOps.check(source, source_count.to_i64)
      MetalOps.check(target, target_count.to_i64)
      MetalOps.distinct(target, source)
      e.set_pipeline(pipeline("wan_vae_crop_frames_f32"))
      e.set_buffer(source, 0).set_buffer(target, 1, ML::Metal::BufferAccess::Write)
      e.set_value(cfg.batches.to_u32, 2).set_value(3_u32, 3)
      e.set_value(source_frames.to_u32, 4).set_value(source_offset.to_u32, 5)
      e.set_value(target_frames.to_u32, 6).set_value(cfg.height.to_u32, 7)
      e.set_value(cfg.width.to_u32, 8)
      e.dispatch_1d(target_count)
    end

    # Copy a contiguous temporal slice from BCTHW source into a caller-selected
    # temporal offset of a BCTHW target. Source and destination extents are
    # explicit so later encoder chunks append rather than overwrite frame zero.
    def self.copy_frames!(e : ML::Metal::ComputeEncoder, source : ML::MetalBuffer,
                          target : ML::MetalBuffer, batches : Int32, channels : Int32,
                          source_frames : Int32, source_offset : Int32,
                          height : Int32, width : Int32, target_frames : Int32,
                          target_offset : Int32, slice_frames : Int32) : Nil
      dispatch_copy_frames(e, source, target, batches, channels, source_frames,
        source_offset, height, width, target_frames, target_offset, slice_frames)
    end

    private def self.validate_workspace!(workspace : WanVAEWorkspace, cfg : WanVAEConfig) : Nil
      unless workspace.config.batches == cfg.batches && workspace.config.height == cfg.height &&
             workspace.config.width == cfg.width && workspace.config.dim == cfg.dim &&
             workspace.config.z_dim == cfg.z_dim && workspace.config.dim_mult == cfg.dim_mult &&
             workspace.config.num_res_blocks == cfg.num_res_blocks &&
             workspace.config.temporal_downsample == cfg.temporal_downsample
        raise ArgumentError.new("Wan VAE workspace/config mismatch")
      end
      activation_count = WanVAEShape.count("activation scratch", cfg.batches, cfg.temporal_factor,
        cfg.output_channels, cfg.height, cfg.width)
      (0...5).each { |index| MetalOps.check(workspace.buffer("wan_vae_tmp_#{index}"), activation_count.to_i64) }
      MetalOps.check(workspace.buffer("wan_vae_scale_mean"), cfg.z_dim.to_i64)
      MetalOps.check(workspace.buffer("wan_vae_scale_inv_std"), cfg.z_dim.to_i64)
      specs = encoder_cache_specs(cfg) + decoder_cache_specs(cfg)
      unless workspace.cache_names.size == specs.size
        raise ArgumentError.new("Wan VAE workspace cache set does not match its config")
      end
      specs.each do |spec|
        state = workspace.cache(spec.name)
        first = workspace.buffer("wan_vae_cache_#{spec.name}_0")
        second = workspace.buffer("wan_vae_cache_#{spec.name}_1")
        unless state.first.same?(first) && state.second.same?(second)
          raise ArgumentError.new("Wan VAE cache identity changed")
        end
        frame_plane = WanVAEShape.count("cache frame", cfg.batches,
          spec.channels, spec.height, spec.width)
        cache_count = WanVAEShape.count("two-frame cache", 2, frame_plane)
        MetalOps.check(state.first, cache_count.to_i64)
        MetalOps.check(state.second, cache_count.to_i64)
      end
    end

    private def self.validate_external_buffer!(buffer : ML::MetalBuffer,
                                               weights : Arena,
                                               workspace : WanVAEWorkspace) : Nil
      workspace.arena.buffers.each_value do |resident|
        if MetalOps.overlap?(buffer, resident)
          raise ArgumentError.new("Wan VAE input/output aliases workspace residency")
        end
      end
      weights.buffers.each_value do |resident|
        if MetalOps.overlap?(buffer, resident)
          raise ArgumentError.new("Wan VAE input/output aliases checkpoint weights")
        end
      end
    end

    private def self.validate_disjoint_residency!(weights : Arena,
                                                  workspace : WanVAEWorkspace) : Nil
      resident = workspace.arena.buffers.values
      resident.each_with_index do |left, index|
        ((index + 1)...resident.size).each do |other_index|
          if MetalOps.overlap?(left, resident[other_index])
            raise ArgumentError.new("Wan VAE workspace buffers overlap")
          end
        end
      end
      weights.buffers.each_value do |weight|
        if resident.any? { |buffer| MetalOps.overlap?(weight, buffer) }
          raise ArgumentError.new("Wan VAE weights overlap workspace residency")
        end
      end
    end

    private def self.validate_weights!(weights : Arena, cfg : WanVAEConfig) : Nil
      checkpoint_bindings(cfg).each do |binding|
        count = binding.shape.product
        weights.check_weight(binding.target, count)
      end
    end

    private def self.encode_encoder(e : ML::Metal::ComputeEncoder, input : Activation,
                                    weights : Arena, workspace : WanVAEWorkspace,
                                    cfg : WanVAEConfig) : Activation
      dims = cfg.encoder_dims
      value = conv_auto(e, input, weights, workspace, "encoder.conv1", dims.first,
        3, 3, 3, 1, 1, 1, 2, 1, 1, "encoder.conv1")
      index = 0
      (0...cfg.dim_mult.size).each do |stage|
        in_channels = dims[stage]
        out_channels = dims[stage + 1]
        cfg.num_res_blocks.times do |block|
          prefix = "encoder.downsamples.#{index}"
          value = residual(e, value, weights, workspace, prefix, in_channels, out_channels,
            prefix, true)
          in_channels = out_channels
          index += 1
        end
        if stage < cfg.dim_mult.size - 1
          prefix = "encoder.downsamples.#{index}"
          value = downsample(e, value, weights, workspace, prefix,
            cfg.temporal_downsample[stage])
          index += 1
        end
      end
      value = residual(e, value, weights, workspace, "encoder.middle.0", dims.last, dims.last,
        "encoder.middle.0", true)
      value = attention(e, value, weights, workspace, "encoder.middle.1")
      value = residual(e, value, weights, workspace, "encoder.middle.2", dims.last, dims.last,
        "encoder.middle.2", true)
      normed = rms_norm(e, value, weights, workspace, "encoder.head.0.gamma", 4)
      activated = silu(e, normed, workspace)
      conv_auto(e, activated, weights, workspace, "encoder.head.2", cfg.z_dim * 2,
        3, 3, 3, 1, 1, 1, 2, 1, 1, "encoder.head.2")
    end

    private def self.decode_decoder(e : ML::Metal::ComputeEncoder, input : Activation,
                                    weights : Arena, workspace : WanVAEWorkspace,
                                    cfg : WanVAEConfig) : Activation
      dims = cfg.decoder_dims
      value = conv_auto(e, input, weights, workspace, "decoder.conv1", dims.first,
        3, 3, 3, 1, 1, 1, 2, 1, 1, "decoder.conv1")
      value = residual(e, value, weights, workspace, "decoder.middle.0", dims.first, dims.first, "decoder.middle.0", true)
      value = attention(e, value, weights, workspace, "decoder.middle.1")
      value = residual(e, value, weights, workspace, "decoder.middle.2", dims.first, dims.first, "decoder.middle.2", true)
      index = 0
      (0...cfg.dim_mult.size).each do |stage|
        source_in = dims[stage]
        out_channels = dims[stage + 1]
        in_channels = stage >= 1 ? source_in // 2 : source_in
        (cfg.num_res_blocks + 1).times do |block|
          prefix = "decoder.upsamples.#{index}"
          value = residual(e, value, weights, workspace, prefix, in_channels, out_channels, prefix, true)
          in_channels = out_channels
          index += 1
        end
        if stage < cfg.dim_mult.size - 1
          prefix = "decoder.upsamples.#{index}"
          value = upsample(e, value, weights, workspace, prefix,
            cfg.temporal_downsample.reverse[stage])
          index += 1
        end
      end
      normed = rms_norm(e, value, weights, workspace, "decoder.head.0.gamma", 4)
      activated = silu(e, normed, workspace)
      conv_auto(e, activated, weights, workspace, "decoder.head.2", 3,
        3, 3, 3, 1, 1, 1, 2, 1, 1, "decoder.head.2")
    end

    private def self.residual(e : ML::Metal::ComputeEncoder, input : Activation,
                              weights : Arena, workspace : WanVAEWorkspace,
                              prefix : String, in_channels : Int32, out_channels : Int32,
                              cache_prefix : String?, cached : Bool) : Activation
      skip = workspace.buffer("wan_vae_tmp_4")
      # The fifth scratch slot doubles as the stable residual path; decoder's
      # latent slice is consumed before residual blocks begin.
      if input.buffer.same?(skip)
        skip = workspace.temp(input.buffer)
      end
      copy(e, input.buffer, skip, input_count(input))
      if in_channels != out_channels
        skip_out = conv(e, input, skip, weights, workspace, "#{prefix}.shortcut",
          out_channels, 1, 1, 1, 1, 1, 1, 0, 0, 0, nil, false)
        skip = skip_out.buffer
      end

      norm1 = rms_norm(e, input, weights, workspace, "#{prefix}.residual.0.gamma", 4, skip)
      silu1 = silu(e, norm1, workspace, skip)
      conv1_cache = cached ? "#{cache_prefix}.residual.2" : nil
      conv1 = conv_auto(e, silu1, weights, workspace, "#{prefix}.residual.2", out_channels,
        3, 3, 3, 1, 1, 1, 2, 1, 1, conv1_cache, skip)
      norm2 = rms_norm(e, conv1, weights, workspace, "#{prefix}.residual.3.gamma", 4, skip)
      silu2 = silu(e, norm2, workspace, skip)
      conv2_cache = cached ? "#{cache_prefix}.residual.6" : nil
      conv2 = conv_auto(e, silu2, weights, workspace, "#{prefix}.residual.6", out_channels,
        3, 3, 3, 1, 1, 1, 2, 1, 1, conv2_cache, skip)
      skip_activation = Activation.new(skip, input.batches, out_channels,
        input.frames, input.height, input.width)
      add(e, conv2, skip_activation, workspace)
    end

    private def self.attention(e : ML::Metal::ComputeEncoder, input : Activation,
                               weights : Arena, workspace : WanVAEWorkspace,
                               prefix : String) : Activation
      normed = rms_norm(e, input, weights, workspace, "#{prefix}.norm.gamma", 3)
      qkv = conv_auto(e, normed, weights, workspace, "#{prefix}.to_qkv", input.channels * 3,
        1, 1, 1, 1, 1, 1, 0, 0, 0, nil, input.buffer)
      attended_buffer = workspace.temp(qkv.buffer, input.buffer)
      e.set_pipeline(pipeline("wan_vae_attention_f32"))
      e.set_buffer(qkv.buffer, 0).set_buffer(attended_buffer, 1, ML::Metal::BufferAccess::Write)
      e.set_value(cfg_batches(workspace).to_u32, 2).set_value(input.channels.to_u32, 3)
      e.set_value(input.frames.to_u32, 4).set_value(input.height.to_u32, 5)
      e.set_value(input.width.to_u32, 6)
      const_threads = Math.min(512, next_power_of_two(input.channels))
      query_count = input.frames * input.height * input.width * cfg_batches(workspace)
      e.dispatch_threadgroups({query_count, 1, 1}, {const_threads, 1, 1})
      attended = Activation.new(attended_buffer, input.batches, input.channels, input.frames, input.height, input.width)
      projected = conv_auto(e, attended, weights, workspace, "#{prefix}.proj", input.channels,
        1, 1, 1, 1, 1, 1, 0, 0, 0, nil, input.buffer)
      add(e, projected, input, workspace)
    end

    private def self.downsample(e : ML::Metal::ComputeEncoder, input : Activation,
                                weights : Arena, workspace : WanVAEWorkspace,
                                prefix : String, temporal : Bool) : Activation
      out_height = input.height // 2
      out_width = input.width // 2
      output = workspace.temp(input.buffer)
      conv(e, input, output, weights, workspace, "#{prefix}.resample.1", input.channels,
        1, 3, 3, 1, 2, 2, 0, 0, 0, nil, false,
        input.frames, out_height, out_width)
      spatial = Activation.new(output, input.batches, input.channels, input.frames, out_height, out_width)
      return spatial unless temporal

      cache_name = "#{prefix}.time_conv"
      state = workspace.cache(cache_name)
      # Upstream downsample3d skips time_conv on its cold call, then retains
      # exactly one last spatial frame for the next stride-2 temporal window.
      if state.history == 0
        cache_input(e, spatial, workspace, cache_name, false, true)
        return spatial
      end
      temporal_output = workspace.temp(spatial.buffer, input.buffer)
      conv(e, spatial, temporal_output, weights, workspace, "#{prefix}.time_conv", input.channels,
        3, 1, 1, 2, 1, 1, 0, 0, 0, cache_name, true,
        nil, nil, nil, false, true)
    end

    private def self.upsample(e : ML::Metal::ComputeEncoder, input : Activation,
                              weights : Arena, workspace : WanVAEWorkspace,
                              prefix : String, temporal : Bool) : Activation
      value = input
      if temporal
        state = workspace.cache("#{prefix}.time_conv")
        if state.history == 0 && !state.replicate
          state.replicate = true
        else
          # Upstream's Rep sentinel uses a cold causal temporal conv. Subsequent
          # calls consume the two cached frames through the same generic conv.
          time_conv = conv_auto(e, value, weights, workspace, "#{prefix}.time_conv",
            value.channels * 2, 3, 1, 1, 1, 1, 1, 2, 0, 0,
            "#{prefix}.time_conv", nil, state.replicate)
          time_frames = time_conv.frames
          shuffled = workspace.temp(time_conv.buffer, value.buffer)
          e.set_pipeline(pipeline("wan_vae_temporal_shuffle_f32"))
          e.set_buffer(time_conv.buffer, 0).set_buffer(shuffled, 1, ML::Metal::BufferAccess::Write)
          e.set_value(cfg_batches(workspace).to_u32, 2).set_value(value.channels.to_u32, 3)
          e.set_value(time_frames.to_u32, 4).set_value(value.height.to_u32, 5)
          e.set_value(value.width.to_u32, 6)
          shuffled_frames = time_frames * 2
          shuffle_count = WanVAEShape.count("temporal shuffle", cfg_batches(workspace),
            value.channels, shuffled_frames, value.height, value.width)
          e.dispatch_1d(shuffle_count)
          value = Activation.new(shuffled, value.batches, value.channels, shuffled_frames, value.height, value.width)
        end
      end
      upsampled = workspace.temp(value.buffer, input.buffer)
      e.set_pipeline(pipeline("wan_vae_nearest2x_f32"))
      e.set_buffer(value.buffer, 0).set_buffer(upsampled, 1, ML::Metal::BufferAccess::Write)
      e.set_value(cfg_batches(workspace).to_u32, 2).set_value(value.channels.to_u32, 3)
      e.set_value(value.frames.to_u32, 4).set_value(value.height.to_u32, 5)
      e.set_value(value.width.to_u32, 6)
      nearest_count = WanVAEShape.count("nearest upsample", cfg_batches(workspace),
        value.channels, value.frames, value.height * 2, value.width * 2)
      e.dispatch_1d(nearest_count)
      upsampled_value = Activation.new(upsampled, value.batches, value.channels, value.frames,
        value.height * 2, value.width * 2)
      conv_auto(e, upsampled_value, weights, workspace, "#{prefix}.resample.1",
        value.channels // 2, 1, 3, 3, 1, 1, 1, 0, 1, 1)
    end

    private def self.rms_norm(e : ML::Metal::ComputeEncoder, input : Activation,
                              weights : Arena, workspace : WanVAEWorkspace,
                              gamma_name : String, rank : Int32,
                              forbidden : ML::MetalBuffer? = nil) : Activation
      output = workspace.temp(input.buffer, forbidden)
      gamma = weights[gamma_name]
      weights.check_weight(gamma_name, input.channels.to_i64)
      e.set_pipeline(pipeline("wan_vae_rms_norm_f32"))
      e.set_buffer(input.buffer, 0).set_buffer(gamma, 1).set_buffer(gamma, 2)
      e.set_buffer(output, 3, ML::Metal::BufferAccess::Write)
      e.set_value(cfg_batches(workspace).to_u32, 4).set_value(input.channels.to_u32, 5)
      e.set_value(input.frames.to_u32, 6).set_value(input.height.to_u32, 7)
      e.set_value(input.width.to_u32, 8)
      e.set_value(weights.dtype(gamma_name).bf16? ? 1_u32 : 0_u32, 9)
      spatial_count = WanVAEShape.count("RMS norm", cfg_batches(workspace),
        input.frames, input.height, input.width)
      e.dispatch_1d(spatial_count)
      Activation.new(output, input.batches, input.channels, input.frames, input.height, input.width)
    end

    private def self.silu(e : ML::Metal::ComputeEncoder, input : Activation,
                          workspace : WanVAEWorkspace,
                          forbidden : ML::MetalBuffer? = nil) : Activation
      output = workspace.temp(input.buffer, forbidden)
      count = input_count(input)
      e.set_pipeline(pipeline("wan_vae_silu_f32"))
      e.set_buffer(input.buffer, 0).set_buffer(output, 1, ML::Metal::BufferAccess::Write)
      e.set_value(count.to_u32, 2)
      e.dispatch_1d(count)
      Activation.new(output, input.batches, input.channels, input.frames, input.height, input.width)
    end

    private def self.add(e : ML::Metal::ComputeEncoder, left : Activation,
                         right : Activation, workspace : WanVAEWorkspace) : Activation
      unless left.channels == right.channels && left.frames == right.frames &&
             left.height == right.height && left.width == right.width
        raise ArgumentError.new("Wan VAE residual shape mismatch")
      end
      output = workspace.temp(left.buffer, right.buffer)
      count = input_count(left)
      e.set_pipeline(pipeline("wan_vae_add_f32"))
      e.set_buffer(left.buffer, 0).set_buffer(right.buffer, 1)
      e.set_buffer(output, 2, ML::Metal::BufferAccess::Write).set_value(count.to_u32, 3)
      e.dispatch_1d(count)
      Activation.new(output, left.batches, left.channels, left.frames, left.height, left.width)
    end

    private def self.conv_auto(e : ML::Metal::ComputeEncoder, input : Activation,
                               weights : Arena, workspace : WanVAEWorkspace,
                               source : String, out_channels : Int32,
                               kernel_t : Int32, kernel_h : Int32, kernel_w : Int32,
                               stride_t : Int32, stride_h : Int32, stride_w : Int32,
                               pad_t : Int32, pad_h : Int32, pad_w : Int32,
                               cache_name : String? = nil,
                               forbidden : ML::MetalBuffer? = nil,
                               cache_replicate : Bool = false,
                               cache_last_only : Bool = false) : Activation
      output = workspace.temp(input.buffer, forbidden)
      conv(e, input, output, weights, workspace, source, out_channels,
        kernel_t, kernel_h, kernel_w, stride_t, stride_h, stride_w,
        pad_t, pad_h, pad_w, cache_name, !cache_name.nil?, nil, nil, nil,
        cache_replicate, cache_last_only)
    end

    private def self.conv(e : ML::Metal::ComputeEncoder, input : Activation,
                          output : ML::MetalBuffer, weights : Arena,
                          workspace : WanVAEWorkspace, source : String,
                          out_channels : Int32, kernel_t : Int32, kernel_h : Int32,
                          kernel_w : Int32, stride_t : Int32, stride_h : Int32,
                          stride_w : Int32, pad_t : Int32, pad_h : Int32,
                          pad_w : Int32, cache_name : String?, cached : Bool,
                          output_frames : Int32? = nil, output_height : Int32? = nil,
                          output_width : Int32? = nil,
                          cache_replicate : Bool = false,
                          cache_last_only : Bool = false) : Activation
      weight_name = "#{source}.weight"
      bias_name = "#{source}.bias"
      weight_count = out_channels.to_i64 * input.channels * kernel_t * kernel_h * kernel_w
      bias_count = out_channels.to_i64
      weights.check_weight(weight_name, weight_count)
      weights.check_weight(bias_name, bias_count)
      cache_state = cache_name ? workspace.cache(cache_name) : nil
      cache_frames = cached && cache_state ? cache_state.history : 0
      cache_buffer = if cache_frames > 0 && cache_state
                       cache_state.current
                     else
                       input.buffer
                     end
      effective_pad_t = Math.max(0, pad_t - cache_frames)
      out_t = output_frames || ((cache_frames + input.frames + effective_pad_t - kernel_t) // stride_t + 1)
      out_h = output_height || ((input.height + 2 * pad_h - kernel_h) // stride_h + 1)
      out_w = output_width || ((input.width + 2 * pad_w - kernel_w) // stride_w + 1)
      count = WanVAEShape.count("convolution output", cfg_batches(workspace),
        out_channels, out_t, out_h, out_w)
      MetalOps.check(output, count.to_i64)
      weight = weights[weight_name]
      bias = weights[bias_name]
      e.set_pipeline(pipeline("wan_vae_conv3d_f32"))
      e.set_buffer(input.buffer, 0).set_buffer(cache_buffer, 1)
      e.set_buffer(weight, 2).set_buffer(weight, 3)
      e.set_buffer(bias, 4).set_buffer(bias, 5)
      e.set_buffer(output, 6, ML::Metal::BufferAccess::Write)
      e.set_value(cfg_batches(workspace).to_u32, 7).set_value(input.channels.to_u32, 8)
      e.set_value(out_channels.to_u32, 9).set_value(input.frames.to_u32, 10)
      e.set_value(input.height.to_u32, 11).set_value(input.width.to_u32, 12)
      e.set_value(cache_frames.to_u32, 13).set_value(kernel_t.to_u32, 14)
      e.set_value(kernel_h.to_u32, 15).set_value(kernel_w.to_u32, 16)
      e.set_value(stride_t.to_u32, 17).set_value(stride_h.to_u32, 18)
      e.set_value(stride_w.to_u32, 19).set_value(effective_pad_t.to_u32, 20)
      e.set_value(pad_h.to_u32, 21).set_value(pad_w.to_u32, 22)
      e.set_value(out_t.to_u32, 23).set_value(out_h.to_u32, 24).set_value(out_w.to_u32, 25)
      e.set_value(weights.dtype(weight_name).bf16? ? 1_u32 : 0_u32, 26)
      e.set_value(weights.dtype(bias_name).bf16? ? 1_u32 : 0_u32, 27)
      e.dispatch_1d(count)
      if cached && cache_name && cache_state
        cache_after_conv(e, input, cache_name, workspace, cache_replicate, cache_last_only)
      end
      Activation.new(output, cfg_batches(workspace), out_channels, out_t, out_h, out_w)
    end

    private def self.cache_input(e : ML::Metal::ComputeEncoder, input : Activation,
                                 workspace : WanVAEWorkspace, name : String,
                                 replicate : Bool, last_only : Bool = false) : Nil
      state = workspace.cache(name)
      replicate_zero = replicate && input.frames == 1
      old_buffer = state.history > 0 ? state.current : input.buffer
      e.set_pipeline(pipeline("wan_vae_cache_last2_f32"))
      e.set_buffer(input.buffer, 0).set_buffer(old_buffer, 1)
      e.set_buffer(state.spare, 2, ML::Metal::BufferAccess::Write)
      e.set_value(cfg_batches(workspace).to_u32, 3).set_value(input.channels.to_u32, 4)
      e.set_value(input.frames.to_u32, 5).set_value(input.height.to_u32, 6)
      e.set_value(input.width.to_u32, 7).set_value(state.history.to_u32, 8)
      e.set_value(replicate_zero ? 1_u32 : 0_u32, 9)
      e.set_value(last_only ? 1_u32 : 0_u32, 10)
      e.dispatch_1d(state.frame_plane * 2)
      new_history = replicate ? 2 : (last_only ? 1 : Math.min(2, state.history + input.frames))
      state.advance!(new_history, false)
    end

    private def self.cache_after_conv(e : ML::Metal::ComputeEncoder, input : Activation,
                                      name : String, workspace : WanVAEWorkspace,
                                      replicate : Bool, last_only : Bool) : Nil
      cache_input(e, input, workspace, name, replicate, last_only)
    end

    private def self.dispatch_scale(e : ML::Metal::ComputeEncoder, input : ML::MetalBuffer,
                                    output : ML::MetalBuffer, workspace : WanVAEWorkspace,
                                    cfg : WanVAEConfig, frames : Int32,
                                    height : Int32, width : Int32, unscale : Bool) : Nil
      count = WanVAEShape.count("latent scale", cfg.batches, cfg.z_dim, frames, height, width)
      e.set_pipeline(pipeline("wan_vae_scale_f32"))
      e.set_buffer(input, 0).set_buffer(workspace.buffer("wan_vae_scale_mean"), 1)
      e.set_buffer(workspace.buffer("wan_vae_scale_inv_std"), 2)
      e.set_buffer(output, 3, ML::Metal::BufferAccess::Write)
      e.set_value(cfg.batches.to_u32, 4).set_value(cfg.z_dim.to_u32, 5)
      e.set_value(frames.to_u32, 6).set_value(height.to_u32, 7).set_value(width.to_u32, 8)
      e.set_value(unscale ? 1_u32 : 0_u32, 9)
      e.dispatch_1d(count)
    end

    private def self.dispatch_split_scale(e : ML::Metal::ComputeEncoder, input : ML::MetalBuffer,
                                          output : ML::MetalBuffer, workspace : WanVAEWorkspace,
                                          cfg : WanVAEConfig, frames : Int32,
                                          height : Int32, width : Int32) : Nil
      count = WanVAEShape.count("posterior mean scaling", cfg.batches, cfg.z_dim,
        frames, height, width)
      e.set_pipeline(pipeline("wan_vae_split_scale_f32"))
      e.set_buffer(input, 0).set_buffer(workspace.buffer("wan_vae_scale_mean"), 1)
      e.set_buffer(workspace.buffer("wan_vae_scale_inv_std"), 2)
      e.set_buffer(output, 3, ML::Metal::BufferAccess::Write)
      e.set_value(cfg.batches.to_u32, 4).set_value(cfg.z_dim.to_u32, 5)
      e.set_value(frames.to_u32, 6).set_value(height.to_u32, 7).set_value(width.to_u32, 8)
      e.dispatch_1d(count)
    end

    private def self.dispatch_time_slice(e : ML::Metal::ComputeEncoder, source : ML::MetalBuffer,
                                         target : ML::MetalBuffer, cfg : WanVAEConfig,
                                         frames : Int32, selected : Int32) : Nil
      count = WanVAEShape.count("latent frame slice", cfg.batches, cfg.z_dim,
        1, cfg.latent_height, cfg.latent_width)
      e.set_pipeline(pipeline("wan_vae_time_slice_f32"))
      e.set_buffer(source, 0).set_buffer(target, 1, ML::Metal::BufferAccess::Write)
      e.set_value(cfg.batches.to_u32, 2).set_value(cfg.z_dim.to_u32, 3)
      e.set_value(frames.to_u32, 4).set_value(cfg.latent_height.to_u32, 5)
      e.set_value(cfg.latent_width.to_u32, 6).set_value(selected.to_u32, 7)
      e.set_value(1_u32, 8)
      e.dispatch_1d(count)
    end

    private def self.dispatch_copy_frames(e : ML::Metal::ComputeEncoder, source : ML::MetalBuffer,
                                          target : ML::MetalBuffer, batches : Int32,
                                          channels : Int32, source_frames : Int32,
                                          source_offset : Int32, height : Int32, width : Int32,
                                          target_frames : Int32, target_offset : Int32,
                                          slice_frames : Int32) : Nil
      unless source_frames > 0 && source_offset >= 0 && slice_frames > 0 &&
             source_offset <= source_frames && slice_frames <= source_frames - source_offset
        raise ArgumentError.new("Wan VAE frame slice is out of bounds")
      end
      unless target_frames > 0 && target_offset >= 0 &&
             target_offset <= target_frames && slice_frames <= target_frames - target_offset
        raise ArgumentError.new("Wan VAE frame destination is out of bounds")
      end
      source_count = WanVAEShape.count("frame-copy source", batches, channels,
        source_frames, height, width)
      target_count = WanVAEShape.count("frame-copy target", batches, channels,
        target_frames, height, width)
      MetalOps.check(source, source_count.to_i64)
      MetalOps.check(target, target_count.to_i64)
      MetalOps.distinct(target, source)
      e.set_pipeline(pipeline("wan_vae_copy_frames_f32"))
      e.set_buffer(source, 0).set_buffer(target, 1, ML::Metal::BufferAccess::Write)
      e.set_value(batches.to_u32, 2).set_value(channels.to_u32, 3)
      e.set_value(source_frames.to_u32, 4).set_value(source_offset.to_u32, 5)
      e.set_value(height.to_u32, 6).set_value(width.to_u32, 7)
      e.set_value(target_frames.to_u32, 8).set_value(target_offset.to_u32, 9)
      e.set_value(slice_frames.to_u32, 10)
      e.dispatch_1d(WanVAEShape.count("frame-copy slice", batches, channels,
        slice_frames, height, width))
    end

    private def self.dispatch_copy_clamp(e : ML::Metal::ComputeEncoder, input : Activation,
                                         output : ML::MetalBuffer, cfg : WanVAEConfig,
                                         target_frames : Int32, offset : Int32) : Nil
      count = input_count(input)
      e.set_pipeline(pipeline("wan_vae_copy_clamp_chunk_f32"))
      e.set_buffer(input.buffer, 0).set_buffer(output, 1, ML::Metal::BufferAccess::Write)
      e.set_value(cfg.batches.to_u32, 2).set_value(input.channels.to_u32, 3)
      e.set_value(input.frames.to_u32, 4).set_value(input.height.to_u32, 5)
      e.set_value(input.width.to_u32, 6).set_value(target_frames.to_u32, 7)
      e.set_value(offset.to_u32, 8)
      e.dispatch_1d(count)
    end

    private def self.dispatch_copy(e : ML::Metal::ComputeEncoder, source : ML::MetalBuffer,
                                   target : ML::MetalBuffer, count : Int32) : Nil
      e.set_pipeline(MetalOps.pipeline("video_copy_f32"))
      e.set_buffer(source, 0).set_buffer(target, 1, ML::Metal::BufferAccess::Write)
      e.set_value(count.to_u32, 2)
      e.dispatch_1d(count)
    end

    private def self.copy(e : ML::Metal::ComputeEncoder, source : ML::MetalBuffer,
                          target : ML::MetalBuffer, count : Int32) : Nil
      dispatch_copy(e, source, target, count)
    end

    private def self.input_count(value : Activation) : Int32
      WanVAEShape.count("activation", value.batches, value.channels,
        value.frames, value.height, value.width)
    end

    private def self.cfg_batches(workspace : WanVAEWorkspace) : Int32
      workspace.config.batches
    end

    private def self.next_power_of_two(value : Int32) : Int32
      power = 32
      while power < value
        power *= 2
      end
      power
    end

    private def self.add_conv3d(bindings : Array(WeightBinding), prefix : String,
                                input : Int32, output : Int32,
                                kt : Int32, kh : Int32, kw : Int32) : Nil
      add_binding(bindings, prefix, "weight", [output, input, kt, kh, kw])
      add_binding(bindings, prefix, "bias", [output])
    end

    private def self.add_conv2d(bindings : Array(WeightBinding), prefix : String,
                                input : Int32, output : Int32,
                                kh : Int32, kw : Int32) : Nil
      add_binding(bindings, prefix, "weight", [output, input, kh, kw])
      add_binding(bindings, prefix, "bias", [output])
    end

    private def self.add_norm_binding(bindings : Array(WeightBinding), source : String,
                                      channels : Int32, rank : Int32) : Nil
      shape = [channels.to_i64]
      (rank - 1).times { shape << 1_i64 }
      bindings << WeightBinding.new(source, source, shape)
    end

    private def self.add_residual_bindings(bindings : Array(WeightBinding), prefix : String,
                                           input : Int32, output : Int32) : Nil
      add_norm_binding(bindings, "#{prefix}.residual.0.gamma", input, 4)
      add_conv3d(bindings, "#{prefix}.residual.2", input, output, 3, 3, 3)
      add_norm_binding(bindings, "#{prefix}.residual.3.gamma", output, 4)
      add_conv3d(bindings, "#{prefix}.residual.6", output, output, 3, 3, 3)
      add_conv3d(bindings, "#{prefix}.shortcut", input, output, 1, 1, 1) if input != output
    end

    private def self.add_attention_bindings(bindings : Array(WeightBinding), prefix : String,
                                            channels : Int32) : Nil
      add_norm_binding(bindings, "#{prefix}.norm.gamma", channels, 3)
      add_conv2d(bindings, "#{prefix}.to_qkv", channels, channels * 3, 1, 1)
      add_conv2d(bindings, "#{prefix}.proj", channels, channels, 1, 1)
    end

    private def self.add_binding(bindings : Array(WeightBinding), prefix : String,
                                 suffix : String, shape : Array(Int32)) : Nil
      name = "#{prefix}.#{suffix}"
      bindings << WeightBinding.new(name, name, shape.map(&.to_i64))
    end

    private def self.encoder_cache_specs(cfg : WanVAEConfig) : Array(CacheSpec)
      specs = [] of CacheSpec
      dims = cfg.encoder_dims
      height = cfg.height
      width = cfg.width
      add_cache_spec(specs, "encoder.conv1", 3, height, width)
      index = 0
      (0...cfg.dim_mult.size).each do |stage|
        in_channels = dims[stage]
        out_channels = dims[stage + 1]
        cfg.num_res_blocks.times do
          prefix = "encoder.downsamples.#{index}"
          add_residual_cache_specs(specs, prefix, in_channels, out_channels, height, width)
          in_channels = out_channels
          index += 1
        end
        if stage < cfg.dim_mult.size - 1
          height //= 2
          width //= 2
          if cfg.temporal_downsample[stage]
            add_cache_spec(specs, "encoder.downsamples.#{index}.time_conv",
              out_channels, height, width)
          end
          index += 1
        end
      end
      add_residual_cache_specs(specs, "encoder.middle.0", dims.last, dims.last, height, width)
      add_residual_cache_specs(specs, "encoder.middle.2", dims.last, dims.last, height, width)
      add_cache_spec(specs, "encoder.head.2", dims.last, height, width)
      specs
    end

    private def self.decoder_cache_specs(cfg : WanVAEConfig) : Array(CacheSpec)
      specs = [] of CacheSpec
      dims = cfg.decoder_dims
      height = cfg.latent_height
      width = cfg.latent_width
      add_cache_spec(specs, "decoder.conv1", cfg.z_dim, height, width)
      add_residual_cache_specs(specs, "decoder.middle.0", dims.first, dims.first, height, width)
      add_residual_cache_specs(specs, "decoder.middle.2", dims.first, dims.first, height, width)
      index = 0
      (0...cfg.dim_mult.size).each do |stage|
        source_in = dims[stage]
        out_channels = dims[stage + 1]
        in_channels = stage >= 1 ? source_in // 2 : source_in
        (cfg.num_res_blocks + 1).times do
          add_residual_cache_specs(specs, "decoder.upsamples.#{index}",
            in_channels, out_channels, height, width)
          in_channels = out_channels
          index += 1
        end
        if stage < cfg.dim_mult.size - 1
          if cfg.temporal_downsample.reverse[stage]
            add_cache_spec(specs, "decoder.upsamples.#{index}.time_conv",
              out_channels, height, width)
          end
          index += 1
          height *= 2
          width *= 2
          in_channels = out_channels // 2
        end
      end
      add_cache_spec(specs, "decoder.head.2", dims.last, height, width)
      specs
    end

    private def self.add_residual_cache_specs(specs : Array(CacheSpec), prefix : String,
                                              input : Int32, output : Int32,
                                              height : Int32, width : Int32) : Nil
      add_cache_spec(specs, "#{prefix}.residual.2", input, height, width)
      add_cache_spec(specs, "#{prefix}.residual.6", output, height, width)
    end

    private def self.add_cache_spec(specs : Array(CacheSpec), name : String,
                                    channels : Int32, height : Int32, width : Int32) : Nil
      specs << CacheSpec.new(name, channels, height, width)
    end
  end

  # Workspace cache-state registration helpers are kept outside the inference
  # module's public API; they are setup-only and never allocate during encode.
  class WanVAEWorkspace
    def add_cache_state(name : String, first : ML::MetalBuffer, second : ML::MetalBuffer,
                        frame_plane : Int32) : Nil
      @cache_states[name] = WanVAECachedState.new(first, second, frame_plane)
    end
  end
end
