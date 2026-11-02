require "./metal_ops"
require "./checkpoint"
require "./metal_mps_command_chain"
require "./dcae_v_mps"
require "./dcae_v_norm_mps"

module ML::Video
  private record DCAEVCacheSpec, name : String, channels : Int32,
    height : Int32, width : Int32
  private record DCAEVNormScratchSpec, name : String, values : Int32

  private module DCAEVShape
    def self.count(label : String, *factors : Int32) : Int32
      result = 1_i64
      factors.each do |factor|
        unless factor > 0 && result <= Int32::MAX.to_i64 // factor
          raise ArgumentError.new("DCAE-V #{label} exceeds the supported element count")
        end
        result *= factor
      end
      result.to_i32
    end

    def self.bytes(label : String, count : Int64) : Int64
      unless count >= 0 && count <= Int64::MAX // 2
        raise ArgumentError.new("DCAE-V #{label} byte count overflows")
      end
      count * 2
    end

    def self.add_bytes(label : String, total : Int64, additional : Int64) : Int64
      unless total >= 0 && additional >= 0 && additional <= Int64::MAX - total
        raise ArgumentError.new("DCAE-V #{label} byte count overflows")
      end
      total + additional
    end
  end

  # Geometry for the released f32t4c32 BF16 encoder. Full tiles use the
  # upstream 40-frame causal cache path; a final incomplete tile uses ordinary
  # symmetric zero padding, as in upstream ConvLayer3d.
  struct DCAEVEncoderConfig
    TILE_FRAMES = 40

    getter batches : Int32
    getter height : Int32
    getter width : Int32
    getter frames : Int32
    getter tile_frames : Int32

    def initialize(@height : Int32, @width : Int32, @frames : Int32 = 120,
                   @tile_frames : Int32 = TILE_FRAMES, @batches : Int32 = 1)
      unless @batches > 0 && @height > 0 && @width > 0 && @frames > 0
        raise ArgumentError.new("DCAE-V dimensions must be positive")
      end
      unless @height % 32 == 0 && @width % 32 == 0
        raise ArgumentError.new("DCAE-V height and width must be divisible by 32")
      end
      unless @tile_frames == TILE_FRAMES && @frames % 4 == 0
        raise ArgumentError.new("DCAE-V f32t4c32 requires 40-frame tiles and input frames divisible by 4")
      end
      DCAEVShape.count("input clip", @batches, 3, @frames, @height, @width)
      DCAEVShape.count("reference RGB frame", @height, @width, 3)
      DCAEVShape.count("latent output", @batches, 32, latent_frames, latent_height, latent_width)
      DCAEVShape.count("stage-one tile", @batches, 256, max_tile_frames,
        @height // 2, @width // 2)
    end

    def tile_count : Int32
      (@frames + @tile_frames - 1) // @tile_frames
    end

    def tile_frame_count(tile : Int32) : Int32
      unless 0 <= tile < tile_count
        raise ArgumentError.new("DCAE-V tile index is out of range")
      end
      {@tile_frames, @frames - tile * @tile_frames}.min
    end

    def max_tile_frames : Int32
      {@tile_frames, @frames}.min
    end

    def latent_frames : Int32
      @frames // 4
    end

    def latent_tile_frames : Int32
      @tile_frames // 4
    end

    def latent_height : Int32
      @height // 32
    end

    def latent_width : Int32
      @width // 32
    end

    def input_values : Int32
      DCAEVShape.count("input clip", @batches, 3, @frames, @height, @width)
    end

    def input_elements : Int32
      input_values
    end

    def input_bytes : Int64
      DCAEVShape.bytes("input clip", input_values.to_i64)
    end

    def latent_values : Int32
      DCAEVShape.count("latent output", @batches, 32,
        latent_frames, latent_height, latent_width)
    end

    def output_elements : Int32
      latent_values
    end

    def output_bytes : Int64
      DCAEVShape.bytes("latent output", latent_values.to_i64)
    end

    def rgb_bytes : Int64
      DCAEVShape.count("reference RGB frame", @height, @width, 3).to_i64
    end
  end

  # Optional diagnostic copies of the stage-five output, one per input tile.
  # The caller owns BF16 targets named `trace.stages.5.tileN` and reads them
  # only after the same terminal drain as the latent output.
  class DCAEVEncoderTrace
    getter buffers : Hash(String, ML::MetalBuffer)

    def initialize(@buffers : Hash(String, ML::MetalBuffer))
    end

    def buffer(name : String) : ML::MetalBuffer?
      @buffers[name]?
    end
  end

  # Owns three reusable BF16 tile activations and one BF16 last-frame cache per
  # chunk-causal convolution. All residency is admitted by prepare! before an
  # encoder command is submitted; reset affects only host-side cache validity.
  class DCAEVEncoderWorkspace
    getter arena : Arena
    getter config : DCAEVEncoderConfig
    getter scratch_values : Int32
    getter cache_values : Int64
    getter norm_values : Int64
    getter planned_bytes : Int64
    @cache_specs : Array(DCAEVCacheSpec)
    @norm_specs : Array(DCAEVNormScratchSpec)
    @cache_valid = {} of String => Bool

    def initialize(@arena : Arena, @config : DCAEVEncoderConfig,
                   @scratch_values : Int32, @cache_values : Int64,
                   @norm_values : Int64, @planned_bytes : Int64,
                   @cache_specs : Array(DCAEVCacheSpec),
                   @norm_specs : Array(DCAEVNormScratchSpec))
      @cache_specs.each { |spec| @cache_valid[spec.name] = false }
    end

    def scratch(index : Int32) : ML::MetalBuffer
      raise ArgumentError.new("Invalid DCAE-V scratch slot") unless 0 <= index < 3
      @arena["dcae_v.scratch.#{index}"]
    end

    def cache(name : String) : ML::MetalBuffer
      @arena["dcae_v.cache.#{name}"]
    end

    def norm_mean_a(name : String) : ML::MetalBuffer
      @arena["dcae_v.norm.#{name}.mean_a"]
    end

    def norm_mean_b(name : String) : ML::MetalBuffer
      @arena["dcae_v.norm.#{name}.mean_b"]
    end

    def cache_valid?(name : String) : Bool
      @cache_valid[name]? || false
    end

    def mark_cache_valid!(name : String) : Nil
      raise ArgumentError.new("Unknown DCAE-V feature cache #{name}") unless @cache_valid.has_key?(name)
      @cache_valid[name] = true
    end

    def reset_cache_state! : Nil
      @cache_valid.each_key { |name| @cache_valid[name] = false }
    end

    def cache_names : Array(String)
      @cache_specs.map(&.name)
    end

    def norm_scratch_names : Array(String)
      @norm_specs.map(&.name)
    end
  end

  # Owns every fixed MPSGraph binding for one encoder workspace. Release only
  # after the command chain that used these plans has completed its terminal
  # drain; private MPSGraph allocations are opaque to MetalBuffer accounting.
  class DCAEVEncoderMPSPlanSet
    getter caller_owned_extra_bytes : Int64 = 0_i64
    getter graph_private_bytes_known : Bool = false
    @conv_plans = {} of String => DCAEVConvMPS::Plan
    @rms_plans = {} of String => DCAEVNormMPS::RMSNormPlan
    @silu_plans = {} of String => DCAEVNormMPS::SiLUPlan
    @templates = {} of String => DCAEVConvMPS::Template
    @released = false
    @input : ML::MetalBuffer? = nil
    @latent : ML::MetalBuffer? = nil
    @weights : Arena? = nil
    @workspace : DCAEVEncoderWorkspace? = nil
    @config : DCAEVEncoderConfig? = nil

    def self.require_prepared!(plans : DCAEVEncoderMPSPlanSet?) : DCAEVEncoderMPSPlanSet
      plans || raise ArgumentError.new("DCAE-V encode requires prepared MPSGraph plans")
    end

    def bind_owner!(input : ML::MetalBuffer, latent : ML::MetalBuffer,
                    weights : Arena, workspace : DCAEVEncoderWorkspace,
                    config : DCAEVEncoderConfig) : Nil
      @input = input
      @latent = latent
      @weights = weights
      @workspace = workspace
      @config = config
    end

    def validate_owner!(input : ML::MetalBuffer, latent : ML::MetalBuffer,
                        weights : Arena, workspace : DCAEVEncoderWorkspace,
                        config : DCAEVEncoderConfig) : Nil
      unless !@released && @input.try(&.same?(input)) &&
             @latent.try(&.same?(latent)) && @weights.try(&.same?(weights)) &&
             @workspace.try(&.same?(workspace)) && @config == config
        raise ArgumentError.new("DCAE-V prepared MPS plans belong to different encoder buffers/config")
      end
    end

    def add_conv(key : String, plan : DCAEVConvMPS::Plan) : Nil
      raise ArgumentError.new("Duplicate DCAE-V MPS plan key #{key}") if @conv_plans.has_key?(key)
      @conv_plans[key] = plan
    end

    def add_norm(key : String, plan : DCAEVNormMPS::RMSNormPlan) : Nil
      raise ArgumentError.new("Duplicate DCAE-V RMS plan key #{key}") if @rms_plans.has_key?(key)
      @rms_plans[key] = plan
    end

    def add_silu(key : String, plan : DCAEVNormMPS::SiLUPlan) : Nil
      raise ArgumentError.new("Duplicate DCAE-V SiLU plan key #{key}") if @silu_plans.has_key?(key)
      @silu_plans[key] = plan
    end

    def add_template(key : String, template : DCAEVConvMPS::Template) : DCAEVConvMPS::Template
      @templates[key] ||= template
    end

    def template(key : String) : DCAEVConvMPS::Template?
      @templates[key]?
    end

    def conv(key : String) : DCAEVConvMPS::Plan
      @conv_plans[key]? || raise ArgumentError.new("Missing prepared DCAE-V convolution plan #{key}")
    end

    def norm(key : String) : DCAEVNormMPS::RMSNormPlan
      @rms_plans[key]? || raise ArgumentError.new("Missing prepared DCAE-V RMSNorm plan #{key}")
    end

    def silu(key : String) : DCAEVNormMPS::SiLUPlan
      @silu_plans[key]? || raise ArgumentError.new("Missing prepared DCAE-V SiLU plan #{key}")
    end

    def template_count : Int32
      @templates.size.to_i32
    end

    def conv_plan_count : Int32
      @conv_plans.size.to_i32
    end

    def norm_plan_count : Int32
      @rms_plans.size.to_i32
    end

    def silu_plan_count : Int32
      @silu_plans.size.to_i32
    end

    def release(chain : ML::Metal::MPSCommandChain? = nil) : Nil
      return if @released
      if active = chain
        raise ArgumentError.new("DCAE-V plan set chain has not drained") unless active.completed?
      end
      @conv_plans.each_value { |plan| plan.release(chain) }
      @rms_plans.each_value { |plan| plan.release(chain) }
      @silu_plans.each_value { |plan| plan.release(chain) }
      @templates.each_value(&.release)
      @conv_plans.clear
      @rms_plans.clear
      @silu_plans.clear
      @templates.clear
      @input = nil
      @latent = nil
      @weights = nil
      @workspace = nil
      @config = nil
      @released = true
    end
  end

  # Semantic convolution roles determine checkpoint bias presence. Keeping the
  # role in each prepared operation plan prevents positional Boolean drift.
  enum DCAEVConvRole
    ProjectIn
    ResidualConv1
    ResidualConv2
    Downsample
    ProjectOut
  end

  module DCAEVEncoder
    private record DCAEVWeightSpec, name : String, shape : Array(Int64)

    SOURCE                    = {{ read_file("#{__DIR__}/kernels/dcae_v_encoder.metal") }}
    WIDTHS                    = [128, 256, 512, 512, 1024, 1024, 1024]
    DEPTHS                    = [0, 5, 10, 4, 4, 4, 4]
    NORM_EPSILON              = 1.0e-5_f32
    PROJECT_IN_WEIGHT_PREFIX  = "encoder.project_in.conv.conv"
    PROJECT_OUT_WEIGHT_PREFIX = "encoder.project_out.conv"

    def self.pipeline(name : String) : ML::Metal::ComputePipeline
      ML::Metal::PipelineCache.get(name) { ML::Metal::ComputePipeline.new(name, SOURCE) }
    end

    def self.prepare! : Nil
      MetalOps.prepare!
      {"dcae_v_normalize_repeat_bf16", "dcae_v_conv3d_bf16",
       "dcae_v_conv_unshuffle_bf16", "dcae_v_rms_norm_bf16_inplace",
       "dcae_v_residual_add_bf16_inplace", "dcae_v_cache_last_frame_bf16",
       "dcae_v_copy_bf16", "dcae_v_scatter_latent_bf16"}.each { |name| pipeline(name) }
    end

    # Names and tensor shapes mirror the upstream DCAEVEncoder state_dict. All
    # weights are original BF16; no checkpoint key rewriting or extra model
    # prefix is admitted here.
    def self.checkpoint_bindings(cfg : DCAEVEncoderConfig) : Array(WeightBinding)
      dispatch_weight_specs(cfg).map do |spec|
        WeightBinding.new(spec.name, spec.name, spec.shape, ML::GGUF::SafeTensorDType::BF16)
      end
    end

    # Pure CPU inventory of every original checkpoint tensor the runtime
    # dispatches. Runtime path construction uses the same prefix helpers.
    def self.dispatch_weight_names(cfg : DCAEVEncoderConfig) : Array(String)
      dispatch_weight_specs(cfg).map(&.name)
    end

    # Exact encoded weight residency and temporary residency, both available
    # before a Metal buffer or command is created.
    def self.weight_bytes(cfg : DCAEVEncoderConfig) : Int64
      checkpoint_bindings(cfg).reduce(0_i64) do |total, binding|
        count = binding.shape.product
        bytes = DCAEVShape.bytes("checkpoint weights", count)
        DCAEVShape.add_bytes("checkpoint weights", total, bytes)
      end
    end

    def self.trace_bytes(cfg : DCAEVEncoderConfig) : Int64
      values = DCAEVShape.count("stage-five trace", cfg.batches, 1024,
        cfg.latent_frames, cfg.latent_height, cfg.latent_width)
      DCAEVShape.bytes("stage-five trace", values.to_i64)
    end

    def self.project_in_trace_bytes(cfg : DCAEVEncoderConfig, tile : Int32 = 0) : Int64
      frames = cfg.tile_frame_count(tile)
      values = DCAEVShape.count("project-in trace", cfg.batches, 256,
        frames, cfg.height // 2, cfg.width // 2)
      DCAEVShape.bytes("project-in trace", values.to_i64)
    end

    def self.workspace_required_bytes(cfg : DCAEVEncoderConfig) : Int64
      scratch = scratch_values(cfg).to_i64 * 3
      cache = cache_values(cfg)
      norm = norm_values(cfg).to_i64 * 2
      bytes = DCAEVShape.add_bytes("workspace", DCAEVShape.bytes("activation scratch", scratch),
        DCAEVShape.bytes("feature caches", cache))
      DCAEVShape.add_bytes("workspace", bytes, DCAEVShape.bytes("norm scratch", norm))
    end

    # Plan all scratch and feature-cache buffers before allocation. The caller's
    # reserve must account for checkpoint weights, the input clip, latent output,
    # and any concurrently resident runtime state.
    def self.prepare!(workspace_arena : Arena, cfg : DCAEVEncoderConfig,
                      budget_bytes : Int64, reserve_bytes : Int64 = 0_i64) : DCAEVEncoderWorkspace
      raise ArgumentError.new("DCAE-V workspace arena must start empty") unless workspace_arena.resident_bytes == 0
      unless budget_bytes > 0 && reserve_bytes >= 0 && reserve_bytes <= budget_bytes
        raise ArgumentError.new("Invalid DCAE-V residency budget")
      end

      scratch_values = scratch_values(cfg)
      specs = cache_specs(cfg)
      cache_values = cache_values(cfg)
      norm_specs = norm_specs(cfg)
      norm_values = norm_values(cfg)
      planned = workspace_required_bytes(cfg)
      unless planned <= budget_bytes && reserve_bytes <= budget_bytes - planned
        raise ArgumentError.new("DCAE-V workspace exceeds the residency budget")
      end

      prepare!
      allocated = [] of String
      begin
        3.times do |index|
          name = "dcae_v.scratch.#{index}"
          workspace_arena.alloc_weight(name, scratch_values, WeightDType::BF16)
          allocated << name
        end
        specs.each do |spec|
          name = "dcae_v.cache.#{spec.name}"
          count = DCAEVShape.count("feature cache", cfg.batches,
            spec.channels, spec.height, spec.width)
          workspace_arena.alloc_weight(name, count, WeightDType::BF16)
          allocated << name
        end
        norm_specs.each do |spec|
          {"mean_a", "mean_b"}.each do |slot|
            name = "dcae_v.norm.#{spec.name}.#{slot}"
            workspace_arena.alloc_weight(name, spec.values, WeightDType::BF16)
            allocated << name
          end
        end
      rescue ex
        allocated.each { |name| workspace_arena.remove(name) }
        raise ex
      end
      unless workspace_arena.resident_bytes == planned
        raise ArgumentError.new("DCAE-V workspace allocation differs from its plan")
      end
      DCAEVEncoderWorkspace.new(workspace_arena, cfg, scratch_values,
        cache_values, norm_values, planned, specs, norm_specs)
    end

    # Compile and bind the MPSGraph operation set after all caller-owned
    # buffers are resident, but before any timed command is encoded.
    def self.prepare_mps!(input : ML::MetalBuffer, latent : ML::MetalBuffer,
                          weights : Arena, workspace : DCAEVEncoderWorkspace,
                          cfg : DCAEVEncoderConfig) : DCAEVEncoderMPSPlanSet
      validate_workspace!(workspace, cfg)
      validate_weights!(weights, cfg)
      validate_residency!(weights, workspace)
      MetalOps.check_storage(input, cfg.input_values.to_i64, WeightDType::BF16)
      MetalOps.check_storage(latent, cfg.latent_values.to_i64, WeightDType::BF16)
      MetalOps.distinct(latent, input)
      validate_external_buffer!(input, weights, workspace)
      validate_external_buffer!(latent, weights, workspace)

      plans = DCAEVEncoderMPSPlanSet.new
      begin
        cfg.tile_count.times do |tile|
          frames = cfg.tile_frame_count(tile)
          chunk_causal = frames == cfg.tile_frames
          current_index = 0_i32
          current_channels = 256_i32
          current_frames = frames
          current_height = cfg.height // 2
          current_width = cfg.width // 2

          add_conv_plan(plans, "tile#{tile}:project_in", input, workspace,
            weights, cfg, PROJECT_IN_WEIGHT_PREFIX, "encoder.project_in.conv",
            3, 256, 64, frames, cfg.height, cfg.width,
            tile * cfg.tile_frames, cfg.frames, true, chunk_causal, tile > 0,
            DCAEVConvRole::ProjectIn, 2, 1, false, workspace.scratch(0))

          (1..6).each do |stage|
            width = WIDTHS[stage]
            DEPTHS[stage].times do |block|
              prefix = stage_block_prefix(stage, block)
              conv1_output_index = (current_index + 1) % 3
              conv1_temp_index = (current_index + 2) % 3
              conv1_name = residual_conv_cache_name(prefix, "conv1")
              add_conv_plan(plans, "tile#{tile}:#{prefix}.conv1",
                workspace.scratch(current_index), workspace, weights, cfg,
                residual_conv_weight_prefix(prefix, "conv1"), conv1_name,
                width, width, width, current_frames, current_height, current_width,
                0, current_frames, false, chunk_causal, tile > 0,
                DCAEVConvRole::ResidualConv1, 1, 1, false,
                workspace.scratch(conv1_temp_index))

              plans.add_silu("tile#{tile}:#{prefix}.silu",
                DCAEVNormMPS::SiLUPlan.new(
                  [cfg.batches, width, current_frames, current_height, current_width],
                  workspace.scratch(conv1_temp_index),
                  workspace.scratch(conv1_output_index)))

              conv2_output_index = conv1_temp_index
              conv2_name = residual_conv_cache_name(prefix, "conv2")
              add_conv_plan(plans, "tile#{tile}:#{prefix}.conv2",
                workspace.scratch(conv1_output_index), workspace, weights, cfg,
                residual_conv_weight_prefix(prefix, "conv2"), conv2_name,
                width, width, width, current_frames, current_height, current_width,
                0, current_frames, false, chunk_causal, tile > 0,
                DCAEVConvRole::ResidualConv2, 1, 1, false,
                workspace.scratch(conv2_output_index))

              norm_prefix = residual_norm_prefix(prefix)
              plans.add_norm("tile#{tile}:#{prefix}.norm",
                DCAEVNormMPS::RMSNormPlan.new(
                  [cfg.batches, width, current_frames, current_height, current_width],
                  workspace.scratch(conv2_output_index),
                  workspace.scratch(conv1_output_index),
                  workspace.norm_mean_a("shared"), workspace.norm_mean_b("shared"),
                  weights["#{norm_prefix}.weight"], weights["#{norm_prefix}.bias"],
                  NORM_EPSILON))
              current_index = conv1_output_index
            end

            if stage < 6
              spatial_factor = stage < 5 ? 2 : 1
              temporal_factor = stage == 5 ? 4 : 1
              out_channels = WIDTHS[stage + 1]
              factor = spatial_factor * spatial_factor * temporal_factor
              conv_channels = out_channels // factor
              downsample = downsample_weight_prefix(stage)
              cache_name = downsample_cache_name(stage)
              output_index = (current_index + 1) % 3
              add_conv_plan(plans, "tile#{tile}:stage#{stage}.downsample",
                workspace.scratch(current_index), workspace, weights, cfg,
                downsample, cache_name, current_channels, out_channels,
                conv_channels, current_frames, current_height, current_width,
                0, current_frames, false, chunk_causal, tile > 0,
                DCAEVConvRole::Downsample,
                spatial_factor, temporal_factor, true, workspace.scratch(output_index))
              current_index = output_index
              current_channels = out_channels
              current_frames //= temporal_factor
              current_height //= spatial_factor
              current_width //= spatial_factor
            end
          end

          project_out_index = (current_index + 1) % 3
          add_conv_plan(plans, "tile#{tile}:project_out",
            workspace.scratch(current_index), workspace, weights, cfg,
            PROJECT_OUT_WEIGHT_PREFIX, "encoder.project_out",
            1024, 32, 32, current_frames, current_height, current_width,
            0, current_frames, false, chunk_causal, tile > 0,
            DCAEVConvRole::ProjectOut,
            1, 1, false, workspace.scratch(project_out_index))
        end
        plans.bind_owner!(input, latent, weights, workspace, cfg)
        plans
      rescue ex
        plans.release
        raise ex
      end
    end

    # GPU-only reference-image ingestion. `rgb8_hwc` is one fitted RGB frame;
    # `clip_bf16_bcthw` is caller-owned storage for the repeated normalized clip.
    # No allocation, readback, or wait occurs in this method.
    def self.encode_image_pixels!(e : ML::Metal::ComputeEncoder,
                                  rgb8_hwc : ML::MetalBuffer,
                                  clip_bf16_bcthw : ML::MetalBuffer,
                                  cfg : DCAEVEncoderConfig) : Nil
      validate_byte_buffer!(rgb8_hwc, cfg.rgb_bytes, "DCAE-V RGB image")
      MetalOps.check_storage(clip_bf16_bcthw, cfg.input_values.to_i64, WeightDType::BF16)
      MetalOps.distinct(clip_bf16_bcthw, rgb8_hwc)
      e.set_pipeline(pipeline("dcae_v_normalize_repeat_bf16"))
      e.set_buffer(rgb8_hwc, 0).set_buffer(clip_bf16_bcthw, 1, ML::Metal::BufferAccess::Write)
      e.set_value(cfg.batches.to_u32, 2).set_value(cfg.frames.to_u32, 3)
      e.set_value(cfg.height.to_u32, 4).set_value(cfg.width.to_u32, 5)
      e.dispatch_1d(cfg.input_values)
    end

    # Encode the complete input clip through prepared MPSGraph operations,
    # carrying one preceding frame for each chunk-causal convolution. `input`
    # and `latent` are caller-owned BF16 BCTHW buffers. With an MPSCommandChain,
    # commits are asynchronous continuations; only the caller performs the
    # terminal wait after the whole GPU phase.
    def self.encode(e : ML::Metal::ComputeEncoder, input : ML::MetalBuffer,
                    latent : ML::MetalBuffer, weights : Arena,
                    workspace : DCAEVEncoderWorkspace, cfg : DCAEVEncoderConfig,
                    execution_chain : ML::Metal::MPSCommandChain? = nil,
                    trace : DCAEVEncoderTrace? = nil,
                    mps_plans : DCAEVEncoderMPSPlanSet? = nil) : Int32
      plans = DCAEVEncoderMPSPlanSet.require_prepared!(mps_plans)
      validate_workspace!(workspace, cfg)
      validate_weights!(weights, cfg)
      validate_residency!(weights, workspace)
      MetalOps.check_storage(input, cfg.input_values.to_i64, WeightDType::BF16)
      MetalOps.check_storage(latent, cfg.latent_values.to_i64, WeightDType::BF16)
      MetalOps.distinct(latent, input)
      validate_external_buffer!(input, weights, workspace)
      validate_external_buffer!(latent, weights, workspace)
      validate_trace!(trace, cfg, input, latent, weights, workspace)

      workspace.reset_cache_state!
      plans.validate_owner!(input, latent, weights, workspace, cfg)
      chain = execution_chain || raise ArgumentError.new("DCAE-V MPSGraph encode requires an MPS command chain")
      encode_mps(chain, input, latent, weights, workspace, cfg, trace, plans)
    end

    private def self.encode_mps(chain : ML::Metal::MPSCommandChain,
                                input : ML::MetalBuffer, latent : ML::MetalBuffer,
                                weights : Arena, workspace : DCAEVEncoderWorkspace,
                                cfg : DCAEVEncoderConfig, trace : DCAEVEncoderTrace?,
                                plans : DCAEVEncoderMPSPlanSet) : Int32
      cfg.tile_count.times do |tile|
        tile_frames = cfg.tile_frame_count(tile)
        chunk_causal = tile_frames == cfg.tile_frames
        current_index = 0_i32
        current_buffer = workspace.scratch(0)
        current_channels = 256_i32
        current_frames = tile_frames
        current_height = cfg.height // 2
        current_width = cfg.width // 2

        plans.conv("tile#{tile}:project_in").encode(chain)
        e = chain.encoder
        if chunk_causal
          e = encode_cache_update(e, input, workspace.cache("encoder.project_in.conv"),
            cfg, 3, tile_frames, cfg.height, cfg.width, tile * cfg.tile_frames,
            "encoder.project_in.conv", workspace, chain, true)
        end
        capture_trace(e, trace, "trace.project_in.tile#{tile}", current_buffer,
          cfg, 256, tile_frames, current_height, current_width, chain)

        (1..6).each do |stage|
          width = WIDTHS[stage]
          DEPTHS[stage].times do |block|
            prefix = stage_block_prefix(stage, block)
            current_buffer, current_index, e = encode_residual_block_mps(
              chain, current_buffer, current_index, workspace, cfg, prefix,
              width, current_frames, current_height, current_width,
              chunk_causal, tile, plans)
          end

          if stage < 6
            spatial_factor = stage < 5 ? 2 : 1
            temporal_factor = stage == 5 ? 4 : 1
            out_channels = WIDTHS[stage + 1]
            output_index = (current_index + 1) % 3
            input_buffer = current_buffer
            cache_name = downsample_cache_name(stage)
            plans.conv("tile#{tile}:stage#{stage}.downsample").encode(chain)
            e = chain.encoder
            if chunk_causal
              e = encode_cache_update(e, input_buffer, workspace.cache(cache_name),
                cfg, current_channels, current_frames, current_height, current_width,
                0, cache_name, workspace, chain)
            end
            current_index = output_index
            current_buffer = workspace.scratch(current_index)
            current_channels = out_channels
            current_frames //= temporal_factor
            current_height //= spatial_factor
            current_width //= spatial_factor
            e = continue_execution!(chain.encoder, chain)
          end

          if stage == 5
            capture_trace(chain.encoder, trace, "trace.stages.5.tile#{tile}",
              current_buffer, cfg, current_channels, current_frames,
              current_height, current_width, chain)
          end
        end

        project_input = current_buffer
        plans.conv("tile#{tile}:project_out").encode(chain)
        e = chain.encoder
        if chunk_causal
          e = encode_cache_update(e, project_input, workspace.cache("encoder.project_out"),
            cfg, 1024, current_frames, current_height, current_width,
            0, "encoder.project_out", workspace, chain)
        end
        project_out_index = (current_index + 1) % 3
        encode_latent_scatter(e, workspace.scratch(project_out_index), latent,
          cfg, current_frames, tile, chain)
        if tile + 1 < cfg.tile_count
          continue_execution!(chain.encoder, chain)
        end
      end
      cfg.latent_frames
    end

    private def self.encode_residual_block_mps(chain : ML::Metal::MPSCommandChain,
                                               input : ML::MetalBuffer,
                                               input_index : Int32,
                                               workspace : DCAEVEncoderWorkspace,
                                               cfg : DCAEVEncoderConfig,
                                               prefix : String, channels : Int32,
                                               frames : Int32, height : Int32, width : Int32,
                                               chunk_causal : Bool, tile : Int32,
                                               plans : DCAEVEncoderMPSPlanSet) : {ML::MetalBuffer, Int32, ML::Metal::ComputeEncoder}
      conv1_output_index = (input_index + 1) % 3
      conv1_temp_index = (input_index + 2) % 3
      conv1_name = residual_conv_cache_name(prefix, "conv1")
      plans.conv("tile#{tile}:#{prefix}.conv1").encode(chain)
      e = chain.encoder
      if chunk_causal
        e = encode_cache_update(e, input, workspace.cache(conv1_name), cfg,
          channels, frames, height, width, 0, conv1_name, workspace, chain)
      end

      plans.silu("tile#{tile}:#{prefix}.silu").encode(chain)
      conv1_output = workspace.scratch(conv1_output_index)
      conv2_output_index = conv1_temp_index
      conv2_output = workspace.scratch(conv2_output_index)
      conv2_name = residual_conv_cache_name(prefix, "conv2")
      plans.conv("tile#{tile}:#{prefix}.conv2").encode(chain)
      e = chain.encoder
      if chunk_causal
        e = encode_cache_update(e, conv1_output, workspace.cache(conv2_name), cfg,
          channels, frames, height, width, 0, conv2_name, workspace, chain)
      end
      plans.norm("tile#{tile}:#{prefix}.norm").encode(chain)

      e = chain.encoder
      count = DCAEVShape.count("residual activation", cfg.batches,
        channels, frames, height, width)
      MetalOps.check_storage(input, count.to_i64, WeightDType::BF16)
      MetalOps.check_storage(conv1_output, count.to_i64, WeightDType::BF16)
      MetalOps.distinct(input, conv1_output)
      e.set_pipeline(pipeline("dcae_v_residual_add_bf16_inplace"))
      e.set_buffer(conv1_output, 0, ML::Metal::BufferAccess::ReadWrite)
      e.set_buffer(input, 1)
      e.set_value(count.to_u32, 2)
      e.dispatch_1d(count)
      e = continue_execution!(e, chain)
      {conv1_output, conv1_output_index, e}
    end

    private def self.encode_latent_scatter(e : ML::Metal::ComputeEncoder,
                                           source : ML::MetalBuffer,
                                           latent : ML::MetalBuffer,
                                           cfg : DCAEVEncoderConfig,
                                           tile_frames : Int32, tile : Int32,
                                           chain : ML::Metal::MPSCommandChain?) : ML::Metal::ComputeEncoder
      source_values = DCAEVShape.count("project-out tile", cfg.batches, 32,
        tile_frames, cfg.latent_height, cfg.latent_width)
      MetalOps.check_storage(source, source_values.to_i64, WeightDType::BF16)
      MetalOps.check_storage(latent, cfg.latent_values.to_i64, WeightDType::BF16)
      MetalOps.distinct(source, latent)
      e = current_encoder(e, chain)
      e.set_pipeline(pipeline("dcae_v_scatter_latent_bf16"))
      e.set_buffer(source, 0).set_buffer(latent, 1, ML::Metal::BufferAccess::Write)
      e.set_value(cfg.batches.to_u32, 2).set_value(32_u32, 3)
      e.set_value(tile_frames.to_u32, 4).set_value(cfg.latent_frames.to_u32, 5)
      e.set_value(cfg.latent_height.to_u32, 6).set_value(cfg.latent_width.to_u32, 7)
      e.set_value((tile * cfg.latent_tile_frames).to_u32, 8)
      e.dispatch_1d(source_values)
      e
    end

    private def self.dispatch_weight_specs(cfg : DCAEVEncoderConfig) : Array(DCAEVWeightSpec)
      specs = [] of DCAEVWeightSpec
      add_conv_weight_specs(specs, PROJECT_IN_WEIGHT_PREFIX, 3, 64,
        conv_role_uses_bias?(DCAEVConvRole::ProjectIn))

      (1..6).each do |stage|
        width = WIDTHS[stage]
        DEPTHS[stage].times do |block|
          prefix = stage_block_prefix(stage, block)
          add_conv_weight_specs(specs, residual_conv_weight_prefix(prefix, "conv1"),
            width, width, conv_role_uses_bias?(DCAEVConvRole::ResidualConv1))
          add_conv_weight_specs(specs, residual_conv_weight_prefix(prefix, "conv2"),
            width, width, conv_role_uses_bias?(DCAEVConvRole::ResidualConv2))
          add_norm_weight_specs(specs, residual_norm_prefix(prefix), width)
        end
        if stage < 6
          add_conv_weight_specs(specs, downsample_weight_prefix(stage),
            width, WIDTHS[stage + 1] // 4,
            conv_role_uses_bias?(DCAEVConvRole::Downsample))
        end
      end
      add_conv_weight_specs(specs, PROJECT_OUT_WEIGHT_PREFIX, 1024, 32,
        conv_role_uses_bias?(DCAEVConvRole::ProjectOut))
      specs
    end

    def self.conv_role_uses_bias?(role : DCAEVConvRole) : Bool
      case role
      when DCAEVConvRole::ProjectIn, DCAEVConvRole::ResidualConv1,
           DCAEVConvRole::Downsample, DCAEVConvRole::ProjectOut
        true
      when DCAEVConvRole::ResidualConv2
        false
      else
        raise ArgumentError.new("Unknown DCAE-V convolution role")
      end
    end

    private def self.add_conv_weight_specs(specs : Array(DCAEVWeightSpec), prefix : String,
                                           in_channels : Int32, out_channels : Int32,
                                           use_bias : Bool) : Nil
      specs << DCAEVWeightSpec.new("#{prefix}.weight",
        [out_channels.to_i64, in_channels.to_i64, 3_i64, 3_i64, 3_i64])
      if use_bias
        specs << DCAEVWeightSpec.new("#{prefix}.bias", [out_channels.to_i64])
      end
    end

    private def self.add_norm_weight_specs(specs : Array(DCAEVWeightSpec), prefix : String,
                                           channels : Int32) : Nil
      specs << DCAEVWeightSpec.new("#{prefix}.weight", [channels.to_i64])
      specs << DCAEVWeightSpec.new("#{prefix}.bias", [channels.to_i64])
    end

    private def self.add_conv_plan(plans : DCAEVEncoderMPSPlanSet, operation_key : String,
                                   input : ML::MetalBuffer,
                                   workspace : DCAEVEncoderWorkspace,
                                   weights : Arena, cfg : DCAEVEncoderConfig,
                                   weight_prefix : String, cache_name : String,
                                   in_channels : Int32, output_channels : Int32,
                                   conv_channels : Int32, frames : Int32,
                                   height : Int32, width : Int32,
                                   input_frame_offset : Int32, source_frames : Int32,
                                   full_clip : Bool, chunk_causal : Bool,
                                   cache_requested : Bool, role : DCAEVConvRole,
                                   spatial_factor : Int32, temporal_factor : Int32,
                                   shortcut : Bool, output : ML::MetalBuffer) : Nil
      use_bias = conv_role_uses_bias?(role)
      cache_input = chunk_causal && cache_requested
      weight_name = "#{weight_prefix}.weight"
      bias_name = "#{weight_prefix}.bias"
      weight_count = conv_channels.to_i64 * in_channels * 27
      weights.check_weight(weight_name, weight_count)
      if use_bias
        weights.check_weight(bias_name, conv_channels.to_i64)
      end
      cache = cache_input ? workspace.cache(cache_name) : nil
      cache_values = DCAEVShape.count("feature cache", cfg.batches,
        in_channels, height, width)
      MetalOps.check_storage(input,
        DCAEVShape.count("MPSGraph source", cfg.batches, in_channels,
          full_clip ? source_frames : frames, height, width).to_i64,
        WeightDType::BF16)
      if cache
        MetalOps.check_storage(cache, cache_values.to_i64, WeightDType::BF16)
      end
      MetalOps.check_storage(output,
        DCAEVShape.count("MPSGraph output", cfg.batches, output_channels,
          frames // temporal_factor, height // spatial_factor,
          width // spatial_factor).to_i64, WeightDType::BF16)
      MetalOps.distinct(output, input, cache) if cache
      MetalOps.distinct(output, input) unless cache

      template_key = [cfg.batches, in_channels, conv_channels, output_channels,
                      frames, source_frames, height, width, input_frame_offset,
                      full_clip, chunk_causal, cache_input, use_bias,
                      spatial_factor, temporal_factor, shortcut].map(&.to_s).join("|")
      template = plans.template(template_key)
      unless template
        begin
          template = DCAEVConvMPS::Template.new(
            cfg.batches, in_channels, conv_channels, output_channels,
            frames, source_frames, height, width, input_frame_offset,
            full_clip, chunk_causal, cache_input, use_bias,
            spatial_factor, temporal_factor, shortcut)
        rescue ex
          raise "#{ex.message} (operation=#{operation_key}, weights=#{weight_prefix})"
        end
        plans.add_template(template_key, template)
      end
      bias = use_bias ? weights[bias_name] : nil
      plans.add_conv(operation_key,
        template.bind(input, cache, weights[weight_name], bias, output))
    end

    private def self.stage_block_prefix(stage : Int32, block : Int32) : String
      "encoder.stages.#{stage}.op_list.#{block}.main"
    end

    private def self.residual_conv_cache_name(prefix : String, branch : String) : String
      "#{prefix}.#{branch}"
    end

    private def self.residual_conv_weight_prefix(prefix : String, branch : String) : String
      "#{residual_conv_cache_name(prefix, branch)}.conv"
    end

    private def self.residual_norm_prefix(prefix : String) : String
      "#{prefix}.conv2.norm"
    end

    private def self.downsample_cache_name(stage : Int32) : String
      "encoder.stages.#{stage}.op_list.#{DEPTHS[stage]}.main.conv"
    end

    private def self.downsample_weight_prefix(stage : Int32) : String
      "#{downsample_cache_name(stage)}.conv"
    end

    private def self.scratch_values(cfg : DCAEVEncoderConfig) : Int32
      DCAEVShape.count("tile activation", cfg.batches, 256, cfg.max_tile_frames,
        cfg.height // 2, cfg.width // 2)
    end

    # RMSNorm's MPS plans run serially, so one shared pair sized for the largest
    # BTHW row set is sufficient for every residual block and tile.
    private def self.norm_specs(cfg : DCAEVEncoderConfig) : Array(DCAEVNormScratchSpec)
      maximum = 0_i32
      cfg.tile_count.times do |tile|
        frames = cfg.tile_frame_count(tile)
        (1..6).each do |stage|
          divisor = case stage
                    when 1 then 2
                    when 2 then 4
                    when 3 then 8
                    when 4 then 16
                    else        32
                    end
          norm_frames = stage == 6 ? frames // 4 : frames
          rows = DCAEVShape.count("RMSNorm rows", cfg.batches,
            norm_frames, cfg.height // divisor, cfg.width // divisor)
          maximum = rows if rows > maximum
        end
      end
      [DCAEVNormScratchSpec.new("shared", maximum)]
    end

    private def self.norm_values(cfg : DCAEVEncoderConfig) : Int32
      norm_specs(cfg).max_of(&.values)
    end

    private def self.cache_values(cfg : DCAEVEncoderConfig) : Int64
      cache_specs(cfg).reduce(0_i64) do |total, spec|
        frame_values = DCAEVShape.count("feature cache", cfg.batches,
          spec.channels, spec.height, spec.width).to_i64
        unless frame_values <= Int64::MAX - total
          raise ArgumentError.new("DCAE-V feature cache count overflows")
        end
        total + frame_values
      end
    end

    private def self.cache_specs(cfg : DCAEVEncoderConfig) : Array(DCAEVCacheSpec)
      specs = [] of DCAEVCacheSpec
      specs << DCAEVCacheSpec.new("encoder.project_in.conv", 3, cfg.height, cfg.width)
      (1..6).each do |stage|
        spatial_divisor = case stage
                          when 1 then 2
                          when 2 then 4
                          when 3 then 8
                          when 4 then 16
                          else        32
                          end
        height = cfg.height // spatial_divisor
        width = cfg.width // spatial_divisor
        channels = WIDTHS[stage]
        DEPTHS[stage].times do |block|
          prefix = stage_block_prefix(stage, block)
          specs << DCAEVCacheSpec.new(residual_conv_cache_name(prefix, "conv1"), channels, height, width)
          specs << DCAEVCacheSpec.new(residual_conv_cache_name(prefix, "conv2"), channels, height, width)
        end
        if stage < 6
          specs << DCAEVCacheSpec.new(downsample_cache_name(stage), channels, height, width)
        end
      end
      specs << DCAEVCacheSpec.new("encoder.project_out", 1024,
        cfg.latent_height, cfg.latent_width)
      specs
    end

    private def self.encode_project_in(e : ML::Metal::ComputeEncoder,
                                       input : ML::MetalBuffer, weights : Arena,
                                       workspace : DCAEVEncoderWorkspace,
                                       cfg : DCAEVEncoderConfig, frame_offset : Int32,
                                       frames : Int32, chunk_causal : Bool,
                                       chain : ML::Metal::MPSCommandChain?) : {ML::MetalBuffer, Int32}
      output = workspace.scratch(0)
      encode_conv_unshuffle(current_encoder(e, chain), input, output, weights,
        workspace, cfg, PROJECT_IN_WEIGHT_PREFIX, "encoder.project_in.conv",
        3, 256, frames, cfg.height, cfg.width, 2, 1, true,
        chunk_causal, frame_offset, frames, 0, chain)
      {output, 0}
    end

    private def self.encode_residual_block(e : ML::Metal::ComputeEncoder,
                                           input : ML::MetalBuffer, input_index : Int32,
                                           weights : Arena, workspace : DCAEVEncoderWorkspace,
                                           cfg : DCAEVEncoderConfig, prefix : String,
                                           channels : Int32, frames : Int32,
                                           height : Int32, width : Int32,
                                           chunk_causal : Bool,
                                           chain : ML::Metal::MPSCommandChain?) : {ML::MetalBuffer, Int32, ML::Metal::ComputeEncoder}
      conv1_name = residual_conv_cache_name(prefix, "conv1")
      conv1_output_index = (input_index + 1) % 3
      conv1_output = workspace.scratch(conv1_output_index)
      e = encode_conv(e, input, conv1_output, weights, workspace, cfg,
        residual_conv_weight_prefix(prefix, "conv1"), conv1_name, channels, channels, frames,
        height, width, 0, frames, 0, true, true, chunk_causal, chain)

      conv2_name = residual_conv_cache_name(prefix, "conv2")
      conv2_output_index = (input_index + 2) % 3
      conv2_output = workspace.scratch(conv2_output_index)
      e = encode_conv(e, conv1_output, conv2_output, weights, workspace, cfg,
        residual_conv_weight_prefix(prefix, "conv2"), conv2_name, channels, channels, frames,
        height, width, 0, frames, 0, false, false, chunk_causal, chain)

      norm_prefix = residual_norm_prefix(prefix)
      weight_name = "#{norm_prefix}.weight"
      bias_name = "#{norm_prefix}.bias"
      e = current_encoder(e, chain)
      count = DCAEVShape.count("residual activation", cfg.batches,
        channels, frames, height, width)
      weight_count = channels.to_i64
      weights.check_weight(weight_name, weight_count)
      weights.check_weight(bias_name, weight_count)
      e.set_pipeline(pipeline("dcae_v_rms_norm_bf16_inplace"))
      e.set_buffer(conv2_output, 0, ML::Metal::BufferAccess::ReadWrite)
      e.set_buffer(weights[weight_name], 1).set_buffer(weights[bias_name], 2)
      e.set_value(cfg.batches.to_u32, 3).set_value(channels.to_u32, 4)
      e.set_value(frames.to_u32, 5).set_value(height.to_u32, 6)
      e.set_value(width.to_u32, 7).set_value(NORM_EPSILON, 8)
      locations = DCAEVShape.count("RMSNorm locations", cfg.batches, frames, height, width)
      e.dispatch_1d(locations)

      e = current_encoder(e, chain)
      MetalOps.check_storage(input, count.to_i64, WeightDType::BF16)
      MetalOps.check_storage(conv2_output, count.to_i64, WeightDType::BF16)
      MetalOps.distinct(input, conv2_output)
      e.set_pipeline(pipeline("dcae_v_residual_add_bf16_inplace"))
      e.set_buffer(conv2_output, 0, ML::Metal::BufferAccess::ReadWrite)
      e.set_buffer(input, 1)
      e.set_value(count.to_u32, 2)
      e.dispatch_1d(count)
      e = continue_execution!(e, chain)
      {conv2_output, conv2_output_index, e}
    end

    private def self.encode_conv(e : ML::Metal::ComputeEncoder,
                                 input : ML::MetalBuffer, output : ML::MetalBuffer,
                                 weights : Arena, workspace : DCAEVEncoderWorkspace,
                                 cfg : DCAEVEncoderConfig, weight_prefix : String,
                                 cache_name : String, in_channels : Int32,
                                 out_channels : Int32, frames : Int32,
                                 height : Int32, width : Int32,
                                 input_frame_offset : Int32, output_total_frames : Int32,
                                 output_frame_offset : Int32, use_bias : Bool,
                                 use_silu : Bool,
                                 chunk_causal : Bool,
                                 chain : ML::Metal::MPSCommandChain?) : ML::Metal::ComputeEncoder
      weight_name = "#{weight_prefix}.weight"
      bias_name = "#{weight_prefix}.bias"
      weight_count = out_channels.to_i64 * in_channels * 27
      weights.check_weight(weight_name, weight_count)
      weights.check_weight(bias_name, out_channels.to_i64) if use_bias
      cache = workspace.cache(cache_name)
      cache_values = DCAEVShape.count("feature cache", cfg.batches,
        in_channels, height, width)
      input_values = DCAEVShape.count("convolution input", cfg.batches,
        in_channels, frames, height, width)
      output_values = DCAEVShape.count("convolution output", cfg.batches,
        out_channels, output_total_frames, height, width)
      MetalOps.check_storage(input, input_values.to_i64, WeightDType::BF16)
      MetalOps.check_storage(cache, cache_values.to_i64, WeightDType::BF16)
      MetalOps.check_storage(output, output_values.to_i64, WeightDType::BF16)
      MetalOps.distinct(output, input, cache)

      e = current_encoder(e, chain)
      e.set_pipeline(pipeline("dcae_v_conv3d_bf16"))
      e.set_buffer(input, 0).set_buffer(cache, 1)
      bias_buffer = use_bias ? weights[bias_name] : weights[weight_name]
      e.set_buffer(weights[weight_name], 2).set_buffer(bias_buffer, 3)
      e.set_buffer(output, 4, ML::Metal::BufferAccess::Write)
      e.set_value(cfg.batches.to_u32, 5).set_value(in_channels.to_u32, 6)
      e.set_value(out_channels.to_u32, 7).set_value(frames.to_u32, 8)
      e.set_value(frames.to_u32, 9).set_value(height.to_u32, 10)
      e.set_value(width.to_u32, 11).set_value(input_frame_offset.to_u32, 12)
      e.set_value(chunk_causal && workspace.cache_valid?(cache_name) ? 1_u32 : 0_u32, 13)
      e.set_value(use_silu ? 1_u32 : 0_u32, 14)
      e.set_value(output_total_frames.to_u32, 15).set_value(output_frame_offset.to_u32, 16)
      e.set_value(use_bias ? 1_u32 : 0_u32, 17)
      output_tile_values = DCAEVShape.count("convolution tile", cfg.batches,
        out_channels, frames, height, width)
      e.dispatch_1d(output_tile_values)

      if chunk_causal
        encode_cache_update(e, input, cache, cfg, in_channels, frames, height,
          width, input_frame_offset, cache_name, workspace, chain)
      else
        e
      end
    end

    private def self.encode_conv_unshuffle(e : ML::Metal::ComputeEncoder,
                                           input : ML::MetalBuffer,
                                           output : ML::MetalBuffer,
                                           weights : Arena,
                                           workspace : DCAEVEncoderWorkspace,
                                           cfg : DCAEVEncoderConfig,
                                           weight_prefix : String,
                                           cache_name : String,
                                           in_channels : Int32,
                                           out_channels : Int32,
                                           frames : Int32,
                                           height : Int32,
                                           width : Int32,
                                           spatial_factor : Int32,
                                           temporal_factor : Int32,
                                           image_input : Bool,
                                           chunk_causal : Bool,
                                           input_frame_offset : Int32,
                                           output_total_frames : Int32,
                                           output_frame_offset : Int32,
                                           chain : ML::Metal::MPSCommandChain?) : Nil
      factor = spatial_factor * spatial_factor * temporal_factor
      unless out_channels % factor == 0 &&
             (image_input || (in_channels * factor) % out_channels == 0)
        raise ArgumentError.new("Invalid DCAE-V pixel-unshuffle channel geometry")
      end
      conv_channels = out_channels // factor
      weight_name = "#{weight_prefix}.weight"
      bias_name = "#{weight_prefix}.bias"
      weights.check_weight(weight_name, conv_channels.to_i64 * in_channels * 27)
      weights.check_weight(bias_name, conv_channels.to_i64)
      cache = workspace.cache(cache_name)
      input_count = DCAEVShape.count("downsample input", cfg.batches,
        in_channels, frames, height, width)
      cache_count = DCAEVShape.count("feature cache", cfg.batches,
        in_channels, height, width)
      output_frames = frames // temporal_factor
      output_height = height // spatial_factor
      output_width = width // spatial_factor
      output_count = DCAEVShape.count("downsample output", cfg.batches,
        out_channels, output_total_frames, output_height, output_width)
      MetalOps.check_storage(input, input_count.to_i64, WeightDType::BF16)
      MetalOps.check_storage(cache, cache_count.to_i64, WeightDType::BF16)
      MetalOps.check_storage(output, output_count.to_i64, WeightDType::BF16)
      MetalOps.distinct(output, input, cache)

      e = current_encoder(e, chain)
      e.set_pipeline(pipeline("dcae_v_conv_unshuffle_bf16"))
      e.set_buffer(input, 0).set_buffer(cache, 1)
      e.set_buffer(weights[weight_name], 2).set_buffer(weights[bias_name], 3)
      e.set_buffer(output, 4, ML::Metal::BufferAccess::Write)
      e.set_value(cfg.batches.to_u32, 5).set_value(in_channels.to_u32, 6)
      e.set_value(out_channels.to_u32, 7).set_value(frames.to_u32, 8)
      source_frames = image_input ? cfg.frames : frames
      e.set_value(source_frames.to_u32, 9).set_value(height.to_u32, 10)
      e.set_value(width.to_u32, 11).set_value(input_frame_offset.to_u32, 12)
      e.set_value(chunk_causal && workspace.cache_valid?(cache_name) ? 1_u32 : 0_u32, 13)
      e.set_value(spatial_factor.to_u32, 14).set_value(temporal_factor.to_u32, 15)
      e.set_value(output_frames.to_u32, 16).set_value(output_total_frames.to_u32, 17)
      e.set_value(output_frame_offset.to_u32, 18).set_value(image_input ? 0_u32 : 1_u32, 19)
      tile_count = DCAEVShape.count("downsample tile", cfg.batches,
        out_channels, output_frames, output_height, output_width)
      e.dispatch_1d(tile_count)

      if chunk_causal
        encode_cache_update(e, input, cache, cfg, in_channels, frames, height,
          width, input_frame_offset, cache_name, workspace, chain, image_input)
      end
    end

    private def self.encode_project_out(e : ML::Metal::ComputeEncoder,
                                        input : ML::MetalBuffer,
                                        output : ML::MetalBuffer,
                                        weights : Arena,
                                        workspace : DCAEVEncoderWorkspace,
                                        cfg : DCAEVEncoderConfig,
                                        frames : Int32, height : Int32,
                                        width : Int32, tile : Int32,
                                        chunk_causal : Bool,
                                        chain : ML::Metal::MPSCommandChain?) : Nil
      encode_conv(e, input, output, weights, workspace, cfg,
        PROJECT_OUT_WEIGHT_PREFIX, "encoder.project_out", 1024, 32,
        frames, height, width, 0, cfg.latent_frames,
        tile * cfg.latent_tile_frames, true, false, chunk_causal, chain)
    end

    private def self.encode_cache_update(e : ML::Metal::ComputeEncoder,
                                         input : ML::MetalBuffer,
                                         cache : ML::MetalBuffer,
                                         cfg : DCAEVEncoderConfig,
                                         channels : Int32, frames : Int32,
                                         height : Int32, width : Int32,
                                         frame_offset : Int32, name : String,
                                         workspace : DCAEVEncoderWorkspace,
                                         chain : ML::Metal::MPSCommandChain?,
                                         image_input : Bool = false) : ML::Metal::ComputeEncoder
      e = current_encoder(e, chain)
      source_frames = image_input ? cfg.frames : frames
      count = DCAEVShape.count("feature cache", cfg.batches, channels, height, width)
      e.set_pipeline(pipeline("dcae_v_cache_last_frame_bf16"))
      e.set_buffer(input, 0).set_buffer(cache, 1, ML::Metal::BufferAccess::Write)
      e.set_value(channels.to_u32, 2).set_value(frames.to_u32, 3)
      e.set_value(source_frames.to_u32, 4).set_value(frame_offset.to_u32, 5)
      e.set_value(height.to_u32, 6).set_value(width.to_u32, 7)
      e.set_value(cfg.batches.to_u32, 8)
      e.dispatch_1d(count)
      workspace.mark_cache_valid!(name)
      e
    end

    private def self.capture_trace(e : ML::Metal::ComputeEncoder,
                                   trace : DCAEVEncoderTrace?, name : String,
                                   source : ML::MetalBuffer,
                                   cfg : DCAEVEncoderConfig,
                                   channels : Int32, frames : Int32,
                                   height : Int32, width : Int32,
                                   chain : ML::Metal::MPSCommandChain?) : Nil
      return unless active = trace
      return unless target = active.buffer(name)
      count = DCAEVShape.count("stage trace", cfg.batches, channels, frames, height, width)
      e = current_encoder(e, chain)
      e.set_pipeline(pipeline("dcae_v_copy_bf16"))
      e.set_buffer(source, 0).set_buffer(target, 1, ML::Metal::BufferAccess::Write)
      e.set_value(count.to_u32, 2)
      e.dispatch_1d(count)
    end

    private def self.validate_workspace!(workspace : DCAEVEncoderWorkspace,
                                         cfg : DCAEVEncoderConfig) : Nil
      unless workspace.config.batches == cfg.batches && workspace.config.height == cfg.height &&
             workspace.config.width == cfg.width && workspace.config.frames == cfg.frames &&
             workspace.config.tile_frames == cfg.tile_frames
        raise ArgumentError.new("DCAE-V workspace/config mismatch")
      end
      unless workspace.scratch_values == scratch_values(cfg)
        raise ArgumentError.new("DCAE-V scratch plan does not match its config")
      end
      3.times do |index|
        MetalOps.check_storage(workspace.scratch(index), workspace.scratch_values.to_i64,
          WeightDType::BF16)
      end
      specs = cache_specs(cfg)
      unless workspace.cache_names == specs.map(&.name)
        raise ArgumentError.new("DCAE-V feature cache set changed")
      end
      specs.each do |spec|
        count = DCAEVShape.count("feature cache", cfg.batches,
          spec.channels, spec.height, spec.width)
        MetalOps.check_storage(workspace.cache(spec.name), count.to_i64, WeightDType::BF16)
      end
      expected_norm_specs = norm_specs(cfg)
      unless workspace.norm_values == norm_values(cfg) &&
             workspace.norm_scratch_names == expected_norm_specs.map(&.name)
        raise ArgumentError.new("DCAE-V RMSNorm scratch set changed")
      end
      expected_norm_specs.each do |spec|
        {workspace.norm_mean_a(spec.name), workspace.norm_mean_b(spec.name)}.each do |buffer|
          MetalOps.check_storage(buffer, spec.values.to_i64, WeightDType::BF16)
        end
      end
    end

    private def self.validate_weights!(weights : Arena, cfg : DCAEVEncoderConfig) : Nil
      checkpoint_bindings(cfg).each do |binding|
        count = binding.shape.product
        weights.check_weight(binding.target, count)
        unless weights.dtype(binding.target).bf16?
          raise ArgumentError.new("DCAE-V requires original BF16 weight residency")
        end
      end
    end

    private def self.validate_residency!(weights : Arena,
                                         workspace : DCAEVEncoderWorkspace) : Nil
      residents = [] of ML::MetalBuffer
      workspace.arena.buffers.each_value do |buffer|
        if residents.any? { |other| MetalOps.overlap?(other, buffer) }
          raise ArgumentError.new("DCAE-V workspace buffers overlap")
        end
        residents << buffer
      end
      weights.buffers.each_value do |weight|
        if residents.any? { |buffer| MetalOps.overlap?(weight, buffer) }
          raise ArgumentError.new("DCAE-V weights overlap workspace residency")
        end
      end
    end

    private def self.validate_external_buffer!(buffer : ML::MetalBuffer,
                                               weights : Arena,
                                               workspace : DCAEVEncoderWorkspace) : Nil
      workspace.arena.buffers.each_value do |resident|
        if MetalOps.overlap?(buffer, resident)
          raise ArgumentError.new("DCAE-V input/output aliases workspace residency")
        end
      end
      weights.buffers.each_value do |resident|
        if MetalOps.overlap?(buffer, resident)
          raise ArgumentError.new("DCAE-V input/output aliases checkpoint weights")
        end
      end
    end

    private def self.validate_trace!(trace : DCAEVEncoderTrace?, cfg : DCAEVEncoderConfig,
                                     input : ML::MetalBuffer, latent : ML::MetalBuffer,
                                     weights : Arena,
                                     workspace : DCAEVEncoderWorkspace) : Nil
      return unless active = trace
      stage5_tiles = {} of String => Int32
      project_in_tiles = {} of String => Int32
      cfg.tile_count.times do |tile|
        stage5_tiles["trace.stages.5.tile#{tile}"] = tile
        project_in_tiles["trace.project_in.tile#{tile}"] = tile
      end
      targets = [] of ML::MetalBuffer
      active.buffers.each do |name, target|
        if tile = stage5_tiles[name]?
          count = DCAEVShape.count("stage trace", cfg.batches, 1024,
            cfg.tile_frame_count(tile) // 4, cfg.latent_height, cfg.latent_width)
        elsif tile = project_in_tiles[name]?
          count = DCAEVShape.count("project-in trace", cfg.batches, 256,
            cfg.tile_frame_count(tile), cfg.height // 2, cfg.width // 2)
        else
          raise ArgumentError.new("Unknown DCAE-V trace target #{name}")
        end
        MetalOps.check_storage(target, count.to_i64, WeightDType::BF16)
        MetalOps.distinct(target, input, latent)
        validate_external_buffer!(target, weights, workspace)
        if targets.any? { |other| MetalOps.overlap?(other, target) }
          raise ArgumentError.new("DCAE-V trace buffers overlap")
        end
        targets << target
      end
    end

    private def self.validate_byte_buffer!(buffer : ML::MetalBuffer,
                                           bytes : Int64, label : String) : Nil
      unless buffer.valid? && bytes > 0 && buffer.size == bytes
        raise ArgumentError.new("Invalid #{label} byte buffer")
      end
    end

    private def self.current_encoder(e : ML::Metal::ComputeEncoder,
                                     chain : ML::Metal::MPSCommandChain?) : ML::Metal::ComputeEncoder
      chain ? chain.not_nil!.encoder : e
    end

    private def self.continue_execution!(e : ML::Metal::ComputeEncoder,
                                         chain : ML::Metal::MPSCommandChain?) : ML::Metal::ComputeEncoder
      return e unless active = chain
      active.pause!
      active.continue_async!
      active.resume!
      active.encoder
    end
  end
end
