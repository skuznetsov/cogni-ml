require "./metal_ops"
require "./wan_t5"

module ML::Video::MetalLinearMPS
  SOURCE = {{ read_file("#{__DIR__}/kernels/linear_mps.metal") }}

  def self.prepare! : Nil
    WanT5.pipeline("wan_t5_expand_fc2")
    pipeline
  end

  def self.pipeline : ML::Metal::ComputePipeline
    ML::Metal::PipelineCache.get("video_linear_mps_bias_f32") do
      ML::Metal::ComputePipeline.new("video_linear_mps_bias_f32", SOURCE)
    end
  end

  # Preflight separately so a producer can reject the whole append before it
  # writes the shared stage. Plan bindings are identities, not just capacities.
  def self.validate_preexpanded!(input : ML::MetalBuffer, bias : ML::MetalBuffer,
                                 output : ML::MetalBuffer, rows : Int32,
                                 input_dim : Int32, output_dim : Int32,
                                 bias_dtype : WeightDType, plan : WanT5MPSLinearPlan,
                                 weight_stage : ML::MetalBuffer) : Nil
    MetalOps.dims(rows, input_dim, output_dim)
    unless rows.to_i64 == plan.batches.to_i64 * plan.tokens &&
           input_dim == plan.input_dim && output_dim == plan.output_dim
      raise ArgumentError.new("Unsupported preexpanded MPS projection dispatch shape")
    end
    MetalOps.check(input, rows.to_i64 * input_dim)
    MetalOps.check(output, rows.to_i64 * output_dim)
    MetalOps.check(weight_stage, input_dim.to_i64 * output_dim)
    MetalOps.check_storage(bias, output_dim.to_i64, bias_dtype)
    MetalOps.distinct(output, input, bias)
    MetalOps.distinct(weight_stage, input, bias, output)
    plan.validate_buffers!(input, weight_stage, output)
  end

  # Consume a caller-produced, row-major F32 matrix without expanding it again.
  # Prepare pipelines/plans before encoding. The producer and all consumers of
  # this stage must append serially to the same caller-owned command chain.
  # Keep every resource alive through the caller's terminal completion wait.
  def self.project_preexpanded(e : ML::Metal::ComputeEncoder,
                               input : ML::MetalBuffer, bias : ML::MetalBuffer,
                               output : ML::MetalBuffer, rows : Int32,
                               input_dim : Int32, output_dim : Int32,
                               bias_dtype : WeightDType, plan : WanT5MPSLinearPlan,
                               weight_stage : ML::MetalBuffer,
                               apply_bias : Bool = true) : Nil
    validate_preexpanded!(input, bias, output, rows, input_dim, output_dim,
      bias_dtype, plan, weight_stage)
    {% if flag?(:darwin) && !flag?(:cpu_only) %}
      e.encode_wan_t5_mps_fc2(plan.handle)
    {% else %}
      raise ArgumentError.new("MPS projections require macOS Metal Performance Shaders")
    {% end %}
    return unless apply_bias

    count = rows.to_i64 * output_dim
    e.set_pipeline(pipeline)
    e.set_buffer(output, 0, ML::Metal::BufferAccess::ReadWrite)
    e.set_buffer(bias, 1).set_buffer(bias, 2)
    e.set_value(output_dim.to_u32, 3)
    e.set_value(bias_dtype.bf16? ? 1_u32 : 0_u32, 4)
    e.set_value(count.to_u32, 5)
    e.dispatch_1d(count.to_i32)
  end

  # Runs the prepared, rank-three MPSNDArray projection, then adds the resident
  # source-dtype bias on the same open encoder. The plan, stage, inputs, weights,
  # bias, and output must remain alive through the caller's terminal wait.
  def self.project(e : ML::Metal::ComputeEncoder,
                   input : ML::MetalBuffer, weight : ML::MetalBuffer,
                   bias : ML::MetalBuffer, output : ML::MetalBuffer,
                   rows : Int32, input_dim : Int32, output_dim : Int32,
                   weight_dtype : WeightDType, bias_dtype : WeightDType,
                   plan : WanT5MPSLinearPlan,
                   weight_stage : ML::MetalBuffer,
                   apply_bias : Bool = true) : Nil
    MetalOps.dims(rows, input_dim, output_dim)
    MetalOps.check(input, rows.to_i64 * input_dim)
    MetalOps.check_storage(weight, input_dim.to_i64 * output_dim, weight_dtype)
    MetalOps.check_storage(bias, output_dim.to_i64, bias_dtype)
    MetalOps.check(output, rows.to_i64 * output_dim)
    MetalOps.distinct(output, input, weight, bias)
    MetalOps.distinct(weight_stage, bias)

    WanT5.mps_projection(e, input, weight, output, rows, input_dim, output_dim,
      weight_dtype, plan, weight_stage)
    return unless apply_bias

    count = rows.to_i64 * output_dim
    raise ArgumentError.new("Wan MPS bias dispatch exceeds Int32") if count > Int32::MAX
    e.set_pipeline(pipeline)
    e.set_buffer(output, 0, ML::Metal::BufferAccess::ReadWrite)
    # The kernel selects one typed pointer. Bind the same source buffer twice
    # so setup and encode need no dtype-specific temporary.
    e.set_buffer(bias, 1).set_buffer(bias, 2)
    e.set_value(output_dim.to_u32, 3)
    e.set_value(bias_dtype.bf16? ? 1_u32 : 0_u32, 4)
    e.set_value(count.to_u32, 5)
    e.dispatch_1d(count.to_i32)
  end
end

module ML::Video
  # Owns one reusable F32 expansion buffer shared by every fixed MPS plan in a
  # Wan S2V request. For normal/drop-motion comparison runs, construct one stage
  # at the larger of both configurations' requirements and pass it to both
  # workspaces. Release only after every workspace is released after completion.
  class WanS2VMPSLinearStage
    getter weight_count : Int32
    @arena : Arena
    @workspace_count = 0
    @released = false

    def initialize(@weight_count : Int32)
      unless @weight_count > 0
        raise ArgumentError.new("Wan S2V MPS weight stage must be positive")
      end
      @arena = Arena.new
      @arena.alloc("wan_s2v_mps_weight_f32", @weight_count)
      MetalLinearMPS.prepare!
    end

    def self.required_weight_count(dim : Int32, ffn_dim : Int32) : Int32
      unless dim > 0 && ffn_dim > 0
        raise ArgumentError.new("Wan S2V MPS dimensions must be positive")
      end
      dim = dim.to_i64
      ffn = ffn_dim.to_i64
      count = Math.max(dim * ffn, 2_i64 * dim * dim)
      unless count > 0 && count <= Int32::MAX
        raise ArgumentError.new("Wan S2V MPS weight stage exceeds Int32 storage")
      end
      count.to_i32
    end

    def weight_buffer : ML::MetalBuffer
      raise ArgumentError.new("Wan S2V MPS weight stage has been released") if @released
      @arena["wan_s2v_mps_weight_f32"]
    end

    def resident_bytes : Int64
      weight_count.to_i64 * sizeof(Float32)
    end

    # Workspace clients retain the stage until their plans are released.
    def retain_workspace! : Nil
      raise ArgumentError.new("Wan S2V MPS weight stage has been released") if @released
      @workspace_count += 1
    end

    def release_workspace! : Nil
      raise ArgumentError.new("Wan S2V MPS workspace count underflow") unless @workspace_count > 0
      @workspace_count -= 1
    end

    # Explicit lifecycle boundary: call only after the terminal command wait
    # and after all workspaces using this stage have been released.
    def release_after_terminal_wait! : Nil
      return if @released
      unless @workspace_count == 0
        raise ArgumentError.new("Release Wan S2V MPS workspaces before their shared stage")
      end
      @arena.release
      @released = true
    end
  end

  # Prepared fixed-buffer MPS plans for the 10 block and 5 audio projections.
  # All plans share one F32 weight stage; each encode replaces that stage on the
  # same command encoder immediately before consuming it. Scratch arenas and
  # external context/audio buffers must keep their identities until terminal
  # completion. Calls that share a stage must be encoded serially on the same
  # caller-owned command chain. This is opt-in and has no effect on the scalar
  # default path.
  class WanS2VMPSLinearWorkspace
    @stage : WanS2VMPSLinearStage
    @owns_stage : Bool
    @plans = {} of String => WanT5MPSLinearPlan
    @released = false

    def initialize(block_cfg, audio_cfg,
                   block_scratch : Arena, audio_scratch : Arena,
                   context : ML::MetalBuffer, local_audio : ML::MetalBuffer,
                   stage : WanS2VMPSLinearStage? = nil)
      unless block_cfg.batches == audio_cfg.batches &&
             block_cfg.seq == audio_cfg.seq && block_cfg.dim == audio_cfg.dim &&
             block_cfg.heads == audio_cfg.heads &&
             block_cfg.video_frames == audio_cfg.video_frames &&
             block_cfg.spatial_tokens == audio_cfg.spatial_tokens &&
             block_cfg.split == block_cfg.video_tokens
        raise ArgumentError.new("Wan S2V MPS workspace block/audio geometry mismatch")
      end
      unless block_cfg.dim > 0 && block_cfg.ffn_dim > 0
        raise ArgumentError.new("Wan S2V MPS workspace dimensions must be positive")
      end
      @owns_stage = stage.nil?
      required = WanS2VMPSLinearStage.required_weight_count(block_cfg.dim, block_cfg.ffn_dim)
      @stage = stage || WanS2VMPSLinearStage.new(required)
      if @stage.weight_count < required
        raise ArgumentError.new("Wan S2V shared MPS stage is too small for this configuration")
      end
      @stage.retain_workspace!
      begin
        prepare_plans(block_cfg, audio_cfg, block_scratch, audio_scratch, context, local_audio)
      rescue ex
        @plans.each_value(&.release)
        @plans.clear
        @stage.release_workspace!
        @stage.release_after_terminal_wait! if @owns_stage
        @released = true
        raise ex
      end
    end

    def resident_bytes : Int64
      # Explicit owned F32 expansion storage. MPSNDArray plan storage is opaque
      # to Arena accounting and is not included here.
      @stage.resident_bytes
    end

    def plan_count : Int32
      @plans.size.to_i32
    end

    def has_plan?(name : String) : Bool
      !@released && @plans.has_key?(name)
    end

    def validate_kv_preparation_destinations!(prepared_k : ML::MetalBuffer,
                                              prepared_v : ML::MetalBuffer) : Nil
      raise ArgumentError.new("Wan S2V MPS workspace has been released") if @released
      stage_buffer = @stage.weight_buffer
      if MetalOps.overlap?(stage_buffer, prepared_k) || MetalOps.overlap?(stage_buffer, prepared_v)
        raise ArgumentError.new("Prepared K/V buffers must be disjoint from the MPS weight stage")
      end
    end

    def project(e : ML::Metal::ComputeEncoder, name : String,
                input : ML::MetalBuffer, weight : ML::MetalBuffer,
                bias : ML::MetalBuffer, output : ML::MetalBuffer,
                rows : Int32, input_dim : Int32, output_dim : Int32,
                weight_dtype : WeightDType, bias_dtype : WeightDType,
                apply_bias : Bool = true) : Nil
      raise ArgumentError.new("Wan S2V MPS workspace has been released") if @released
      plan = @plans[name]?
      raise ArgumentError.new("Unknown Wan S2V MPS projection plan #{name}") unless plan
      MetalLinearMPS.project(e, input, weight, bias, output, rows, input_dim,
        output_dim, weight_dtype, bias_dtype, plan, @stage.weight_buffer,
        apply_bias: apply_bias)
    end

    # Release plans after the caller's terminal wait. A shared stage remains
    # live until its owner releases it after all workspaces have exited.
    def release_after_terminal_wait! : Nil
      return if @released
      @plans.each_value(&.release)
      @plans.clear
      @stage.release_workspace!
      @released = true
      @stage.release_after_terminal_wait! if @owns_stage
    end

    private def prepare_plans(block_cfg, audio_cfg,
                              block_scratch : Arena, audio_scratch : Arena,
                              context : ML::MetalBuffer,
                              local_audio : ML::MetalBuffer) : Nil
      dim = block_cfg.dim
      rows = block_cfg.rows
      ctx_rows = block_cfg.context_rows

      add_plan("block.self_q", block_scratch["wan_b_norm1"], block_scratch["wan_b_self_q"], rows, dim, dim)
      add_plan("block.self_k", block_scratch["wan_b_norm1"], block_scratch["wan_b_self_k"], rows, dim, dim)
      add_plan("block.self_v", block_scratch["wan_b_norm1"], block_scratch["wan_b_self_v"], rows, dim, dim)
      add_plan("block.self_o", block_scratch["wan_b_self_attn"], block_scratch["wan_b_self_proj"], rows, dim, dim)
      add_plan("block.cross_q", block_scratch["wan_b_cross_norm"], block_scratch["wan_b_self_q"], rows, dim, dim)
      add_plan("block.cross_k", context, block_scratch["wan_b_cross_k"], ctx_rows, dim, dim)
      add_plan("block.cross_v", context, block_scratch["wan_b_cross_v"], ctx_rows, dim, dim)
      add_plan("block.cross_o", block_scratch["wan_b_self_attn"], block_scratch["wan_b_self_proj"], rows, dim, dim)
      add_plan("block.ffn_in", block_scratch["wan_b_cross_norm"], block_scratch["wan_b_ffn_in"], rows, dim, block_cfg.ffn_dim)
      add_plan("block.ffn_out", block_scratch["wan_b_ffn_in"], block_scratch["wan_b_ffn_out"], rows, block_cfg.ffn_dim, dim)

      add_plan("audio.adain", audio_scratch["wan_a_global_silu"], audio_scratch["wan_a_adain_proj"], audio_cfg.frame_rows, dim, 2 * dim)
      add_plan("audio.q", audio_scratch["wan_a_gather"], audio_scratch["wan_a_q"], audio_cfg.gathered_rows, dim, dim)
      add_plan("audio.k", local_audio, audio_scratch["wan_a_k"], audio_cfg.audio_rows, dim, dim)
      add_plan("audio.v", local_audio, audio_scratch["wan_a_v"], audio_cfg.audio_rows, dim, dim)
      add_plan("audio.o", audio_scratch["wan_a_attn"], audio_scratch["wan_a_proj"], audio_cfg.gathered_rows, dim, dim)

      {block_scratch, audio_scratch}.each do |arena|
        arena.buffers.each_value { |buffer| MetalOps.distinct(@stage.weight_buffer, buffer) }
      end
      MetalOps.distinct(@stage.weight_buffer, context, local_audio)
      raise "Wan S2V MPS plan set is incomplete" unless @plans.size == 15
    end

    private def add_plan(name : String, input : ML::MetalBuffer,
                         output : ML::MetalBuffer, rows : Int32,
                         input_dim : Int32, output_dim : Int32) : Nil
      raise ArgumentError.new("Duplicate Wan S2V MPS plan #{name}") if @plans.has_key?(name)
      count = rows.to_i64 * input_dim
      output_count = rows.to_i64 * output_dim
      weight_count = input_dim.to_i64 * output_dim
      unless rows > 0 && count <= Int32::MAX && output_count <= Int32::MAX &&
             weight_count <= Int32::MAX
        raise ArgumentError.new("Wan S2V MPS plan #{name} exceeds Int32 storage")
      end
      plan_cfg = WanT5Config.new(vocab: 1, dim: input_dim, ffn_dim: output_dim,
        heads: input_dim, layers: 1, batches: 1, tokens: rows)
      plan = WanT5MPSLinearPlan.new(plan_cfg, input_dim, output_dim,
        input, @stage.weight_buffer, output)
      @plans[name] = plan
    end
  end
end
