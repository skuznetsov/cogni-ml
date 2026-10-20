# Copyright 2024-2025 The Alibaba Wan Team Authors.
# Apache-2.0. Native F32 encoder adapted from Wan2.2's UMT5 equations.
require "./checkpoint"
require "./metal_mps_command_chain"

{% if flag?(:darwin) && !flag?(:cpu_only) %}
  @[Link(ldflags: "-framework Metal -framework Foundation -framework MetalPerformanceShaders")]
  lib WanT5MPSFC2FFI
    fun plan_create = gs_wan_t5_mps_fc2_plan_create(device : Pointer(Void), batches : Int32,
                                                    tokens : Int32, din : Int32, dout : Int32, input : Pointer(Void),
                                                    weight_f32 : Pointer(Void), output : Pointer(Void)) : Pointer(Void)
    fun plan_release = gs_wan_t5_mps_fc2_plan_release(plan : Pointer(Void)) : Void
  end

  @[Link(ldflags: "-framework Metal -framework Foundation -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph")]
  lib WanT5RMSMPSFFI
    fun plan_create = gs_wan_t5_rms_mps_plan_create(device : Pointer(Void), batches : Int32,
                                                    tokens : Int32, dim : Int32,
                                                    square : Pointer(Void), mean : Pointer(Void)) : Pointer(Void)
    fun encode = gs_wan_t5_rms_mps_encode(plan : Pointer(Void), command : Pointer(Void)) : Int32
    fun plan_release = gs_wan_t5_rms_mps_plan_release(plan : Pointer(Void)) : Void
  end

  @[Link(ldflags: "-framework Metal -framework Foundation -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph")]
  lib WanT5MPSAttentionFFI
    fun plan_create = gs_wan_t5_attention_mps_plan_create(device : Pointer(Void), batches : Int32,
                                                          tokens : Int32, heads : Int32, head_dim : Int32,
                                                          q : Pointer(Void), k : Pointer(Void), v : Pointer(Void),
                                                          bias : Pointer(Void), logits : Pointer(Void),
                                                          scores : Pointer(Void), probabilities : Pointer(Void),
                                                          context : Pointer(Void)) : Pointer(Void)
    fun encode = gs_wan_t5_attention_mps_encode(plan : Pointer(Void), command : Pointer(Void)) : Int32
    fun plan_release = gs_wan_t5_attention_mps_plan_release(plan : Pointer(Void)) : Void
  end
{% end %}

module ML::Video
  record WanT5Config, vocab : Int32 = 256384, dim : Int32 = 4096,
    ffn_dim : Int32 = 10240, heads : Int32 = 64, layers : Int32 = 24,
    buckets : Int32 = 32, max_distance : Int32 = 128,
    batches : Int32 = 1, tokens : Int32 = 512, eps : Float32 = 1e-6_f32 do
    def validate! : Nil
      MetalOps.dims(vocab, dim, ffn_dim, heads, layers, buckets, max_distance, batches, tokens)
      unless vocab <= 16777216 && dim % heads == 0 && dim // heads <= 256 && buckets >= 4 && buckets % 4 == 0 &&
             max_distance > buckets // 4 && eps.finite? && eps > 0
        raise ArgumentError.new("Unsupported Wan UMT5 configuration")
      end
      {batches.to_i64 * tokens * dim, batches.to_i64 * tokens * ffn_dim,
       vocab.to_i64 * dim, dim.to_i64 * Math.max(dim, ffn_dim)}.each do |n|
        raise ArgumentError.new("Wan UMT5 shape exceeds addressable storage") unless n <= Int32::MAX
      end
    end

    def rows : Int32
      (batches.to_i64 * tokens).to_i32
    end
  end

  # Fixed-length, shared-batch-shape UMT5 encoder. IDs/masks and all activations
  # are GPU buffers. Original BF16 weights are expanded only at consumption;
  # the admitted oracle computes neural arithmetic in F32, as the video blocks.
  # Setup owns all allocations. The caller owns serial submission/completion.
  # Owns persistent MPSNDArray descriptors, arrays, and matrix kernel for one
  # workspace. Keep it alive, and release it only after queued commands finish.
  class WanT5MPSLinearPlan
    getter batches : Int32
    getter tokens : Int32
    getter input_dim : Int32
    getter output_dim : Int32
    @input_buffer : ML::MetalBuffer
    @weight_buffer : ML::MetalBuffer
    @output_buffer : ML::MetalBuffer
    @handle : Pointer(Void)
    @released : Bool = false

    def initialize(cfg : WanT5Config, input_dim : Int32, output_dim : Int32,
                   input : ML::MetalBuffer, weight_f32 : ML::MetalBuffer,
                   output : ML::MetalBuffer)
      @handle = Pointer(Void).null
      cfg.validate!
      unless cfg.batches > 0 && cfg.tokens > 0 && input_dim > 0 && output_dim > 0
        raise ArgumentError.new("Wan T5 MPS projection dimensions must be positive")
      end
      rows = cfg.batches.to_i64 * cfg.tokens
      input_count = rows * input_dim
      output_count = rows * output_dim
      weight_count = input_dim.to_i64 * output_dim
      unless {rows, input_count, output_count, weight_count}.all? { |n| n > 0 && n <= Int32::MAX }
        raise ArgumentError.new("Wan T5 MPS projection shape exceeds addressable storage")
      end
      MetalOps.check(input, input_count)
      MetalOps.check(weight_f32, weight_count)
      MetalOps.check(output, output_count)
      MetalOps.distinct(weight_f32, input, output)
      MetalOps.distinct(output, input)
      @batches = cfg.batches
      @tokens = cfg.tokens
      @input_dim = input_dim
      @output_dim = output_dim
      @input_buffer = input
      @weight_buffer = weight_f32
      @output_buffer = output
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        device = ML::Metal::Device.instance.device_handle
        @handle = WanT5MPSFC2FFI.plan_create(device, @batches, @tokens, @input_dim, @output_dim,
          input.handle, weight_f32.handle, output.handle)
        raise "Failed to prepare Wan T5 MPS projection plan" if @handle.null?
      {% else %}
        @handle = Pointer(Void).null
        raise ArgumentError.new("Wan T5 MPS projections require macOS Metal Performance Shaders")
      {% end %}
    end

    def handle : Pointer(Void)
      raise "Wan T5 MPS projection plan has been released" if @released || @handle.null?
      @handle
    end

    def validate_shape!(cfg : WanT5Config, input_dim : Int32, output_dim : Int32) : Nil
      unless cfg.batches == @batches && cfg.tokens == @tokens && input_dim == @input_dim && output_dim == @output_dim
        raise ArgumentError.new("Wan T5 MPS projection plan shape does not match encoder")
      end
      handle
    end

    def validate_buffers!(input : ML::MetalBuffer, weight_f32 : ML::MetalBuffer,
                          output : ML::MetalBuffer) : Nil
      unless input.valid? && weight_f32.valid? && output.valid? &&
             input.handle == @input_buffer.handle && weight_f32.handle == @weight_buffer.handle &&
             output.handle == @output_buffer.handle
        raise ArgumentError.new("Wan T5 MPS projection buffers do not match workspace scratch")
      end
      handle
    end

    def release : Nil
      return if @released
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        WanT5MPSFC2FFI.plan_release(@handle) unless @handle.null?
      {% end %}
      @handle = Pointer(Void).null
      @released = true
    end

    def finalize
      release
    end
  end

  # Persistent MPSGraph reduction plan over the resident, separately rounded
  # F32 square buffer. The plan binds exactly one workspace's square and mean
  # buffers and must outlive the caller's MPS command chain.
  class WanT5RMSMeanPlan
    getter batches : Int32
    getter tokens : Int32
    getter dim : Int32
    @square_buffer : ML::MetalBuffer
    @mean_buffer : ML::MetalBuffer
    @handle : Pointer(Void)
    @released : Bool = false

    def initialize(cfg : WanT5Config, square : ML::MetalBuffer, mean : ML::MetalBuffer)
      @handle = Pointer(Void).null
      cfg.validate!
      rows = cfg.rows.to_i64
      elements = rows * cfg.dim
      unless rows > 0 && elements > 0 && elements <= Int32::MAX
        raise ArgumentError.new("Wan T5 RMS mean shape exceeds addressable storage")
      end
      MetalOps.check(square, elements)
      MetalOps.check(mean, rows)
      MetalOps.distinct(square, mean)
      @batches = cfg.batches
      @tokens = cfg.tokens
      @dim = cfg.dim
      @square_buffer = square
      @mean_buffer = mean
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        device = ML::Metal::Device.instance.device_handle
        @handle = WanT5RMSMPSFFI.plan_create(device, @batches, @tokens, @dim,
          square.handle, mean.handle)
        raise "Failed to prepare Wan T5 RMS mean plan" if @handle.null?
      {% else %}
        raise ArgumentError.new("Wan T5 RMS MPSGraph requires macOS Metal Performance Shaders")
      {% end %}
    end

    def validate!(cfg : WanT5Config, square : ML::MetalBuffer, mean : ML::MetalBuffer) : Nil
      unless cfg.batches == @batches && cfg.tokens == @tokens && cfg.dim == @dim &&
             square.valid? && mean.valid? && square.handle == @square_buffer.handle &&
             mean.handle == @mean_buffer.handle
        raise ArgumentError.new("Wan T5 RMS mean plan buffers do not match workspace scratch")
      end
      raise "Wan T5 RMS mean plan has been released" if @released || @handle.null?
      MetalOps.distinct(square, mean)
    end

    def encode(command : Pointer(Void)) : Nil
      raise "Wan T5 RMS mean plan has been released" if @released || @handle.null?
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        status = WanT5RMSMPSFFI.encode(@handle, command)
        raise "Wan T5 RMS MPSGraph encode rejected (#{status})" unless status == 0
      {% else %}
        raise ArgumentError.new("Wan T5 RMS MPSGraph requires macOS Metal Performance Shaders")
      {% end %}
    end

    def release : Nil
      return if @released
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        WanT5RMSMPSFFI.plan_release(@handle) unless @handle.null?
      {% end %}
      @handle = Pointer(Void).null
      @released = true
    end

    def finalize
      release
    end
  end

  # Prepared rank-three MPSGraph attention with materialized F32 checkpoints.
  # The collapsed batch/head products preserve PyTorch einsum's bmm rank.
  # The opt-in path consumes resident Q/K/V and a GPU-created bias tensor.
  class WanT5MPSAttentionPlan
    getter batches : Int32
    getter tokens : Int32
    getter heads : Int32
    getter head_dim : Int32
    @q_buffer : ML::MetalBuffer
    @k_buffer : ML::MetalBuffer
    @v_buffer : ML::MetalBuffer
    @bias_buffer : ML::MetalBuffer
    @logits_buffer : ML::MetalBuffer
    @scores_buffer : ML::MetalBuffer
    @probabilities_buffer : ML::MetalBuffer
    @context_buffer : ML::MetalBuffer
    @handle : Pointer(Void)
    @released : Bool = false

    def initialize(cfg : WanT5Config, q : ML::MetalBuffer, k : ML::MetalBuffer,
                   v : ML::MetalBuffer, bias : ML::MetalBuffer,
                   logits : ML::MetalBuffer, scores : ML::MetalBuffer,
                   probabilities : ML::MetalBuffer, context : ML::MetalBuffer)
      @handle = Pointer(Void).null
      cfg.validate!
      @batches = cfg.batches
      @tokens = cfg.tokens
      @heads = cfg.heads
      @head_dim = cfg.dim // cfg.heads
      q_count = cfg.rows.to_i64 * cfg.dim
      score_count = cfg.batches.to_i64 * cfg.heads * cfg.tokens * cfg.tokens
      unless q_count > 0 && score_count > 0 &&
             q_count <= Int32::MAX && score_count <= Int32::MAX
        raise ArgumentError.new("Wan T5 MPS attention probe shape exceeds addressable storage")
      end
      {q, k, v, context}.each { |buffer| MetalOps.check(buffer, q_count) }
      {bias, logits, scores, probabilities}.each { |buffer| MetalOps.check(buffer, score_count) }
      MetalOps.distinct(q, k, v, bias, logits, scores, probabilities, context)
      @q_buffer = q
      @k_buffer = k
      @v_buffer = v
      @bias_buffer = bias
      @logits_buffer = logits
      @scores_buffer = scores
      @probabilities_buffer = probabilities
      @context_buffer = context
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        device = ML::Metal::Device.instance.device_handle
        @handle = WanT5MPSAttentionFFI.plan_create(device, @batches, @tokens, @heads, @head_dim,
          q.handle, k.handle, v.handle, bias.handle, logits.handle, scores.handle,
          probabilities.handle, context.handle)
        raise "Failed to prepare Wan T5 MPS attention diagnostic" if @handle.null?
      {% else %}
        raise ArgumentError.new("Wan T5 MPS attention diagnostic requires macOS Metal Performance Shaders")
      {% end %}
    end

    def validate_buffers!(q : ML::MetalBuffer, k : ML::MetalBuffer, v : ML::MetalBuffer,
                          bias : ML::MetalBuffer, logits : ML::MetalBuffer,
                          scores : ML::MetalBuffer, probabilities : ML::MetalBuffer,
                          context : ML::MetalBuffer) : Nil
      unless {q, k, v, bias, logits, scores, probabilities, context}.all?(&.valid?) &&
             q.handle == @q_buffer.handle && k.handle == @k_buffer.handle &&
             v.handle == @v_buffer.handle && bias.handle == @bias_buffer.handle &&
             logits.handle == @logits_buffer.handle && scores.handle == @scores_buffer.handle &&
             probabilities.handle == @probabilities_buffer.handle && context.handle == @context_buffer.handle
        raise ArgumentError.new("Wan T5 MPS attention diagnostic buffers do not match its prepared plan")
      end
      raise "Wan T5 MPS attention diagnostic has been released" if @released || @handle.null?
    end

    def validate_shape!(cfg : WanT5Config) : Nil
      unless cfg.batches == @batches && cfg.tokens == @tokens &&
             cfg.heads == @heads && cfg.dim // cfg.heads == @head_dim
        raise ArgumentError.new("Wan T5 MPS attention dimensions do not match its prepared plan")
      end
    end

    def encode(command : Pointer(Void)) : Nil
      raise "Wan T5 MPS attention diagnostic has been released" if @released || @handle.null?
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        status = WanT5MPSAttentionFFI.encode(@handle, command)
        raise "Wan T5 MPS attention graph encode rejected (#{status})" unless status == 0
      {% else %}
        raise ArgumentError.new("Wan T5 MPS attention diagnostic requires macOS Metal Performance Shaders")
      {% end %}
    end

    def release : Nil
      return if @released
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        WanT5MPSAttentionFFI.plan_release(@handle) unless @handle.null?
      {% end %}
      @handle = Pointer(Void).null
      @released = true
    end

    def finalize
      release
    end
  end

  # One serial workspace uses one reusable F32 weight stage. Each plan has its
  # own activation bindings; the stage is rewritten before every projection.
  class WanT5MPSPlans
    getter rms_mean : WanT5RMSMeanPlan
    getter q : WanT5MPSLinearPlan
    getter k : WanT5MPSLinearPlan
    getter v : WanT5MPSLinearPlan
    getter o : WanT5MPSLinearPlan
    getter gate : WanT5MPSLinearPlan
    getter fc1 : WanT5MPSLinearPlan
    getter fc2 : WanT5MPSLinearPlan
    getter attention : WanT5MPSAttentionPlan?
    @released : Bool = false

    def initialize(@rms_mean : WanT5RMSMeanPlan,
                   @q : WanT5MPSLinearPlan, @k : WanT5MPSLinearPlan,
                   @v : WanT5MPSLinearPlan, @o : WanT5MPSLinearPlan,
                   @gate : WanT5MPSLinearPlan, @fc1 : WanT5MPSLinearPlan,
                   @fc2 : WanT5MPSLinearPlan,
                   @attention : WanT5MPSAttentionPlan? = nil)
    end

    def validate!(cfg : WanT5Config, norm : ML::MetalBuffer,
                  q_output : ML::MetalBuffer, k_output : ML::MetalBuffer,
                  v_output : ML::MetalBuffer, attention : ML::MetalBuffer,
                  projection : ML::MetalBuffer, gate_output : ML::MetalBuffer,
                  fc1_output : ML::MetalBuffer, ffn_output : ML::MetalBuffer,
                  weight_stage : ML::MetalBuffer,
                  rms_square : ML::MetalBuffer, rms_mean : ML::MetalBuffer) : Nil
      raise "Wan T5 MPS plan set has been released" if @released
      @rms_mean.validate!(cfg, rms_square, rms_mean)
      @q.validate_shape!(cfg, cfg.dim, cfg.dim)
      @k.validate_shape!(cfg, cfg.dim, cfg.dim)
      @v.validate_shape!(cfg, cfg.dim, cfg.dim)
      @o.validate_shape!(cfg, cfg.dim, cfg.dim)
      @gate.validate_shape!(cfg, cfg.dim, cfg.ffn_dim)
      @fc1.validate_shape!(cfg, cfg.dim, cfg.ffn_dim)
      @fc2.validate_shape!(cfg, cfg.ffn_dim, cfg.dim)
      @q.validate_buffers!(norm, weight_stage, q_output)
      @k.validate_buffers!(norm, weight_stage, k_output)
      @v.validate_buffers!(norm, weight_stage, v_output)
      @o.validate_buffers!(attention, weight_stage, projection)
      @gate.validate_buffers!(norm, weight_stage, gate_output)
      @fc1.validate_buffers!(norm, weight_stage, fc1_output)
      @fc2.validate_buffers!(fc1_output, weight_stage, ffn_output)
      MetalOps.distinct(norm, q_output, k_output, v_output, attention, projection,
        gate_output, fc1_output, ffn_output, weight_stage, rms_square, rms_mean)
      MetalOps.distinct(q_output, k_output, v_output, attention, projection,
        gate_output, fc1_output, ffn_output, rms_square, rms_mean)
    end

    def release : Nil
      return if @released
      @attention.try(&.release)
      @rms_mean.release
      @fc2.release
      @fc1.release
      @gate.release
      @o.release
      @v.release
      @k.release
      @q.release
      @released = true
    end

    def finalize
      release
    end
  end

  module WanT5
    SOURCE = {{ read_file("#{__DIR__}/kernels/wan_t5.metal") }}

    def self.pipeline(name : String) : ML::Metal::ComputePipeline
      ML::Metal::PipelineCache.get(name) { ML::Metal::ComputePipeline.new(name, SOURCE) }
    end

    def self.prepare! : Nil
      {"wan_t5_embed", "wan_t5_attention", "wan_t5_product", "wan_t5_mask_context",
       "wan_t5_expand_fc2", "wan_t5_rms_norm", "wan_t5_rms_norm_contract_off",
       "wan_t5_rms_square", "wan_t5_rms_from_mean", "wan_t5_gelu_source_f32",
       "wan_t5_attention_bias"}.each { |n| pipeline(n) }
    end

    # Native scalar RMS diagnostic retained for same-input topology probes.
    # Production encode uses rms_norm_mps_mean to match PyTorch's MPS reduction.
    def self.rms_norm(e : ML::Metal::ComputeEncoder, x : ML::MetalBuffer,
                      weight : ML::MetalBuffer, y : ML::MetalBuffer,
                      rows : Int32, dim : Int32, eps : Float32,
                      weight_dtype : WeightDType = WeightDType::F32) : Nil
      threads = Math.min(256, ((dim + 31) // 32) * 32)
      rms_norm_dispatch(e, x, weight, y, rows, dim, eps, weight_dtype, threads, false, nil, nil)
    end

    # Production path: native F32 power writes a separately rounded square,
    # then MPSGraph computes the feature mean on the caller's command chain.
    # The refreshed root receives the native reciprocal-sqrt and affine stage.
    def self.rms_norm_mps_mean(chain : ML::Metal::MPSCommandChain,
                               plan : WanT5RMSMeanPlan, cfg : WanT5Config,
                               x : ML::MetalBuffer, weight : ML::MetalBuffer,
                               y : ML::MetalBuffer, square : ML::MetalBuffer,
                               mean : ML::MetalBuffer,
                               weight_dtype : WeightDType = WeightDType::F32) : Nil
      cfg.validate!
      rows = cfg.rows
      dim = cfg.dim
      n = rows.to_i64 * dim
      plan.validate!(cfg, square, mean)
      MetalOps.check(x, n)
      MetalOps.check(square, n)
      MetalOps.check(mean, rows.to_i64)
      MetalOps.check(y, n)
      MetalOps.check_storage(weight, dim.to_i64, weight_dtype)
      MetalOps.inplace_or_distinct(x, y)
      MetalOps.distinct(y, weight)
      MetalOps.distinct(square, mean, x, y, weight)
      MetalOps.distinct(mean, square, x, y, weight)

      encoder = chain.encoder
      encoder.set_pipeline(pipeline("wan_t5_rms_square"))
      encoder.set_buffer(x, 0).set_buffer(square, 1, ML::Metal::BufferAccess::Write)
      encoder.set_value(n.to_u32, 2).dispatch_1d(n.to_i32)

      chain.pause!
      plan.encode(chain.handle)
      chain.resume!
      encoder = chain.encoder
      threads = Math.min(256, ((dim + 31) // 32) * 32)
      encoder.set_pipeline(pipeline("wan_t5_rms_from_mean"))
      encoder.set_buffer(x, 0).set_buffer(weight, 1)
      encoder.set_buffer(y, 2, ML::Metal::BufferAccess::Write).set_buffer(mean, 3)
      encoder.set_value(rows.to_u32, 4).set_value(dim.to_u32, 5).set_value(cfg.eps, 6)
      encoder.set_value(weight_dtype.bf16? ? 1_u32 : 0_u32, 7)
      encoder.set_buffer(weight_dtype.bf16? ? x : weight, 8)
      encoder.dispatch_threadgroups({rows, 1, 1}, {threads, 1, 1})
    end

    # Opt-in diagnostic path matching the pinned Wan UMT5 GELU expression's
    # operation order with separately rounded F32 intermediates. It binds only
    # caller-owned buffers and supports exact in-place or distinct outputs.
    def self.gelu_source(e : ML::Metal::ComputeEncoder, x : ML::MetalBuffer,
                         y : ML::MetalBuffer, count : Int32) : Nil
      MetalOps.dims(count)
      MetalOps.check(x, count.to_i64)
      MetalOps.check(y, count.to_i64)
      MetalOps.inplace_or_distinct(x, y)
      e.set_pipeline(pipeline("wan_t5_gelu_source_f32"))
      e.set_buffer(x, 0).set_buffer(y, 1, ML::Metal::BufferAccess::Write)
      e.set_value(count.to_u32, 2).dispatch_1d(count)
    end

    # Diagnostic scalar256 path. The row-wise outputs expose its reduction
    # result before and after reciprocal-square-root.
    def self.rms_norm_trace(e : ML::Metal::ComputeEncoder, x : ML::MetalBuffer,
                            weight : ML::MetalBuffer, y : ML::MetalBuffer,
                            mean_square : ML::MetalBuffer, inverse_rms : ML::MetalBuffer,
                            rows : Int32, dim : Int32, eps : Float32,
                            weight_dtype : WeightDType = WeightDType::F32) : Nil
      threads = Math.min(256, ((dim + 31) // 32) * 32)
      rms_norm_dispatch(e, x, weight, y, rows, dim, eps, weight_dtype, threads, false,
        mean_square, inverse_rms)
    end

    # Diagnostic-only scalar256 trace. The corresponding Metal kernel disables
    # contraction while preserving the production reduction topology.
    def self.rms_norm_contract_off_trace(e : ML::Metal::ComputeEncoder, x : ML::MetalBuffer,
                                         weight : ML::MetalBuffer, y : ML::MetalBuffer,
                                         mean_square : ML::MetalBuffer, inverse_rms : ML::MetalBuffer,
                                         rows : Int32, dim : Int32, eps : Float32,
                                         weight_dtype : WeightDType = WeightDType::F32) : Nil
      threads = Math.min(256, ((dim + 31) // 32) * 32)
      rms_norm_dispatch(e, x, weight, y, rows, dim, eps, weight_dtype, threads, false,
        mean_square, inverse_rms, "wan_t5_rms_norm_contract_off")
    end

    # Test-only topology discriminator. This deliberately exposes a closed set
    # of reduction shapes so the oracle spec can compare candidate RMS trees on
    # identical input without changing the production dispatch.
    def self.rms_norm_probe(e : ML::Metal::ComputeEncoder, x : ML::MetalBuffer,
                            weight : ML::MetalBuffer, y : ML::MetalBuffer,
                            rows : Int32, dim : Int32, eps : Float32,
                            weight_dtype : WeightDType, threads : Int32,
                            vector4 : Bool) : Nil
      unless {32, 64, 128, 256, 512, 1024}.includes?(threads)
        raise ArgumentError.new("Unsupported T5 RMS probe thread count")
      end
      if vector4 && dim % 4 != 0
        raise ArgumentError.new("T5 vector4 RMS probe requires a dimension divisible by four")
      end
      rms_norm_dispatch(e, x, weight, y, rows, dim, eps, weight_dtype, threads, vector4, nil, nil)
    end

    private def self.rms_norm_dispatch(e : ML::Metal::ComputeEncoder, x : ML::MetalBuffer,
                                       weight : ML::MetalBuffer, y : ML::MetalBuffer,
                                       rows : Int32, dim : Int32, eps : Float32,
                                       weight_dtype : WeightDType, threads : Int32,
                                       vector4 : Bool, mean_square_output : ML::MetalBuffer?,
                                       inverse_rms_output : ML::MetalBuffer?,
                                       pipeline_name : String = "wan_t5_rms_norm") : Nil
      MetalOps.dims(rows, dim)
      raise ArgumentError.new("Invalid T5 norm epsilon") unless eps.finite? && eps > 0
      raise ArgumentError.new("Invalid T5 RMS thread count") unless threads >= 32 && threads <= 1024 && threads % 32 == 0
      if mean_square_output.nil? != inverse_rms_output.nil?
        raise ArgumentError.new("T5 RMS diagnostics require both output buffers")
      end
      MetalOps.check(x, rows.to_i64 * dim)
      MetalOps.check(y, rows.to_i64 * dim)
      MetalOps.check_storage(weight, dim.to_i64, weight_dtype)
      MetalOps.inplace_or_distinct(x, y)
      MetalOps.distinct(y, weight)
      capture_stats = !mean_square_output.nil?
      if capture_stats
        mean = mean_square_output.not_nil!
        inverse = inverse_rms_output.not_nil!
        MetalOps.check(mean, rows.to_i64)
        MetalOps.check(inverse, rows.to_i64)
        MetalOps.distinct(mean, inverse)
        {x, y, weight}.each do |buffer|
          MetalOps.distinct(mean, buffer)
          MetalOps.distinct(inverse, buffer)
        end
      end

      e.set_pipeline(pipeline(pipeline_name))
      e.set_buffer(x, 0).set_buffer(weight, 1)
      e.set_buffer(y, 2, ML::Metal::BufferAccess::Write)
      e.set_value(rows.to_u32, 3).set_value(dim.to_u32, 4).set_value(eps, 5)
      e.set_value(weight_dtype.bf16? ? 1_u32 : 0_u32, 6)
      e.set_buffer(weight_dtype.bf16? ? x : weight, 7)
      e.set_value(vector4 ? 1_u32 : 0_u32, 8)
      if capture_stats
        e.set_buffer(mean_square_output.not_nil!, 9, ML::Metal::BufferAccess::Write)
        e.set_buffer(inverse_rms_output.not_nil!, 10, ML::Metal::BufferAccess::Write)
      else
        e.set_buffer(y, 9).set_buffer(y, 10)
      end
      e.set_value(capture_stats ? 1_u32 : 0_u32, 11)
      e.dispatch_threadgroups({rows, 1, 1}, {threads, 1, 1})
    end

    def self.mps_projection(e : ML::Metal::ComputeEncoder, input : ML::MetalBuffer,
                            weights : ML::MetalBuffer, output : ML::MetalBuffer,
                            rows : Int32, din : Int32, dout : Int32, dtype : WeightDType,
                            plan : WanT5MPSLinearPlan, weight_stage : ML::MetalBuffer) : Nil
      MetalOps.dims(rows, din, dout)
      unless rows == plan.batches * plan.tokens &&
             din == plan.input_dim && dout == plan.output_dim
        raise ArgumentError.new("Unsupported Wan T5 MPS projection dispatch shape")
      end
      MetalOps.check(input, rows.to_i64 * din)
      MetalOps.check(output, rows.to_i64 * dout)
      MetalOps.check_storage(weights, din.to_i64 * dout, dtype)
      MetalOps.check(weight_stage, din.to_i64 * dout)
      MetalOps.distinct(output, input, weights)
      MetalOps.distinct(weight_stage, input, weights, output)
      plan.validate_buffers!(input, weight_stage, output)
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        e.set_pipeline(pipeline("wan_t5_expand_fc2"))
        e.set_buffer(weights, 0).set_buffer(weights, 1)
        e.set_buffer(weight_stage, 2, ML::Metal::BufferAccess::Write)
        e.set_value((din * dout).to_u32, 3).set_value(dtype.bf16? ? 1_u32 : 0_u32, 4)
        e.dispatch_1d(din * dout)
        e.encode_wan_t5_mps_fc2(plan.handle)
      {% else %}
        raise ArgumentError.new("Wan T5 MPS projections require macOS Metal Performance Shaders")
      {% end %}
    end

    def self.down_projection(e : ML::Metal::ComputeEncoder, input : ML::MetalBuffer,
                             weights : ML::MetalBuffer, output : ML::MetalBuffer,
                             rows : Int32, din : Int32, dout : Int32, dtype : WeightDType,
                             plan : WanT5MPSLinearPlan, weight_stage : ML::MetalBuffer) : Nil
      mps_projection(e, input, weights, output, rows, din, dout, dtype, plan, weight_stage)
    end

    def self.attention_kernel(e : ML::Metal::ComputeEncoder, q : ML::MetalBuffer,
                              k : ML::MetalBuffer, v : ML::MetalBuffer,
                              bias : ML::MetalBuffer, mask : ML::MetalBuffer,
                              output : ML::MetalBuffer, cfg : WanT5Config,
                              bias_dtype : WeightDType) : Nil
      cfg.validate!
      n = cfg.rows.to_i64 * cfg.dim
      {q, k, v, output}.each { |buffer| MetalOps.check(buffer, n) }
      MetalOps.check_storage(bias, cfg.buckets.to_i64 * cfg.heads, bias_dtype)
      MetalOps.check(mask, cfg.rows.to_i64)
      MetalOps.distinct(output, q, k, v, bias, mask)
      e.set_pipeline(pipeline("wan_t5_attention"))
      e.set_buffer(q, 0).set_buffer(k, 1).set_buffer(v, 2)
      e.set_buffer(bias, 3).set_buffer(mask, 4).set_buffer(output, 5, ML::Metal::BufferAccess::Write)
      e.set_value(cfg.tokens.to_u32, 6).set_value(cfg.heads.to_u32, 7).set_value((cfg.dim // cfg.heads).to_u32, 8)
      e.set_value(cfg.buckets.to_u32, 9).set_value(cfg.max_distance.to_u32, 10)
      e.set_value(bias_dtype.bf16? ? 1_u32 : 0_u32, 11).set_buffer(bias, 12)
      threads = ((cfg.dim // cfg.heads + 31) // 32) * 32
      e.dispatch_threadgroups({cfg.rows * cfg.heads, 1, 1}, {threads, 1, 1})
    end

    def self.checkpoint_bindings(cfg : WanT5Config, prefix : String = "t5.") : Array(WeightBinding)
      cfg.validate!
      result = [] of WeightBinding
      add = ->(name : String, shape : Array(Int64)) { result << WeightBinding.new(prefix + name, name, shape) }
      add.call("token_embedding.weight", [cfg.vocab.to_i64, cfg.dim.to_i64])
      cfg.layers.times do |layer|
        p = "blocks.#{layer}."
        {"norm1.weight", "norm2.weight"}.each { |n| add.call(p + n, [cfg.dim.to_i64]) }
        {"q", "k", "v", "o"}.each { |n| add.call(p + "attn.#{n}.weight", [cfg.dim.to_i64, cfg.dim.to_i64]) }
        {"gate.0", "fc1"}.each { |n| add.call(p + "ffn.#{n}.weight", [cfg.ffn_dim.to_i64, cfg.dim.to_i64]) }
        add.call(p + "ffn.fc2.weight", [cfg.dim.to_i64, cfg.ffn_dim.to_i64])
        add.call(p + "pos_embedding.embedding.weight", [cfg.buckets.to_i64, cfg.heads.to_i64])
      end
      add.call("norm.weight", [cfg.dim.to_i64])
      result
    end

    def self.attention_score_count(cfg : WanT5Config) : Int32
      count = cfg.batches.to_i64 * cfg.heads * cfg.tokens * cfg.tokens
      raise ArgumentError.new("Wan T5 attention scores exceed addressable storage") unless count > 0 && count <= Int32::MAX
      count.to_i32
    end

    def self.scratch_bytes(cfg : WanT5Config, graph_attention : Bool = false) : Int64
      cfg.validate!
      (9_i64 * cfg.rows * cfg.dim + cfg.rows + 2_i64 * cfg.rows * cfg.ffn_dim +
        Math.max(cfg.dim, cfg.ffn_dim) + cfg.dim.to_i64 * Math.max(cfg.dim, cfg.ffn_dim) +
        (graph_attention ? 4_i64 * attention_score_count(cfg) : 0_i64)) * 4
    end

    def self.prepare_scratch!(a : Arena, cfg : WanT5Config, prefix : String = "t5.s.",
                              graph_attention : Bool = false) : Nil
      cfg.validate!
      prepare!
      {"x", "norm", "q", "k", "v", "attention", "projection", "ffn_out"}.each do |n|
        a.alloc(prefix + n, cfg.rows * cfg.dim)
      end
      a.alloc(prefix + "rms_square", cfg.rows * cfg.dim)
      a.alloc(prefix + "rms_mean", cfg.rows)
      {"gate", "fc1"}.each { |n| a.alloc(prefix + n, cfg.rows * cfg.ffn_dim) }
      a.alloc(prefix + "fc2_weight_f32", (cfg.dim.to_i64 * Math.max(cfg.dim, cfg.ffn_dim)).to_i32)
      if graph_attention
        {"attention_bias", "attention_logits", "attention_scores", "attention_probs"}.each do |name|
          a.alloc(prefix + name, attention_score_count(cfg))
        end
      end
      a.upload(prefix + "zero", Array(Float32).new(Math.max(cfg.dim, cfg.ffn_dim), 0_f32))
    end

    def self.prepare_mps_plans(cfg : WanT5Config, a : Arena,
                               scratch : String = "t5.s.", graph_attention : Bool = false) : WanT5MPSPlans
      cfg.validate!
      input = a[scratch + "norm"]
      weight_stage = a[scratch + "fc2_weight_f32"]
      created = [] of WanT5MPSLinearPlan
      rms_mean = nil.as(WanT5RMSMeanPlan?)
      attention = nil.as(WanT5MPSAttentionPlan?)
      begin
        rms_mean = WanT5RMSMeanPlan.new(cfg, a[scratch + "rms_square"], a[scratch + "rms_mean"])
        q = WanT5MPSLinearPlan.new(cfg, cfg.dim, cfg.dim, input, weight_stage, a[scratch + "q"])
        created << q
        k = WanT5MPSLinearPlan.new(cfg, cfg.dim, cfg.dim, input, weight_stage, a[scratch + "k"])
        created << k
        v = WanT5MPSLinearPlan.new(cfg, cfg.dim, cfg.dim, input, weight_stage, a[scratch + "v"])
        created << v
        o = WanT5MPSLinearPlan.new(cfg, cfg.dim, cfg.dim, a[scratch + "attention"],
          weight_stage, a[scratch + "projection"])
        created << o
        gate = WanT5MPSLinearPlan.new(cfg, cfg.dim, cfg.ffn_dim, input,
          weight_stage, a[scratch + "gate"])
        created << gate
        fc1 = WanT5MPSLinearPlan.new(cfg, cfg.dim, cfg.ffn_dim, input,
          weight_stage, a[scratch + "fc1"])
        created << fc1
        fc2 = WanT5MPSLinearPlan.new(cfg, cfg.ffn_dim, cfg.dim,
          a[scratch + "fc1"], weight_stage, a[scratch + "ffn_out"])
        created << fc2
        if graph_attention
          attention = WanT5MPSAttentionPlan.new(cfg, a[scratch + "q"], a[scratch + "k"],
            a[scratch + "v"], a[scratch + "attention_bias"], a[scratch + "attention_logits"],
            a[scratch + "attention_scores"], a[scratch + "attention_probs"], a[scratch + "attention"])
        end
        WanT5MPSPlans.new(rms_mean.not_nil!, q, k, v, o, gate, fc1, fc2, attention)
      rescue ex
        attention.try(&.release)
        rms_mean.try(&.release)
        created.each(&.release)
        raise ex
      end
    end

    # IDs and binary masks are F32 setup data with exact integer representation.
    # Validate token ranges on host before upload; encode never reads activations.
    def self.upload_inputs!(a : Arena, ids : Array(Int32), mask : Array(Int32),
                            cfg : WanT5Config, prefix : String = "t5.input.") : Nil
      cfg.validate!
      unless ids.size == cfg.rows && mask.size == cfg.rows &&
             ids.all? { |id| id >= 0 && id < cfg.vocab } && mask.all? { |m| m == 0 || m == 1 }
        raise ArgumentError.new("Invalid Wan UMT5 token IDs or mask")
      end
      cfg.batches.times do |b|
        unless mask[b * cfg.tokens, cfg.tokens].any? { |m| m == 1 }
          raise ArgumentError.new("Wan UMT5 requires an unmasked token per sample")
        end
      end
      a.upload(prefix + "ids", ids.map(&.to_f32))
      a.upload(prefix + "mask", mask.map(&.to_f32))
    end

    def self.encode(chain : ML::Metal::MPSCommandChain, a : Arena, cfg : WanT5Config,
                    ids : ML::MetalBuffer, mask : ML::MetalBuffer, output : ML::MetalBuffer,
                    plans : WanT5MPSPlans,
                    weights : String = "t5.", scratch : String = "t5.s.",
                    trace : Hash(String, ML::MetalBuffer)? = nil,
                    source_gelu : Bool = false, partial_trace : Bool = false) : Nil
      validate_trace_request!(trace, partial_trace)
      cfg.validate!
      plans.validate!(cfg, a[scratch + "norm"], a[scratch + "q"], a[scratch + "k"],
        a[scratch + "v"], a[scratch + "attention"], a[scratch + "projection"],
        a[scratch + "gate"], a[scratch + "fc1"], a[scratch + "ffn_out"],
        a[scratch + "fc2_weight_f32"], a[scratch + "rms_square"], a[scratch + "rms_mean"])
      n = cfg.rows * cfg.dim
      MetalOps.check(ids, cfg.rows.to_i64)
      MetalOps.check(mask, cfg.rows.to_i64)
      MetalOps.check(output, n.to_i64)
      MetalOps.distinct(output, ids, mask)
      {"x", "norm", "q", "k", "v", "attention", "projection", "ffn_out"}.each { |name| MetalOps.check(a[scratch + name], n.to_i64) }
      {"gate", "fc1"}.each { |name| MetalOps.check(a[scratch + name], cfg.rows.to_i64 * cfg.ffn_dim) }
      writable = {"x", "norm", "q", "k", "v", "attention", "projection", "ffn_out",
                  "rms_square", "rms_mean", "gate", "fc1"}.map { |s| a[scratch + s] }.to_a
      if attention_plan = plans.attention
        attention_plan.validate_shape!(cfg)
        attention_plan.validate_buffers!(a[scratch + "q"], a[scratch + "k"], a[scratch + "v"],
          a[scratch + "attention_bias"], a[scratch + "attention_logits"],
          a[scratch + "attention_scores"], a[scratch + "attention_probs"], a[scratch + "attention"])
        {"attention_bias", "attention_logits", "attention_scores", "attention_probs"}.each do |name|
          MetalOps.check(a[scratch + name], attention_score_count(cfg).to_i64)
          writable << a[scratch + name]
        end
      end
      writable.each_with_index do |buffer, i|
        MetalOps.distinct(buffer, ids, mask, a[scratch + "zero"])
        i.times { |j| MetalOps.distinct(buffer, writable[j]) }
      end
      MetalOps.distinct(output, a[scratch + "zero"])
      checkpoint_bindings(cfg, weights).each do |b|
        a.check_weight(b.target, b.shape.product)
        writable.each { |buffer| MetalOps.distinct(buffer, a[b.target]) }
        MetalOps.distinct(output, a[b.target])
      end
      if trace
        names = ["embedding", "output"]
        cfg.layers.times do |l|
          {"norm1", "q", "k", "v", "attention", "attn_projection", "attn_residual", "norm2", "gate_pre", "gate", "fc1", "product", "ffn", "residual"}.each { |s| names << "blocks.#{l}.#{s}" }
        end
        valid_trace = if partial_trace
                        trace.has_key?("embedding") && trace.has_key?("output") &&
                          trace.keys.all? { |name| names.includes?(name) }
                      else
                        trace.keys.sort == names.sort
                      end
        unless valid_trace
          raise ArgumentError.new("Wan UMT5 trace set differs from the complete checkpoint set")
        end
        trace.each do |name, buffer|
          count = {".gate_pre", ".gate", ".fc1", ".product"}.any? { |s| name.ends_with?(s) } ? cfg.rows.to_i64 * cfg.ffn_dim : n.to_i64
          MetalOps.check(buffer, count)
          MetalOps.distinct(buffer, ids, mask, output)
          a.buffers.each_value { |resident| MetalOps.distinct(buffer, resident) }
        end
        values = trace.values
        values.each_with_index { |v, i| values[0, i].each { |prior| MetalOps.distinct(v, prior) } }
      end
      x = a[scratch + "x"]
      embedding_name = weights + "token_embedding.weight"
      e = chain.encoder
      e.set_pipeline(pipeline("wan_t5_embed"))
      e.set_buffer(ids, 0).set_buffer(a[embedding_name], 1).set_buffer(x, 2, ML::Metal::BufferAccess::Write)
      e.set_value(n.to_u32, 3).set_value(cfg.dim.to_u32, 4)
      e.set_value(a.dtype(embedding_name).bf16? ? 1_u32 : 0_u32, 5)
      e.set_buffer(a[embedding_name], 6)
      e.dispatch_1d(n)
      capture(e, trace, "embedding", x, n)
      cfg.layers.times do |l|
        p = weights + "blocks.#{l}."
        t = "blocks.#{l}."
        norm = a[scratch + "norm"]
        rms_norm_mps_mean(chain, plans.rms_mean, cfg, x, a[p + "norm1.weight"], norm,
          a[scratch + "rms_square"], a[scratch + "rms_mean"], a.dtype(p + "norm1.weight"))
        e = chain.encoder
        capture(e, trace, t + "norm1", norm, n)
        w = p + "attn.q.weight"
        mps_projection(e, norm, a[w], a[scratch + "q"], cfg.rows, cfg.dim, cfg.dim,
          a.dtype(w), plans.q, a[scratch + "fc2_weight_f32"])
        capture(e, trace, t + "q", a[scratch + "q"], n)
        w = p + "attn.k.weight"
        mps_projection(e, norm, a[w], a[scratch + "k"], cfg.rows, cfg.dim, cfg.dim,
          a.dtype(w), plans.k, a[scratch + "fc2_weight_f32"])
        capture(e, trace, t + "k", a[scratch + "k"], n)
        w = p + "attn.v.weight"
        mps_projection(e, norm, a[w], a[scratch + "v"], cfg.rows, cfg.dim, cfg.dim,
          a.dtype(w), plans.v, a[scratch + "fc2_weight_f32"])
        capture(e, trace, t + "v", a[scratch + "v"], n)
        bias = p + "pos_embedding.embedding.weight"
        attention = a[scratch + "attention"]
        if attention_plan = plans.attention
          e.set_pipeline(pipeline("wan_t5_attention_bias"))
          e.set_buffer(a[bias], 0).set_buffer(a[bias], 1).set_buffer(mask, 2)
          e.set_buffer(a[scratch + "attention_bias"], 3, ML::Metal::BufferAccess::Write)
          e.set_value(cfg.tokens.to_u32, 4).set_value(cfg.heads.to_u32, 5)
          e.set_value(cfg.buckets.to_u32, 6).set_value(cfg.max_distance.to_u32, 7)
          e.set_value(a.dtype(bias).bf16? ? 1_u32 : 0_u32, 8)
          e.set_value(attention_score_count(cfg).to_u32, 9).dispatch_1d(attention_score_count(cfg))
          chain.pause!
          attention_plan.encode(chain.handle)
          chain.resume!
          e = chain.encoder
        else
          attention_kernel(e, a[scratch + "q"], a[scratch + "k"], a[scratch + "v"],
            a[bias], mask, attention, cfg, a.dtype(bias))
        end
        capture(e, trace, t + "attention", attention, n)
        projection = a[scratch + "projection"]
        w = p + "attn.o.weight"
        mps_projection(e, attention, a[w], projection, cfg.rows, cfg.dim, cfg.dim,
          a.dtype(w), plans.o, a[scratch + "fc2_weight_f32"])
        capture(e, trace, t + "attn_projection", projection, n)
        MetalOps.add(e, x, projection, x, n)
        capture(e, trace, t + "attn_residual", x, n)
        rms_norm_mps_mean(chain, plans.rms_mean, cfg, x, a[p + "norm2.weight"], norm,
          a[scratch + "rms_square"], a[scratch + "rms_mean"], a.dtype(p + "norm2.weight"))
        e = chain.encoder
        capture(e, trace, t + "norm2", norm, n)
        gate = a[scratch + "gate"]
        fc1 = a[scratch + "fc1"]
        w = p + "ffn.gate.0.weight"
        mps_projection(e, norm, a[w], gate, cfg.rows, cfg.dim, cfg.ffn_dim,
          a.dtype(w), plans.gate, a[scratch + "fc2_weight_f32"])
        capture(e, trace, t + "gate_pre", gate, cfg.rows * cfg.ffn_dim)
        if source_gelu
          gelu_source(e, gate, gate, cfg.rows * cfg.ffn_dim)
        else
          MetalOps.gelu(e, gate, gate, cfg.rows * cfg.ffn_dim)
        end
        capture(e, trace, t + "gate", gate, cfg.rows * cfg.ffn_dim)
        w = p + "ffn.fc1.weight"
        mps_projection(e, norm, a[w], fc1, cfg.rows, cfg.dim, cfg.ffn_dim,
          a.dtype(w), plans.fc1, a[scratch + "fc2_weight_f32"])
        capture(e, trace, t + "fc1", fc1, cfg.rows * cfg.ffn_dim)
        e.set_pipeline(pipeline("wan_t5_product"))
        e.set_buffer(fc1, 0, ML::Metal::BufferAccess::ReadWrite).set_buffer(gate, 1)
        e.set_value((cfg.rows * cfg.ffn_dim).to_u32, 2).dispatch_1d(cfg.rows * cfg.ffn_dim)
        capture(e, trace, t + "product", fc1, cfg.rows * cfg.ffn_dim)
        w = p + "ffn.fc2.weight"
        down_projection(e, fc1, a[w], a[scratch + "ffn_out"], cfg.rows, cfg.ffn_dim, cfg.dim,
          a.dtype(w), plans.fc2, a[scratch + "fc2_weight_f32"])
        capture(e, trace, t + "ffn", a[scratch + "ffn_out"], n)
        MetalOps.add(e, x, a[scratch + "ffn_out"], x, n)
        capture(e, trace, t + "residual", x, n)
      end
      w = weights + "norm.weight"
      rms_norm_mps_mean(chain, plans.rms_mean, cfg, x, a[w], output,
        a[scratch + "rms_square"], a[scratch + "rms_mean"], a.dtype(w))
      e = chain.encoder
      capture(e, trace, "output", output, n)
    end

    # Validate the trace mode independently of device setup so partial tracing
    # cannot silently become an untraced encode. In partial mode the caller
    # chooses a recognized subset, but must include embedding and output.
    def self.validate_trace_request!(trace : Hash(String, ML::MetalBuffer)?,
                                     partial_trace : Bool) : Nil
      if partial_trace && trace.nil?
        raise ArgumentError.new("Wan UMT5 partial trace requires a trace buffer map")
      end
    end

    # Wan pads the unpadded UMT5 context with zeros before its text projection.
    def self.mask_context(e : ML::Metal::ComputeEncoder, x : ML::MetalBuffer,
                          mask : ML::MetalBuffer, output : ML::MetalBuffer, cfg : WanT5Config) : Nil
      cfg.validate!
      MetalOps.check(x, cfg.rows.to_i64 * cfg.dim)
      MetalOps.check(mask, cfg.rows.to_i64)
      MetalOps.check(output, cfg.rows.to_i64 * cfg.dim)
      MetalOps.inplace_or_distinct(x, output)
      MetalOps.distinct(output, mask)
      e.set_pipeline(pipeline("wan_t5_mask_context"))
      e.set_buffer(x, 0).set_buffer(mask, 1).set_buffer(output, 2, ML::Metal::BufferAccess::Write)
      e.set_value((cfg.rows * cfg.dim).to_u32, 3).set_value(cfg.dim.to_u32, 4).dispatch_1d(cfg.rows * cfg.dim)
    end

    private def self.capture(e, trace, name, x, n) : Nil
      if trace
        if destination = trace[name]?
          MetalOps.copy(e, x, destination, n)
        end
      end
    end
  end
end
