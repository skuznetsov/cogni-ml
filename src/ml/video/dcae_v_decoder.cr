require "./dcae_v_encoder"
require "./dcae_v_decoder_c8"
require "./dcae_v_decoder_profile"

module ML::Video
  # The released decoder's cache grid is ten latent frames. A short final
  # tile takes a different upstream padding path and is deliberately rejected.
  struct DCAEVDecoderConfig
    TILE_FRAMES = 10
    getter height : Int32
    getter width : Int32
    getter frames : Int32
    getter batches : Int32

    def initialize(@height : Int32, @width : Int32, @frames : Int32,
                   @batches : Int32 = 1)
      unless {@height, @width, @frames, @batches}.all?(&.positive?)
        raise ArgumentError.new("DCAE-V decoder dimensions must be positive")
      end
      unless @frames % TILE_FRAMES == 0
        raise ArgumentError.new("DCAE-V decoder requires multiples of 10 latent frames")
      end
      unless @height <= Int32::MAX // 32 && @width <= Int32::MAX // 32 && @frames <= Int32::MAX // 4
        raise ArgumentError.new("DCAE-V decoder geometry overflows")
      end
      input_values
      output_values
      DCAEVShape.count("decoder tile scratch", @batches, 256, 40, @height * 16, @width * 16)
    end

    def tile_count : Int32
      @frames // TILE_FRAMES
    end

    def tile_offset(tile : Int32) : Int32
      raise ArgumentError.new("DCAE-V decoder tile index is out of range") unless 0 <= tile < tile_count
      tile * TILE_FRAMES
    end

    def output_frames : Int32
      @frames * 4
    end

    def output_height : Int32
      @height * 32
    end

    def output_width : Int32
      @width * 32
    end

    def input_values : Int32
      DCAEVShape.count("decoder latent", @batches, 32, @frames, @height, @width)
    end

    def output_values : Int32
      DCAEVShape.count("decoder output", @batches, 3, output_frames, output_height, output_width)
    end

    def input_bytes : Int64
      input_values.to_i64 * 2
    end

    def output_bytes : Int64
      output_values.to_i64 * 2
    end
  end

  class DCAEVDecoderWorkspace
    getter arena : Arena
    getter config : DCAEVDecoderConfig
    getter planned_bytes : Int64
    getter packed_c8 : Bool
    getter packed_c8_stage1 : Bool

    def initialize(@arena : Arena, @config : DCAEVDecoderConfig, @planned_bytes : Int64,
                   @packed_c8 : Bool = false, @packed_c8_stage1 : Bool = false)
    end

    def scratch(index : Int32) : ML::MetalBuffer
      raise ArgumentError.new("Invalid DCAE-V decoder scratch slot") unless 0 <= index < 3
      @arena["dcae_v_decoder.scratch.#{index}"]
    end

    def cache(name : String) : ML::MetalBuffer
      @arena["dcae_v_decoder.cache.#{name}"]
    end

    def mean_a : ML::MetalBuffer
      @arena["dcae_v_decoder.mean_a"]
    end

    def mean_b : ML::MetalBuffer
      @arena["dcae_v_decoder.mean_b"]
    end
  end

  # An immutable sequence of fixed bindings. Every invocation starts with a
  # no-cache tile and overwrites all cache frames on GPU before their next use.
  # Closures, graph resources and all buffer owners are prepared before encode.
  struct DCAEVDecoderPlanOperation
    getter name : String
    getter category : String
    getter stage : String
    getter encode : Proc(ML::Metal::MPSCommandChain, Nil)

    def initialize(@name : String, @category : String, @stage : String,
                   @encode : Proc(ML::Metal::MPSCommandChain, Nil))
    end
  end

  class DCAEVDecoderPlan
    @operations = [] of DCAEVDecoderPlanOperation
    @convs = [] of DCAEVConvMPS::Plan
    @norms = [] of DCAEVNormMPS::RMSNormPlan
    @silus = [] of DCAEVNormMPS::SiLUPlan
    @templates = {} of String => DCAEVConvMPS::Template
    @last_chain : ML::Metal::MPSCommandChain? = nil
    @released = false
    @owners : Array(ML::MetalBuffer)
    getter graph_private_bytes_known : Bool = false
    getter c8_convolution_count = 0_i32

    def initialize(@owners : Array(ML::MetalBuffer), @c8 : DCAEVDecoderC8::Plan? = nil,
                   @c8_stage1 : DCAEVDecoderC8::Plan? = nil)
    end

    def add(name : String, category : String, stage : String,
            &operation : ML::Metal::MPSCommandChain -> Nil) : Nil
      @operations << DCAEVDecoderPlanOperation.new(name, category, stage, operation)
    end

    def template(key : String, &factory : -> DCAEVConvMPS::Template) : DCAEVConvMPS::Template
      @templates[key] ||= factory.call
    end

    def add(plan : DCAEVConvMPS::Plan, name : String,
            category : String, stage : String) : Nil
      @convs << plan
      add(name, category, stage) { |chain| plan.encode(chain) }
    end

    def add_c8(input : ML::MetalBuffer, cache : ML::MetalBuffer?, weight : ML::MetalBuffer,
               bias : ML::MetalBuffer?, output : ML::MetalBuffer, name : String,
               stage : String, stage_index : Int32) : Nil
      c8 = (stage_index == 1 ? @c8_stage1 : @c8) ||
           raise ArgumentError.new("DCAE-V C8 workspace was not prepared for stage #{stage_index}")
      @c8_convolution_count += 1
      add(name, "residual_conv3d_c8", stage) do |chain|
        c8.encode(chain, input, cache, weight, bias, output)
      end
    end

    def add(plan : DCAEVNormMPS::RMSNormPlan, name : String,
            stage : String) : Nil
      @norms << plan
      add(name, "norm", stage) { |chain| plan.encode(chain) }
    end

    def add(plan : DCAEVNormMPS::SiLUPlan, name : String,
            stage : String) : Nil
      @silus << plan
      add(name, "silu", stage) { |chain| plan.encode(chain) }
    end

    def encode(chain : ML::Metal::MPSCommandChain,
               *, profile : DCAEVDecoderProfileRecorder? = nil) : Nil
      raise ArgumentError.new("DCAE-V decoder plan has been released") if @released
      if previous = @last_chain
        unless previous.same?(chain) || previous.completed?
          raise ArgumentError.new("DCAE-V decoder plan cannot span in-flight chains")
        end
      end
      @last_chain = chain
      @operations.each do |operation|
        if recorder = profile
          recorder.measure(chain, operation.name, operation.category, operation.stage) do
            operation.encode.call(chain)
          end
        else
          operation.encode.call(chain)
        end
      end
    end

    def operation_count : Int32
      @operations.size
    end

    def template_count : Int32
      @templates.size
    end

    def release : Nil
      return if @released
      if previous = @last_chain
        raise ArgumentError.new("Drain the DCAE-V decoder chain before release") unless previous.completed?
      end
      @convs.each(&.release)
      @norms.each(&.release)
      @silus.each(&.release)
      @templates.each_value(&.release)
      @operations.clear
      @convs.clear
      @norms.clear
      @silus.clear
      @templates.clear
      @owners.clear
      @c8 = nil
      @c8_stage1 = nil
      @last_chain = nil
      @released = true
    end
  end

  module DCAEVDecoder
    SOURCE = {{ read_file("#{__DIR__}/kernels/dcae_v_decoder.metal") }}
    WIDTHS = [128, 256, 512, 512, 1024, 1024, 1024]
    DEPTHS = [0, 5, 10, 4, 4, 4, 4]
    private record CacheSpec, name : String, channels : Int32, height : Int32, width : Int32

    def self.pipeline(name : String) : ML::Metal::ComputePipeline
      ML::Metal::PipelineCache.get(name) { ML::Metal::ComputePipeline.new(name, SOURCE) }
    end

    def self.prepare! : Nil
      DCAEVEncoder.prepare!
      {"dcae_v_decoder_gather_latent_bf16", "dcae_v_decoder_pixel_shuffle_bf16",
       "dcae_v_decoder_scatter_output_bf16"}.each { |name| pipeline(name) }
    end

    private def self.residual_prefix(stage : Int32, block : Int32) : String
      offset = stage == 6 ? 0 : 1
      "decoder.stages.#{stage}.op_list.#{block + offset}.main"
    end

    private def self.upsample_prefix(stage : Int32) : String
      "decoder.stages.#{stage}.op_list.0.main.conv"
    end

    def self.checkpoint_bindings(cfg : DCAEVDecoderConfig) : Array(WeightBinding)
      bindings = [] of WeightBinding
      add_conv_bindings(bindings, "decoder.project_in", 32, 1024, true)
      6.downto(1) do |stage|
        channels = WIDTHS[stage]
        if stage < 6
          add_conv_bindings(bindings, upsample_prefix(stage), WIDTHS[stage + 1], channels * 4, true)
        end
        DEPTHS[stage].times do |block|
          prefix = residual_prefix(stage, block)
          add_conv_bindings(bindings, "#{prefix}.conv1", channels, channels, true)
          add_conv_bindings(bindings, "#{prefix}.conv2", channels, channels, false)
          add_norm_bindings(bindings, "#{prefix}.conv2.norm", channels)
        end
      end
      add_norm_bindings(bindings, "decoder.project_out.op_list.0", 256)
      add_conv_bindings(bindings, "decoder.project_out.op_list.2.conv", 256, 12, true)
      bindings
    end

    private def self.add_binding(bindings : Array(WeightBinding), name : String, shape : Array(Int64)) : Nil
      bindings << WeightBinding.new(name, name, shape, ML::GGUF::SafeTensorDType::BF16)
    end

    private def self.add_conv_bindings(bindings : Array(WeightBinding), prefix : String,
                                       input : Int32, output : Int32, bias : Bool) : Nil
      add_binding(bindings, "#{prefix}.conv.weight", [output.to_i64, input.to_i64, 3_i64, 3_i64, 3_i64])
      add_binding(bindings, "#{prefix}.conv.bias", [output.to_i64]) if bias
    end

    private def self.add_norm_bindings(bindings : Array(WeightBinding), prefix : String, channels : Int32) : Nil
      add_binding(bindings, "#{prefix}.weight", [channels.to_i64])
      add_binding(bindings, "#{prefix}.bias", [channels.to_i64])
    end

    def self.weight_bytes(cfg : DCAEVDecoderConfig) : Int64
      checkpoint_bindings(cfg).sum { |binding| binding.shape.product * 2 }
    end

    def self.scratch_values(cfg : DCAEVDecoderConfig) : Int32
      DCAEVShape.count("decoder activation", cfg.batches, 256, 40, cfg.height * 16, cfg.width * 16)
    end

    private def self.mean_values(cfg : DCAEVDecoderConfig) : Int32
      DCAEVShape.count("decoder RMS rows", cfg.batches, 40, cfg.height * 16, cfg.width * 16)
    end

    private def self.cache_specs(cfg : DCAEVDecoderConfig) : Array(CacheSpec)
      specs = [CacheSpec.new("decoder.project_in", 32, cfg.height, cfg.width)]
      h, w = cfg.height, cfg.width
      6.downto(1) do |stage|
        if stage < 6
          specs << CacheSpec.new(upsample_prefix(stage), WIDTHS[stage + 1], h, w)
          if stage < 5
            h *= 2
            w *= 2
          end
        end
        DEPTHS[stage].times do |block|
          prefix = residual_prefix(stage, block)
          specs << CacheSpec.new("#{prefix}.conv1", WIDTHS[stage], h, w)
          specs << CacheSpec.new("#{prefix}.conv2", WIDTHS[stage], h, w)
        end
      end
      specs << CacheSpec.new("decoder.project_out.op_list.2.conv", 256, h, w)
      specs
    end

    def self.workspace_required_bytes(cfg : DCAEVDecoderConfig, *, packed_c8 : Bool = false,
                                      packed_c8_stage1 : Bool = false) : Int64
      if packed_c8_stage1 && !packed_c8
        raise ArgumentError.new("DCAE-V stage-1 C8 requires stage-2 C8")
      end
      caches = cache_specs(cfg).sum do |spec|
        DCAEVShape.count("decoder cache", cfg.batches, spec.channels, spec.height, spec.width).to_i64
      end
      planned = (scratch_values(cfg).to_i64 * 3 + caches + mean_values(cfg).to_i64 * 2) * 2
      if packed_c8
        unless cfg.batches == 1 && cfg.height == 9 && cfg.width == 16
          raise ArgumentError.new("DCAE-V C8 requires a single latent tile grid of 9x16")
        end
        planned += packed_c8_stage1 ? DCAEVDecoderC8.shared_scratch_bytes : DCAEVDecoderC8.scratch_bytes
      end
      planned
    end

    def self.trace_shapes(cfg : DCAEVDecoderConfig) : Hash(String, Array(Int64))
      result = {} of String => Array(Int64)
      cfg.tile_count.times do |tile|
        h, w, t = cfg.height, cfg.width, 10
        result["trace.project_in.tile#{tile}"] = shape(cfg, 1024, t, h, w)
        6.downto(1) do |stage|
          if stage == 5
            t *= 4
          elsif stage < 5
            h *= 2
            w *= 2
          end
          result["trace.stages.#{stage}.tile#{tile}"] = shape(cfg, WIDTHS[stage], t, h, w)
        end
        result["trace.project_out.tile#{tile}"] = shape(cfg, 3, 40, h * 2, w * 2)
      end
      result
    end

    private def self.shape(cfg : DCAEVDecoderConfig, c : Int32, t : Int32, h : Int32, w : Int32) : Array(Int64)
      [cfg.batches.to_i64, c.to_i64, t.to_i64, h.to_i64, w.to_i64]
    end

    def self.prepare!(arena : Arena, cfg : DCAEVDecoderConfig,
                      budget_bytes : Int64, reserve_bytes : Int64 = 0_i64,
                      *, packed_c8 : Bool = false, packed_c8_stage1 : Bool = false) : DCAEVDecoderWorkspace
      raise ArgumentError.new("DCAE-V decoder workspace must start empty") unless arena.resident_bytes == 0
      planned = workspace_required_bytes(cfg, packed_c8: packed_c8, packed_c8_stage1: packed_c8_stage1)
      unless budget_bytes > 0 && reserve_bytes >= 0 && planned <= budget_bytes && reserve_bytes <= budget_bytes - planned
        raise ArgumentError.new("DCAE-V decoder workspace exceeds residency budget")
      end
      prepare!
      allocated = [] of String
      begin
        3.times do |index|
          name = "dcae_v_decoder.scratch.#{index}"
          arena.alloc_weight(name, scratch_values(cfg), WeightDType::BF16)
          allocated << name
        end
        cache_specs(cfg).each do |spec|
          name = "dcae_v_decoder.cache.#{spec.name}"
          arena.alloc_weight(name, DCAEVShape.count("decoder cache", cfg.batches,
            spec.channels, spec.height, spec.width), WeightDType::BF16)
          allocated << name
        end
        {"mean_a", "mean_b"}.each do |slot|
          name = "dcae_v_decoder.#{slot}"
          arena.alloc_weight(name, mean_values(cfg), WeightDType::BF16)
          allocated << name
        end
        if packed_c8
          input_values = packed_c8_stage1 ? DCAEVDecoderC8::SHARED_INPUT_PACK_VALUES : DCAEVDecoderC8::INPUT_PACK_VALUES
          {DCAEVDecoderC8::INPUT_PACK  => input_values,
           DCAEVDecoderC8::WEIGHT_PACK => DCAEVDecoderC8::WEIGHT_PACK_VALUES}.each do |name, count|
            arena.alloc_weight(name, count, WeightDType::BF16)
            allocated << name
          end
        end
      rescue ex
        allocated.each { |name| arena.remove(name) }
        raise ex
      end
      raise "DCAE-V decoder workspace allocation differs from plan" unless arena.resident_bytes == planned
      DCAEVDecoderWorkspace.new(arena, cfg, planned, packed_c8, packed_c8_stage1)
    end

    # All validation and graph preparation precede the timed encode. Optional
    # diagnostic targets are fixed caller-owned GPU buffers, not host reads.
    def self.prepare_mps!(input : ML::MetalBuffer, output : ML::MetalBuffer,
                          weights : Arena, workspace : DCAEVDecoderWorkspace,
                          cfg : DCAEVDecoderConfig,
                          traces = {} of String => ML::MetalBuffer) : DCAEVDecoderPlan
      unless workspace.config == cfg && workspace.planned_bytes == workspace_required_bytes(cfg,
               packed_c8: workspace.packed_c8, packed_c8_stage1: workspace.packed_c8_stage1) &&
             workspace.arena.resident_bytes == workspace.planned_bytes
        raise ArgumentError.new("DCAE-V decoder workspace/config mismatch")
      end
      MetalOps.check_storage(input, cfg.input_values.to_i64, WeightDType::BF16)
      MetalOps.check_storage(output, cfg.output_values.to_i64, WeightDType::BF16)
      checkpoint_bindings(cfg).each do |binding|
        weights.check_weight(binding.target, binding.shape.product)
        raise ArgumentError.new("DCAE-V decoder requires BF16 weights") unless weights.dtype(binding.target).bf16?
      end
      expected_traces = trace_shapes(cfg)
      traces.each do |name, target|
        dims = expected_traces[name]? || raise ArgumentError.new("Unknown DCAE-V decoder trace #{name}")
        MetalOps.check_storage(target, dims.product, WeightDType::BF16)
      end
      owners = weights.buffers.values + workspace.arena.buffers.values + [input, output] + traces.values
      owners.each_with_index do |buffer, index|
        index.times do |other|
          raise ArgumentError.new("DCAE-V decoder buffers overlap") if MetalOps.overlap?(buffer, owners[other])
        end
      end
      c8 = DCAEVDecoderC8::Plan.new(workspace.arena) if workspace.packed_c8
      c8_stage1 = DCAEVDecoderC8::Plan.new(workspace.arena, stage: 1) if workspace.packed_c8_stage1
      plan = DCAEVDecoderPlan.new(owners, c8, c8_stage1)
      begin
        cfg.tile_count.times do |tile|
          h, w, t, slot = cfg.height, cfg.width, 10, 0
          add_gather(plan, input, workspace.scratch(slot), cfg, tile)
          add_conv(plan, workspace.scratch(slot), workspace.scratch(1), weights, workspace,
            cfg, tile, "decoder.project_in", 32, 1024, t, h, w, true)
          slot = 1
          add_trace(plan, traces, "trace.project_in.tile#{tile}", workspace.scratch(slot), cfg, 1024, t, h, w)
          6.downto(1) do |stage|
            channels = WIDTHS[stage]
            if stage < 6
              sf, tf = stage == 5 ? 1 : 2, stage == 5 ? 4 : 1
              original = workspace.scratch(slot)
              expanded = workspace.scratch((slot + 1) % 3)
              next_slot = (slot + 2) % 3
              add_conv(plan, original, expanded, weights, workspace, cfg, tile,
                upsample_prefix(stage), WIDTHS[stage + 1], channels * 4, t, h, w, true)
              add_shuffle(plan, expanded, original, workspace.scratch(next_slot), cfg,
                WIDTHS[stage + 1], channels, t, h, w, sf, tf, true,
                "tile#{tile}/stage_#{stage}.upsample.shuffle", "stage_#{stage}")
              slot = next_slot
              t *= tf
              h *= sf
              w *= sf
            end
            DEPTHS[stage].times do |block|
              prefix = residual_prefix(stage, block)
              original = workspace.scratch(slot)
              temp = workspace.scratch((slot + 2) % 3)
              next_slot = (slot + 1) % 3
              result = workspace.scratch(next_slot)
              add_conv(plan, original, temp, weights, workspace, cfg, tile,
                "#{prefix}.conv1", channels, channels, t, h, w, true)
              plan.add(DCAEVNormMPS::SiLUPlan.new(shape(cfg, channels, t, h, w).map(&.to_i32), temp, result),
                "tile#{tile}/#{prefix}.silu", "stage_#{stage}")
              add_conv(plan, result, temp, weights, workspace, cfg, tile,
                "#{prefix}.conv2", channels, channels, t, h, w, false)
              add_norm(plan, temp, result, weights, workspace, cfg, tile, "#{prefix}.conv2.norm", channels, t, h, w)
              add_residual(plan, result, original, DCAEVShape.count("decoder residual", cfg.batches, channels, t, h, w),
                "tile#{tile}/#{prefix}.residual_add", "stage_#{stage}")
              slot = next_slot
            end
            add_trace(plan, traces, "trace.stages.#{stage}.tile#{tile}", workspace.scratch(slot), cfg, channels, t, h, w)
          end
          normed = workspace.scratch((slot + 1) % 3)
          activated = workspace.scratch((slot + 2) % 3)
          add_norm(plan, workspace.scratch(slot), normed, weights, workspace, cfg, tile,
            "decoder.project_out.op_list.0", 256, t, h, w)
          plan.add(DCAEVNormMPS::SiLUPlan.new(shape(cfg, 256, t, h, w).map(&.to_i32), normed, activated),
            "tile#{tile}/decoder.project_out.silu", "project_out")
          expanded = workspace.scratch(slot)
          add_conv(plan, activated, expanded, weights, workspace, cfg, tile,
            "decoder.project_out.op_list.2.conv", 256, 12, t, h, w, true)
          add_shuffle(plan, expanded, activated, normed, cfg, 256, 3, t, h, w, 2, 1, false,
            "tile#{tile}/decoder.project_out.shuffle", "project_out")
          add_trace(plan, traces, "trace.project_out.tile#{tile}", normed, cfg, 3, 40, h * 2, w * 2)
          add_scatter(plan, normed, output, cfg, tile)
        end
      rescue ex
        plan.release
        raise ex
      end
      plan
    end

    private def self.add_conv(plan : DCAEVDecoderPlan, input : ML::MetalBuffer,
                              output : ML::MetalBuffer, weights : Arena,
                              workspace : DCAEVDecoderWorkspace, cfg : DCAEVDecoderConfig,
                              tile : Int32, prefix : String, cin : Int32, cout : Int32,
                              t : Int32, h : Int32, w : Int32, bias : Bool) : Nil
      cache = workspace.cache(prefix)
      stage = profile_stage(prefix)
      category = if prefix == "decoder.project_in" || prefix.starts_with?("decoder.project_out")
                   "project_conv3d"
                 elsif prefix.ends_with?(".conv")
                   "upsample_conv3d"
                 else
                   "residual_conv3d"
                 end
      live_cache = tile > 0 ? cache : nil
      weight = weights["#{prefix}.conv.weight"]
      bias_buffer = bias ? weights["#{prefix}.conv.bias"] : nil
      operation = "tile#{tile}/#{prefix}.conv3d"
      c8_stage = case stage
                 when "stage_1" then 1 if workspace.packed_c8_stage1
                 when "stage_2" then 2 if workspace.packed_c8
                 end
      if c8_stage && category == "residual_conv3d" &&
         DCAEVDecoderC8.eligible?(cfg.batches, cin, cout, t, h, w, stage: c8_stage)
        plan.add_c8(input, live_cache, weight, bias_buffer, output, operation, stage, c8_stage)
      else
        key = [cfg.batches, cin, cout, t, h, w, tile > 0, bias].map(&.to_s).join("|")
        template = plan.template(key) do
          DCAEVConvMPS::Template.new(cfg.batches, cin, cout, cout, t, t, h, w,
            0, false, true, tile > 0, bias, 1, 1, false)
        end
        plan.add(template.bind(input, live_cache, weight, bias_buffer, output), operation, category, stage)
      end
      cache_pipeline = DCAEVEncoder.pipeline("dcae_v_cache_last_frame_bf16")
      count = DCAEVShape.count("decoder cache", cfg.batches, cin, h, w)
      plan.add("tile#{tile}/#{prefix}.cache_last_frame", "cache_store", stage) do |chain|
        e = chain.encoder
        e.set_pipeline(cache_pipeline)
        e.set_buffer(input, 0).set_buffer(cache, 1, ML::Metal::BufferAccess::Write)
        e.set_value(cin.to_u32, 2).set_value(t.to_u32, 3).set_value(t.to_u32, 4)
        e.set_value(0_u32, 5).set_value(h.to_u32, 6).set_value(w.to_u32, 7).set_value(cfg.batches.to_u32, 8)
        e.dispatch_1d(count)
      end
    end

    private def self.add_norm(plan : DCAEVDecoderPlan, input : ML::MetalBuffer,
                              output : ML::MetalBuffer, weights : Arena,
                              workspace : DCAEVDecoderWorkspace, cfg : DCAEVDecoderConfig,
                              tile : Int32, prefix : String, c : Int32, t : Int32, h : Int32, w : Int32) : Nil
      plan.add(DCAEVNormMPS::RMSNormPlan.new(shape(cfg, c, t, h, w).map(&.to_i32), input,
        output, workspace.mean_a, workspace.mean_b, weights["#{prefix}.weight"], weights["#{prefix}.bias"]),
        "tile#{tile}/#{prefix}.rmsnorm", profile_stage(prefix))
    end

    private def self.add_residual(plan : DCAEVDecoderPlan, result : ML::MetalBuffer,
                                  original : ML::MetalBuffer, count : Int32,
                                  name : String, stage : String) : Nil
      p = DCAEVEncoder.pipeline("dcae_v_residual_add_bf16_inplace")
      plan.add(name, "residual_add", stage) do |chain|
        e = chain.encoder
        e.set_pipeline(p)
        e.set_buffer(result, 0, ML::Metal::BufferAccess::ReadWrite).set_buffer(original, 1)
        e.set_value(count.to_u32, 2).dispatch_1d(count)
        chain.pause!
        chain.continue_async!
        chain.resume!
      end
    end

    private def self.add_gather(plan : DCAEVDecoderPlan, input : ML::MetalBuffer,
                                output : ML::MetalBuffer, cfg : DCAEVDecoderConfig, tile : Int32) : Nil
      p = pipeline("dcae_v_decoder_gather_latent_bf16")
      count = DCAEVShape.count("decoder latent tile", cfg.batches, 32, 10, cfg.height, cfg.width)
      offset = cfg.tile_offset(tile)
      plan.add("tile#{tile}/gather_latent", "gather", "input") do |chain|
        e = chain.encoder
        e.set_pipeline(p)
        e.set_buffer(input, 0).set_buffer(output, 1, ML::Metal::BufferAccess::Write)
        e.set_value(cfg.frames.to_u32, 2).set_value(10_u32, 3)
        e.set_value(cfg.height.to_u32, 4).set_value(cfg.width.to_u32, 5).set_value(offset.to_u32, 6)
        e.set_value(count.to_u32, 7).dispatch_1d(count)
      end
    end

    private def self.add_shuffle(plan : DCAEVDecoderPlan, expanded : ML::MetalBuffer,
                                 shortcut : ML::MetalBuffer, output : ML::MetalBuffer,
                                 cfg : DCAEVDecoderConfig, cin : Int32, cout : Int32,
                                 t : Int32, h : Int32, w : Int32, sf : Int32, tf : Int32,
                                 use_shortcut : Bool, name : String, stage : String) : Nil
      p = pipeline("dcae_v_decoder_pixel_shuffle_bf16")
      expanded_channels = cout * sf * sf * tf
      repeat = use_shortcut ? expanded_channels // cin : 1
      raise ArgumentError.new("Invalid decoder shortcut channel expansion") if use_shortcut && expanded_channels % cin != 0
      count = DCAEVShape.count("decoder shuffled tile", cfg.batches, cout, t * tf, h * sf, w * sf)
      plan.add(name, "shuffle", stage) do |chain|
        e = chain.encoder
        e.set_pipeline(p)
        e.set_buffer(expanded, 0).set_buffer(shortcut, 1).set_buffer(output, 2, ML::Metal::BufferAccess::Write)
        e.set_value(expanded_channels.to_u32, 3).set_value(cin.to_u32, 4)
        e.set_value(cout.to_u32, 5).set_value(t.to_u32, 6).set_value(h.to_u32, 7).set_value(w.to_u32, 8)
        e.set_value(sf.to_u32, 9).set_value(tf.to_u32, 10).set_value(repeat.to_u32, 11)
        e.set_value(use_shortcut ? 1_u32 : 0_u32, 12).set_value(count.to_u32, 13).dispatch_1d(count)
      end
    end

    private def self.add_scatter(plan : DCAEVDecoderPlan, input : ML::MetalBuffer,
                                 output : ML::MetalBuffer, cfg : DCAEVDecoderConfig, tile : Int32) : Nil
      p = pipeline("dcae_v_decoder_scatter_output_bf16")
      count = DCAEVShape.count("decoder RGB tile", cfg.batches, 3, 40, cfg.output_height, cfg.output_width)
      offset = tile * 40
      plan.add("tile#{tile}/scatter_output", "scatter", "output") do |chain|
        e = chain.encoder
        e.set_pipeline(p)
        e.set_buffer(input, 0).set_buffer(output, 1, ML::Metal::BufferAccess::Write)
        e.set_value(cfg.output_frames.to_u32, 2).set_value(40_u32, 3)
        e.set_value(cfg.output_height.to_u32, 4).set_value(cfg.output_width.to_u32, 5)
        e.set_value(offset.to_u32, 6).set_value(count.to_u32, 7).dispatch_1d(count)
      end
    end

    private def self.add_trace(plan : DCAEVDecoderPlan, traces : Hash(String, ML::MetalBuffer),
                               name : String, input : ML::MetalBuffer, cfg : DCAEVDecoderConfig,
                               c : Int32, t : Int32, h : Int32, w : Int32) : Nil
      return unless target = traces[name]?
      p = DCAEVEncoder.pipeline("dcae_v_copy_bf16")
      count = DCAEVShape.count("decoder trace", cfg.batches, c, t, h, w)
      stage = if name.starts_with?("trace.stages.")
                "stage_#{name.split('.')[2]}"
              elsif name.starts_with?("trace.project_in.")
                "project_in"
              else
                "project_out"
              end
      plan.add(name, "trace_copy", stage) do |chain|
        e = chain.encoder
        e.set_pipeline(p)
        e.set_buffer(input, 0).set_buffer(target, 1, ML::Metal::BufferAccess::Write)
        e.set_value(count.to_u32, 2).dispatch_1d(count)
      end
    end

    private def self.profile_stage(prefix : String) : String
      components = prefix.split('.')
      if components.size > 2 && components[0] == "decoder" && components[1] == "stages"
        "stage_#{components[2]}"
      elsif prefix == "decoder.project_in"
        "project_in"
      else
        "project_out"
      end
    end
  end
end
