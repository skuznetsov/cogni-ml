require "../../src/ml/metal/device"
require "../../src/ml/video/dcae_v_decoder"

# Probe-only builder for the smallest decoder prefix that produces the input
# to decoder.stages.2.op_list.1.main.conv1.conv. It deliberately mirrors the
# established decoder operation order while stopping before that convolution;
# the production decoder builder remains unchanged.
class DCAEVActivationPrefix
  TARGET_CONV     = "decoder.stages.2.op_list.1.main.conv1.conv"
  TARGET_WEIGHT   = "#{TARGET_CONV}.weight"
  TARGET_BIAS     = "#{TARGET_CONV}.bias"
  TARGET_CHANNELS = 512
  TARGET_FRAMES   =  40
  TILE_COUNT      =   2

  ZERO_SOURCE = <<-METAL
  #include <metal_stdlib>
  using namespace metal;
  kernel void dcae_v_activation_prefix_zero_bf16(
      device ushort *output [[buffer(0)]],
      constant uint &count [[buffer(1)]],
      uint id [[thread_position_in_grid]]) {
      if (id < count) output[id] = 0;
  }
  METAL

  alias CacheSpec = Tuple(String, Int32, Int32, Int32)

  @arena : ML::Video::Arena
  @prepared_weights : ML::Video::PreparedWeights?
  @plans = [] of ML::Video::DCAEVDecoderPlan
  @scratch = [] of ML::MetalBuffer
  @activations = [] of ML::MetalBuffer
  @old_caches = [] of ML::MetalBuffer
  @internal_caches = {} of String => ML::MetalBuffer
  @input_latent : ML::MetalBuffer?
  @zero_pipeline : ML::Metal::ComputePipeline?
  @cache_pipeline : ML::Metal::ComputePipeline?
  @copy_pipeline : ML::Metal::ComputePipeline?
  @last_chain : ML::Metal::MPSCommandChain? = nil
  @next_tile = 0
  @released = false
  @target_shape : Array(Int32)
  @target_cache_count : Int32
  getter named_bytes : Int64 = 0_i64

  def initialize(checkpoint : ML::Video::CheckpointFile,
                 reference : ML::Video::CheckpointFile)
    config = ML::Video::DCAEVDecoderConfig.new(9, 16, 20)
    @target_shape = [config.batches, TARGET_CHANNELS, TARGET_FRAMES,
                     config.height * 8, config.width * 8]
    target_count = ML::Video::DCAEVNormMPS.checked_count("activation-prefix target", @target_shape)
    @target_cache_count = ML::Video::DCAEVNormMPS.checked_count("activation-prefix target cache",
      [config.batches, TARGET_CHANNELS, 1, config.height * 8, config.width * 8])

    latent = reference.tensors["input.latent"]? ||
             raise ArgumentError.new("Decoder reference is missing input.latent")
    unless latent.dtype.bf16? && latent.shape == [1_i64, 32_i64, 20_i64, 9_i64, 16_i64]
      raise ArgumentError.new("Decoder reference input.latent must be BF16 [1,32,20,9,16]")
    end

    bindings = self.class.prefix_bindings(config)
    workspace_bytes = self.class.workspace_bytes(config)
    weight_bytes = self.class.binding_bytes(bindings)
    budget_bytes = self.class.add_bytes("activation-prefix budget", weight_bytes, workspace_bytes)
    weight_plan = ML::Video::WeightPlan.new([checkpoint], bindings, budget_bytes,
      reserve_bytes: workspace_bytes,
      resident_dtype: ML::Video::WeightDType::BF16)
    unless weight_plan.destination_bytes == weight_bytes && weight_plan.staging_bytes == 0
      raise "Activation-prefix BF16 weight plan differs from its byte estimate"
    end

    @arena = ML::Video::Arena.new
    begin
      @prepared_weights = ML::Video::PreparedWeights.new(weight_plan, @arena)
      allocate_workspace(config, reference, latent, target_count)
      build_plans(config)
      @zero_pipeline = ML::Metal::PipelineCache.get("dcae_v_activation_prefix_zero_bf16") do
        ML::Metal::ComputePipeline.new("dcae_v_activation_prefix_zero_bf16", ZERO_SOURCE,
          "dcae_v_activation_prefix_zero_bf16")
      end
      @cache_pipeline = ML::Video::DCAEVEncoder.pipeline("dcae_v_cache_last_frame_bf16")
      @copy_pipeline = ML::Video::DCAEVEncoder.pipeline("dcae_v_copy_bf16")
      @named_bytes = @arena.resident_bytes
      unless @named_bytes == budget_bytes
        raise "Activation-prefix residency differs from its plan (#{@named_bytes} != #{budget_bytes})"
      end
    rescue ex
      cleanup
      raise ex
    end
  end

  def activation(tile : Int32) : ML::MetalBuffer
    validate_tile!(tile)
    @activations[tile]
  end

  # The first tile's cache is explicitly GPU-zeroed; the second is the final
  # frame of the first tile's pretarget activation, exactly as a causal Conv3D
  # cache stores its input rather than its output.
  def old_cache(tile : Int32) : ML::MetalBuffer
    validate_tile!(tile)
    @old_caches[tile]
  end

  # Append one ten-latent-frame prefix invocation to the caller's live MPS
  # chain. Calls must be tile 0 then tile 1 on that same chain; no waits or
  # readbacks are introduced here.
  def encode_tile(chain : ML::Metal::MPSCommandChain, tile : Int32) : Nil
    raise ArgumentError.new("DCAE-V activation prefix has been released") if @released
    unless tile == @next_tile && tile < TILE_COUNT
      raise ArgumentError.new("Activation-prefix tiles must be encoded once in order 0, 1")
    end
    if previous = @last_chain
      unless previous.same?(chain) && !previous.completed?
        raise ArgumentError.new("Activation-prefix tiles must share one in-flight MPS chain")
      end
    end
    @last_chain = chain
    encoder = chain.encoder
    if tile == 0
      encoder.set_pipeline(@zero_pipeline.not_nil!)
        .set_buffer(@old_caches[0], 0, ML::Metal::BufferAccess::Write)
        .set_value(@target_cache_count.to_u32, 1)
        .dispatch_1d(@target_cache_count)
    else
      encoder.set_pipeline(@cache_pipeline.not_nil!)
        .set_buffer(@activations[0], 0)
        .set_buffer(@old_caches[1], 1, ML::Metal::BufferAccess::Write)
        .set_value(TARGET_CHANNELS.to_u32, 2)
        .set_value(TARGET_FRAMES.to_u32, 3)
        .set_value(TARGET_FRAMES.to_u32, 4)
        .set_value(0_u32, 5)
        .set_value(@target_shape[3].to_u32, 6)
        .set_value(@target_shape[4].to_u32, 7)
        .set_value(1_u32, 8)
        .dispatch_1d(@target_cache_count)
    end
    @plans[tile].encode(chain)
    @next_tile += 1
  end

  def release(chain : ML::Metal::MPSCommandChain? = nil) : Nil
    return if @released
    if previous = @last_chain
      raise ArgumentError.new("Drain the activation-prefix MPS chain before release") unless previous.completed?
      if supplied = chain
        raise ArgumentError.new("Activation-prefix release chain mismatch") unless previous.same?(supplied)
      end
    elsif chain && !chain.not_nil!.completed?
      raise ArgumentError.new("Complete the supplied activation-prefix chain before release")
    end
    cleanup
    @released = true
  end

  def finalize
    return if @released
    return if @last_chain && !@last_chain.not_nil!.completed?
    release
  end

  # Return exactly the checkpoint bindings preceding the target convolution.
  # The production list is already topologically ordered, so the target weight
  # is a precise and auditable cut point.
  def self.prefix_bindings(config : ML::Video::DCAEVDecoderConfig) : Array(ML::Video::WeightBinding)
    all = ML::Video::DCAEVDecoder.checkpoint_bindings(config)
    result = [] of ML::Video::WeightBinding
    found_target = false
    all.each do |binding|
      if binding.target == TARGET_WEIGHT
        found_target = true
        break
      end
      result << binding
    end
    raise ArgumentError.new("DCAE decoder binding inventory lacks the target convolution") unless found_target
    raise ArgumentError.new("Activation prefix unexpectedly includes its target convolution") if result.any? { |b| b.target == TARGET_WEIGHT || b.target == TARGET_BIAS }
    result
  end

  def self.binding_bytes(bindings : Array(ML::Video::WeightBinding)) : Int64
    bindings.sum { |binding| binding.shape.product * ML::Video::WeightDType::BF16.bytes_per_element }
  end

  # Explicitly named GPU storage, including the two saved outputs and both
  # target-cache snapshots. MPSGraph's private allocations remain opaque.
  def self.workspace_bytes(config : ML::Video::DCAEVDecoderConfig) : Int64
    target_shape = [config.batches, TARGET_CHANNELS, TARGET_FRAMES,
                    config.height * 8, config.width * 8]
    activation_values = ML::Video::DCAEVNormMPS.checked_count("activation-prefix target", target_shape)
    target_cache_values = ML::Video::DCAEVNormMPS.checked_count("activation-prefix target cache",
      [config.batches, TARGET_CHANNELS, 1, config.height * 8, config.width * 8])
    scratch_bytes = activation_values.to_i64 * 2 * 3
    saved_bytes = activation_values.to_i64 * 2 * TILE_COUNT
    target_cache_bytes = target_cache_values.to_i64 * 2 * TILE_COUNT
    cache_bytes = prefix_cache_specs(config).sum do |spec|
      ML::Video::DCAEVNormMPS.checked_count("activation-prefix cache",
        [config.batches, spec[1], 1, spec[2], spec[3]]).to_i64 * 2
    end
    mean_rows = ML::Video::DCAEVNormMPS.checked_count("activation-prefix RMS rows",
      [config.batches, 1, TARGET_FRAMES, config.height * 4, config.width * 4])
    mean_bytes = mean_rows.to_i64 * 2 * 2
    add_bytes("activation-prefix workspace",
      add_bytes("activation-prefix buffers", scratch_bytes, saved_bytes),
      add_bytes("activation-prefix caches", cache_bytes,
        add_bytes("activation-prefix auxiliary buffers", target_cache_bytes,
          add_bytes("activation-prefix means and latent", mean_bytes, config.input_bytes))))
  end

  private def allocate_workspace(config : ML::Video::DCAEVDecoderConfig,
                                 reference : ML::Video::CheckpointFile,
                                 latent : ML::Video::CheckpointTensor,
                                 target_values : Int32) : Nil
    3.times do |index|
      @scratch << allocate_bf16("scratch.#{index}", target_values)
    end
    TILE_COUNT.times do |tile|
      @activations << allocate_bf16("activation.#{tile}", target_values)
      @old_caches << allocate_bf16("old_cache.#{tile}", @target_cache_count)
    end
    cache_specs = self.class.prefix_cache_specs(config)
    cache_specs.each do |spec|
      name, channels, height, width = spec
      values = ML::Video::DCAEVNormMPS.checked_count("activation-prefix cache",
        [config.batches, channels, 1, height, width])
      @internal_caches[name] = allocate_bf16("cache.#{name}", values)
    end
    mean_rows = ML::Video::DCAEVNormMPS.checked_count("activation-prefix RMS rows",
      [config.batches, 1, TARGET_FRAMES, config.height * 4, config.width * 4])
    allocate_bf16("mean_a", mean_rows)
    allocate_bf16("mean_b", mean_rows)

    input_values = config.input_values
    @input_latent = allocate_bf16("input.latent", input_values)
    reference.copy_raw(latent, Bytes.new(@input_latent.not_nil!.contents.as(UInt8*), latent.data_bytes.to_i32))
    unless @input_latent.not_nil!.size == config.input_bytes
      raise "Activation-prefix latent destination has the wrong size"
    end
  end

  private def build_plans(config : ML::Video::DCAEVDecoderConfig) : Nil
    TILE_COUNT.times do |tile|
      plan = ML::Video::DCAEVDecoderPlan.new(@arena.buffers.values)
      @plans << plan
      h, w, t, slot = config.height, config.width, 10, 0
      add_gather(plan, config, tile)
      add_conv(plan, config, tile, "decoder.project_in", 32, 1024, t, h, w, true,
        @scratch[0], @scratch[1])
      slot = 1

      6.downto(3) do |stage|
        channels = ML::Video::DCAEVDecoder::WIDTHS[stage]
        if stage < 6
          spatial_factor = stage == 5 ? 1 : 2
          temporal_factor = stage == 5 ? 4 : 1
          original = @scratch[slot]
          expanded = @scratch[(slot + 1) % 3]
          next_slot = (slot + 2) % 3
          add_conv(plan, config, tile, upsample_prefix(stage),
            ML::Video::DCAEVDecoder::WIDTHS[stage + 1], channels * 4, t, h, w, true,
            original, expanded)
          add_shuffle(plan, config, expanded, original, @scratch[next_slot],
            ML::Video::DCAEVDecoder::WIDTHS[stage + 1], channels, t, h, w,
            spatial_factor, temporal_factor, true, "tile#{tile}/stage_#{stage}.upsample.shuffle")
          slot = next_slot
          t *= temporal_factor
          h *= spatial_factor
          w *= spatial_factor
        end
        ML::Video::DCAEVDecoder::DEPTHS[stage].times do |block|
          prefix = residual_prefix(stage, block)
          original = @scratch[slot]
          temp = @scratch[(slot + 2) % 3]
          next_slot = (slot + 1) % 3
          result = @scratch[next_slot]
          add_conv(plan, config, tile, "#{prefix}.conv1", channels, channels, t, h, w, true,
            original, temp)
          plan.add(ML::Video::DCAEVNormMPS::SiLUPlan.new(
            [config.batches, channels, t, h, w], temp, result),
            "tile#{tile}/#{prefix}.silu", "stage_#{stage}")
          add_conv(plan, config, tile, "#{prefix}.conv2", channels, channels, t, h, w, false,
            result, temp)
          add_norm(plan, config, tile, "#{prefix}.conv2.norm", channels, t, h, w,
            temp, result)
          add_residual(plan, result, original,
            ML::Video::DCAEVNormMPS.checked_count("activation-prefix residual",
              [config.batches, channels, t, h, w]), "tile#{tile}/#{prefix}.residual_add", "stage_#{stage}")
          slot = next_slot
        end
      end

      # Stage 2's op_list.0 upsample is the final prefix operation. The next
      # operation would be the target residual block's conv1, so save the
      # shuffled [1,512,40,72,128] input now and stop.
      stage = 2
      channels = ML::Video::DCAEVDecoder::WIDTHS[stage]
      original = @scratch[slot]
      expanded = @scratch[(slot + 1) % 3]
      next_slot = (slot + 2) % 3
      add_conv(plan, config, tile, upsample_prefix(stage),
        ML::Video::DCAEVDecoder::WIDTHS[stage + 1], channels * 4, t, h, w, true,
        original, expanded)
      add_shuffle(plan, config, expanded, original, @scratch[next_slot],
        ML::Video::DCAEVDecoder::WIDTHS[stage + 1], channels, t, h, w, 2, 1,
        true, "tile#{tile}/stage_2.upsample.shuffle")
      output_count = ML::Video::DCAEVNormMPS.checked_count("activation-prefix saved activation", @target_shape)
      pipeline = ML::Video::DCAEVEncoder.pipeline("dcae_v_copy_bf16")
      saved = @activations[tile]
      source = @scratch[next_slot]
      plan.add("tile#{tile}/capture.stage_2.pre_conv1", "copy", "stage_2") do |chain|
        encoder = chain.encoder
        encoder.set_pipeline(pipeline).set_buffer(source, 0)
          .set_buffer(saved, 1, ML::Metal::BufferAccess::Write)
          .set_value(output_count.to_u32, 2).dispatch_1d(output_count)
      end
    end
  end

  private def add_conv(plan : ML::Video::DCAEVDecoderPlan,
                       config : ML::Video::DCAEVDecoderConfig, tile : Int32,
                       prefix : String, cin : Int32, cout : Int32,
                       frames : Int32, height : Int32, width : Int32,
                       bias : Bool, input : ML::MetalBuffer, output : ML::MetalBuffer) : Nil
    cache = @internal_caches[prefix]
    key = [config.batches, cin, cout, frames, height, width, tile > 0, bias].map(&.to_s).join("|")
    template = plan.template(key) do
      ML::Video::DCAEVConvMPS::Template.new(config.batches, cin, cout, cout,
        frames, frames, height, width, 0, false, true, tile > 0, bias, 1, 1, false)
    end
    weights = @arena["#{prefix}.conv.weight"]
    weights_bias = bias ? @arena["#{prefix}.conv.bias"] : nil
    plan.add(template.bind(input, tile > 0 ? cache : nil, weights, weights_bias, output),
      "tile#{tile}/#{prefix}.conv3d", prefix.ends_with?(".conv") ? "upsample_conv3d" : "residual_conv3d",
      prefix.starts_with?("decoder.project_in") ? "project_in" : prefix.split('.')[2])

    cache_pipeline = ML::Video::DCAEVEncoder.pipeline("dcae_v_cache_last_frame_bf16")
    cache_count = ML::Video::DCAEVNormMPS.checked_count("activation-prefix cache",
      [config.batches, cin, 1, height, width])
    plan.add("tile#{tile}/#{prefix}.cache_last_frame", "cache_store",
      prefix.starts_with?("decoder.project_in") ? "project_in" : prefix.split('.')[2]) do |chain|
      encoder = chain.encoder
      encoder.set_pipeline(cache_pipeline).set_buffer(input, 0)
        .set_buffer(cache, 1, ML::Metal::BufferAccess::Write)
        .set_value(cin.to_u32, 2).set_value(frames.to_u32, 3)
        .set_value(frames.to_u32, 4).set_value(0_u32, 5)
        .set_value(height.to_u32, 6).set_value(width.to_u32, 7)
        .set_value(config.batches.to_u32, 8).dispatch_1d(cache_count)
    end
  end

  private def add_norm(plan : ML::Video::DCAEVDecoderPlan,
                       config : ML::Video::DCAEVDecoderConfig, tile : Int32,
                       prefix : String, channels : Int32, frames : Int32,
                       height : Int32, width : Int32,
                       input : ML::MetalBuffer, output : ML::MetalBuffer) : Nil
    shape = [config.batches, channels, frames, height, width]
    plan.add(ML::Video::DCAEVNormMPS::RMSNormPlan.new(shape, input, output,
      @arena["dcae_v_activation_prefix.mean_a"],
      @arena["dcae_v_activation_prefix.mean_b"],
      @arena["#{prefix}.weight"], @arena["#{prefix}.bias"]),
      "tile#{tile}/#{prefix}.rmsnorm", prefix.split('.')[2])
  end

  private def add_residual(plan : ML::Video::DCAEVDecoderPlan,
                           result : ML::MetalBuffer, original : ML::MetalBuffer,
                           count : Int32, name : String, stage : String) : Nil
    pipeline = ML::Video::DCAEVEncoder.pipeline("dcae_v_residual_add_bf16_inplace")
    plan.add(name, "residual_add", stage) do |chain|
      encoder = chain.encoder
      encoder.set_pipeline(pipeline)
        .set_buffer(result, 0, ML::Metal::BufferAccess::ReadWrite)
        .set_buffer(original, 1).set_value(count.to_u32, 2).dispatch_1d(count)
      chain.pause!
      chain.continue_async!
      chain.resume!
    end
  end

  private def add_gather(plan : ML::Video::DCAEVDecoderPlan,
                         config : ML::Video::DCAEVDecoderConfig, tile : Int32) : Nil
    pipeline = ML::Video::DCAEVDecoder.pipeline("dcae_v_decoder_gather_latent_bf16")
    output = @scratch[0]
    input = @input_latent.not_nil!
    count = ML::Video::DCAEVNormMPS.checked_count("activation-prefix latent tile",
      [config.batches, 32, 10, config.height, config.width])
    offset = config.tile_offset(tile)
    plan.add("tile#{tile}/gather_latent", "gather", "input") do |chain|
      encoder = chain.encoder
      encoder.set_pipeline(pipeline).set_buffer(input, 0)
        .set_buffer(output, 1, ML::Metal::BufferAccess::Write)
        .set_value(config.frames.to_u32, 2).set_value(10_u32, 3)
        .set_value(config.height.to_u32, 4).set_value(config.width.to_u32, 5)
        .set_value(offset.to_u32, 6).set_value(count.to_u32, 7)
        .dispatch_1d(count)
    end
  end

  private def add_shuffle(plan : ML::Video::DCAEVDecoderPlan,
                          config : ML::Video::DCAEVDecoderConfig,
                          expanded : ML::MetalBuffer, shortcut : ML::MetalBuffer,
                          output : ML::MetalBuffer, cin : Int32, cout : Int32,
                          frames : Int32, height : Int32, width : Int32,
                          spatial_factor : Int32, temporal_factor : Int32,
                          use_shortcut : Bool, name : String) : Nil
    pipeline = ML::Video::DCAEVDecoder.pipeline("dcae_v_decoder_pixel_shuffle_bf16")
    expanded_channels = cout * spatial_factor * spatial_factor * temporal_factor
    unless !use_shortcut || expanded_channels % cin == 0
      raise ArgumentError.new("Invalid decoder shortcut channel expansion")
    end
    repeat = use_shortcut ? expanded_channels // cin : 1
    count = ML::Video::DCAEVNormMPS.checked_count("activation-prefix shuffled tile",
      [config.batches, cout, frames * temporal_factor,
       height * spatial_factor, width * spatial_factor])
    plan.add(name, "shuffle", name.split('.')[1]) do |chain|
      encoder = chain.encoder
      encoder.set_pipeline(pipeline).set_buffer(expanded, 0).set_buffer(shortcut, 1)
        .set_buffer(output, 2, ML::Metal::BufferAccess::Write)
        .set_value(expanded_channels.to_u32, 3).set_value(cin.to_u32, 4)
        .set_value(cout.to_u32, 5).set_value(frames.to_u32, 6)
        .set_value(height.to_u32, 7).set_value(width.to_u32, 8)
        .set_value(spatial_factor.to_u32, 9).set_value(temporal_factor.to_u32, 10)
        .set_value(repeat.to_u32, 11).set_value(use_shortcut ? 1_u32 : 0_u32, 12)
        .set_value(count.to_u32, 13).dispatch_1d(count)
    end
  end

  private def allocate_bf16(name : String, values : Int32) : ML::MetalBuffer
    @arena.alloc_weight("dcae_v_activation_prefix.#{name}", values, ML::Video::WeightDType::BF16)
  end

  private def validate_tile!(tile : Int32) : Nil
    raise ArgumentError.new("Activation-prefix tile index is out of range") unless 0 <= tile < TILE_COUNT
  end

  private def cleanup : Nil
    @plans.each(&.release)
    @plans.clear
    @prepared_weights.try(&.release)
    @prepared_weights = nil
    @arena.release
  end

  private def residual_prefix(stage : Int32, block : Int32) : String
    offset = stage == 6 ? 0 : 1
    "decoder.stages.#{stage}.op_list.#{block + offset}.main"
  end

  private def upsample_prefix(stage : Int32) : String
    "decoder.stages.#{stage}.op_list.0.main.conv"
  end

  def self.prefix_cache_specs(config : ML::Video::DCAEVDecoderConfig) : Array(CacheSpec)
    specs = [] of CacheSpec
    specs << {"decoder.project_in", 32, config.height, config.width}
    h, w = config.height, config.width
    6.downto(2) do |stage|
      if stage < 6
        specs << {upsample_prefix(stage), ML::Video::DCAEVDecoder::WIDTHS[stage + 1], h, w}
        if stage < 5
          h *= 2
          w *= 2
        end
      end
      if stage >= 3
        ML::Video::DCAEVDecoder::DEPTHS[stage].times do |block|
          prefix = residual_prefix(stage, block)
          specs << {"#{prefix}.conv1", ML::Video::DCAEVDecoder::WIDTHS[stage], h, w}
          specs << {"#{prefix}.conv2", ML::Video::DCAEVDecoder::WIDTHS[stage], h, w}
        end
      end
    end
    specs
  end

  private def self.residual_prefix(stage : Int32, block : Int32) : String
    offset = stage == 6 ? 0 : 1
    "decoder.stages.#{stage}.op_list.#{block + offset}.main"
  end

  private def self.upsample_prefix(stage : Int32) : String
    "decoder.stages.#{stage}.op_list.0.main.conv"
  end

  def self.add_bytes(label : String, total : Int64, additional : Int64) : Int64
    unless total >= 0 && additional >= 0 && additional <= Int64::MAX - total
      raise ArgumentError.new("DCAE-V #{label} byte count overflows")
    end
    total + additional
  end
end
