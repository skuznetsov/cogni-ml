# PixelUMM generation input stem and output assembly for the native Hy-Nomas
# first-preview path. This bridge owns no denoiser, scheduler, or decoder block.
require "../core/buffer"
require "./metal_mps_command_chain"
require "./metal_ops"
require "./metal_linear_mps"
require "./pixelumm_decoder_block_mps"

module ML::Video::PixelUMMGeneration
  SOURCE = {{ read_file("#{__DIR__}/kernels/pixelumm_generation.metal") }}

  LATENT_FRAMES               = 30
  LATENT_HEIGHT               =  9
  LATENT_WIDTH                = 16
  LATENT_CHANNELS             = 32
  WINDOW_LATENTS              = 10
  WINDOW_OVERLAP_LATENTS      =  5
  WINDOW_STARTS               = [0_i32, 5_i32, 10_i32, 15_i32, 20_i32]
  PIXEL_FRAMES                =  120
  PIXEL_HEIGHT                =  288
  PIXEL_WIDTH                 =  512
  PIXEL_CHANNELS              =    3
  TUBELET_T                   =    4
  TUBELET_H                   =   16
  TUBELET_W                   =   16
  TOKEN_HEIGHT                =   18
  TOKEN_WIDTH                 =   32
  HIDDEN                      = 4096
  LATENT_TOKEN_COUNT          = WINDOW_LATENTS * LATENT_HEIGHT * LATENT_WIDTH
  TOKEN_COUNT                 = WINDOW_LATENTS * TOKEN_HEIGHT * TOKEN_WIDTH
  EMBED_INPUT_DIM             = TUBELET_T * TUBELET_H * TUBELET_W * PIXEL_CHANNELS
  CONDITION_OUTPUT_DIM        = 4 * HIDDEN
  WINDOW_PIXEL_FRAMES         = WINDOW_LATENTS * TUBELET_T
  WINDOW_OVERLAP_PIXEL_FRAMES = WINDOW_OVERLAP_LATENTS * TUBELET_T

  def self.pipeline(name : String) : ML::Metal::ComputePipeline
    ML::Metal::PipelineCache.get(name) { ML::Metal::ComputePipeline.new(name, SOURCE) }
  end

  # Source projection tensors arrive as authenticated F32 checkpoint buffers.
  # StemPlan quantizes each once to BF16 on the active command chain, matching
  # the upstream autocast conversion; MPS then consumes its F32 expansion.
  record StemWeights,
    embed_weight : ML::MetalBuffer,
    embed_bias : ML::MetalBuffer,
    cond_weight : ML::MetalBuffer,
    cond_bias : ML::MetalBuffer

  struct Config
    def initialize
      validate!
    end

    def validate! : Nil
      block_config.validate!
      unless block_config.layout.local_attention? &&
             block_config.layout.tile_count == 12 &&
             output_elements <= Int32::MAX
        raise ArgumentError.new("PixelUMM preview geometry does not use the admitted local decoder layout")
      end
      unless WINDOW_STARTS.size == 5 &&
             WINDOW_STARTS.each_with_index.all? { |start, i| start == i * WINDOW_OVERLAP_LATENTS }
        raise ArgumentError.new("PixelUMM preview window schedule is inconsistent")
      end
    end

    def decoder_config : PixelUMMDecoderBlockMPS::Config
      PixelUMMDecoderBlockMPS::Config.new(
        temporal: WINDOW_LATENTS, height: TOKEN_HEIGHT, width: TOKEN_WIDTH)
    end

    def output_shape : Array(Int32)
      [1_i32, PIXEL_CHANNELS, PIXEL_FRAMES, PIXEL_HEIGHT, PIXEL_WIDTH]
    end

    def window_output_shape : Array(Int32)
      [1_i32, PIXEL_CHANNELS, WINDOW_PIXEL_FRAMES, PIXEL_HEIGHT, PIXEL_WIDTH]
    end

    def output_elements : Int64
      PIXEL_CHANNELS.to_i64 * PIXEL_FRAMES * PIXEL_HEIGHT * PIXEL_WIDTH
    end

    def window_output_elements : Int64
      PIXEL_CHANNELS.to_i64 * WINDOW_PIXEL_FRAMES * PIXEL_HEIGHT * PIXEL_WIDTH
    end

    def block_input_bytes : Int64
      TOKEN_COUNT.to_i64 * HIDDEN * 2
    end

    # Pure CPU admission inventory for explicit StemPlan-owned allocations.
    # MPSNDArray descriptor/temporary storage and borrowed source inputs/weights
    # are not included.
    def stem_named_bytes_by_name : Hash(String, Int64)
      PixelUMMGeneration.stem_named_bytes_by_name(self)
    end

    def assembly_named_bytes_by_name : Hash(String, Int64)
      PixelUMMGeneration.assembly_named_bytes_by_name(self)
    end

    private def block_config : PixelUMMDecoderBlockMPS::Config
      decoder_config
    end
  end

  # Pure preflight helpers. Call before creating Metal buffers or MPS plans.
  def self.stem_named_bytes_by_name(config : Config) : Hash(String, Int64)
    config.validate!
    weight_stage = EMBED_INPUT_DIM.to_i64 * HIDDEN
    condition_weight = LATENT_CHANNELS.to_i64 * CONDITION_OUTPUT_DIM
    weight_stage = Math.max(weight_stage, condition_weight)
    {
      "patch_f32"              => TOKEN_COUNT.to_i64 * EMBED_INPUT_DIM * 4,
      "condition_f32"          => LATENT_TOKEN_COUNT.to_i64 * LATENT_CHANNELS * 4,
      "projection_f32"         => TOKEN_COUNT.to_i64 * HIDDEN * 4,
      "output_bf16"            => TOKEN_COUNT.to_i64 * HIDDEN * 2,
      "tile_major_to_original" => TOKEN_COUNT.to_i64 * sizeof(Int32),
      "mps_weight_stage_f32"   => weight_stage * 4,
      "embed_weight_bf16"      => HIDDEN.to_i64 * EMBED_INPUT_DIM * 2,
      "embed_bias_bf16"        => HIDDEN.to_i64 * 2,
      "cond_weight_bf16"       => condition_weight * 2,
      "cond_bias_bf16"         => CONDITION_OUTPUT_DIM.to_i64 * 2,
    }
  end

  def self.stem_named_bytes(config : Config) : Int64
    stem_named_bytes_by_name(config).values.sum
  end

  def self.assembly_named_bytes_by_name(config : Config) : Hash(String, Int64)
    config.validate!
    {
      "output_f32"         => config.output_elements * 4,
      "blend_weights_bf16" => WINDOW_OVERLAP_PIXEL_FRAMES.to_i64 * 2,
    }
  end

  def self.assembly_named_bytes(config : Config) : Int64
    assembly_named_bytes_by_name(config).values.sum
  end

  class StemPlan
    getter config : Config
    getter output_bf16 : ML::MetalBuffer
    getter named_bytes_by_name : Hash(String, Int64)
    getter named_bytes : Int64
    getter next_window_index : Int32 = 0

    @arena : Arena
    @latent : ML::MetalBuffer
    @noise : ML::MetalBuffer
    @weights : StemWeights
    @embed_plan : WanT5MPSLinearPlan? = nil
    @condition_plan : WanT5MPSLinearPlan? = nil
    @chain : ML::Metal::MPSCommandChain? = nil
    @weights_rounded = false
    @failed = false
    @released = false

    def initialize(@config : Config, *, latent : ML::MetalBuffer,
                   noise : ML::MetalBuffer, weights : StemWeights)
      @config.validate!
      @latent = latent
      @noise = noise
      @weights = weights
      validate_sources!
      @named_bytes_by_name = PixelUMMGeneration.stem_named_bytes_by_name(@config)
      @named_bytes = @named_bytes_by_name.values.sum
      @arena = Arena.new
      begin
        allocate_buffers!
        prepare_plans!
      rescue ex
        @embed_plan.try(&.release)
        @condition_plan.try(&.release)
        @arena.release
        @released = true
        raise ex
      end
      @output_bf16 = @arena["output_bf16"]
    end

    # Append one stem window to the caller-owned serial chain. The same plan
    # supports exactly the five overlapping starts for a complete first preview.
    def encode(chain : ML::Metal::MPSCommandChain,
               start_latent_frame : Int32) : ML::MetalBuffer
      raise ArgumentError.new("PixelUMM stem plan has been released") if @released
      raise ArgumentError.new("PixelUMM stem plan cannot continue after an encode error") if @failed
      expected_start = WINDOW_STARTS[@next_window_index]?
      unless expected_start && start_latent_frame == expected_start
        raise ArgumentError.new("PixelUMM stem windows must use starts 0, 5, 10, 15, 20 exactly once")
      end
      bind_chain!(chain)
      e = chain.encoder
      begin
        encode_weight_rounding(e) unless @weights_rounded
        encode_patchify(e, start_latent_frame)
        embed_plan = @embed_plan || raise ArgumentError.new("PixelUMM embed MPS plan is unavailable")
        MetalLinearMPS.project(e, @arena["patch_f32"], @arena["embed_weight_bf16"],
          @arena["embed_bias_bf16"], @arena["projection_f32"], TOKEN_COUNT,
          EMBED_INPUT_DIM, HIDDEN, WeightDType::BF16, WeightDType::BF16,
          embed_plan, @arena["mps_weight_stage_f32"])
        encode_store_embed(e)
        encode_gather_condition(e, start_latent_frame)
        condition_plan = @condition_plan || raise ArgumentError.new("PixelUMM condition MPS plan is unavailable")
        MetalLinearMPS.project(e, @arena["condition_f32"], @arena["cond_weight_bf16"],
          @arena["cond_bias_bf16"], @arena["projection_f32"], LATENT_TOKEN_COUNT,
          LATENT_CHANNELS, CONDITION_OUTPUT_DIM, WeightDType::BF16, WeightDType::BF16,
          condition_plan, @arena["mps_weight_stage_f32"])
        encode_add_condition(e)
        @weights_rounded = true
        @next_window_index += 1
        @output_bf16
      rescue ex
        @failed = true
        raise ex
      end
    end

    # MPS descriptors and all scratch are kept alive through one caller-owned
    # terminal wait. External inputs and checkpoint buffers remain caller-owned.
    def release_after_terminal_wait! : Nil
      return if @released
      if chain = @chain
        raise ArgumentError.new("PixelUMM stem resources require the terminal chain wait") unless chain.completed?
      end
      @embed_plan.try(&.release)
      @condition_plan.try(&.release)
      @arena.release
      @released = true
    end

    # Match the owning generation runner's lifecycle API. This never waits;
    # callers must first complete the chain that used this plan.
    def release : Nil
      release_after_terminal_wait!
    end

    private def validate_sources! : Nil
      MetalOps.check_storage(@latent,
        LATENT_CHANNELS.to_i64 * LATENT_FRAMES * LATENT_HEIGHT * LATENT_WIDTH,
        WeightDType::F32)
      MetalOps.check_storage(@noise, @config.output_elements, WeightDType::F32)
      {
        {@weights.embed_weight, HIDDEN.to_i64 * EMBED_INPUT_DIM},
        {@weights.embed_bias, HIDDEN.to_i64},
        {@weights.cond_weight, LATENT_CHANNELS.to_i64 * CONDITION_OUTPUT_DIM},
        {@weights.cond_bias, CONDITION_OUTPUT_DIM.to_i64},
      }.each do |buffer, count|
        unless buffer.size == count * sizeof(Float32)
          raise ArgumentError.new("PixelUMM source stem weights must be exact-size F32 buffers")
        end
        MetalOps.check_storage(buffer, count, WeightDType::F32)
      end
      MetalOps.distinct(@latent, @noise, @weights.embed_weight, @weights.embed_bias,
        @weights.cond_weight, @weights.cond_bias)
    end

    private def allocate_buffers! : Nil
      @arena.alloc("patch_f32", TOKEN_COUNT * EMBED_INPUT_DIM)
      @arena.alloc("condition_f32", LATENT_TOKEN_COUNT * LATENT_CHANNELS)
      @arena.alloc("projection_f32", TOKEN_COUNT * HIDDEN)
      @arena.alloc_weight("output_bf16", TOKEN_COUNT * HIDDEN, WeightDType::BF16)
      permutation = @config.decoder_config.layout.permutation
      raise ArgumentError.new("PixelUMM preview tile permutation extent mismatch") unless permutation.size == TOKEN_COUNT
      permutation_buffer = @arena.alloc("tile_major_to_original", TOKEN_COUNT)
      permutation_buffer.write_bytes(permutation.to_unsafe.as(Pointer(UInt8)),
        permutation.size * sizeof(Int32))
      max_weight = Math.max(EMBED_INPUT_DIM.to_i64 * HIDDEN,
        LATENT_CHANNELS.to_i64 * CONDITION_OUTPUT_DIM)
      @arena.alloc("mps_weight_stage_f32", max_weight.to_i32)
      @arena.alloc_weight("embed_weight_bf16", (HIDDEN * EMBED_INPUT_DIM), WeightDType::BF16)
      @arena.alloc_weight("embed_bias_bf16", HIDDEN, WeightDType::BF16)
      @arena.alloc_weight("cond_weight_bf16", (LATENT_CHANNELS * CONDITION_OUTPUT_DIM), WeightDType::BF16)
      @arena.alloc_weight("cond_bias_bf16", CONDITION_OUTPUT_DIM, WeightDType::BF16)
    end

    private def prepare_plans! : Nil
      MetalLinearMPS.prepare!
      stage = @arena["mps_weight_stage_f32"]
      projection = @arena["projection_f32"]
      embed_config = WanT5Config.new(dim: HIDDEN, ffn_dim: HIDDEN, tokens: TOKEN_COUNT)
      condition_config = WanT5Config.new(dim: HIDDEN, ffn_dim: CONDITION_OUTPUT_DIM,
        tokens: LATENT_TOKEN_COUNT)
      @embed_plan = WanT5MPSLinearPlan.new(embed_config, EMBED_INPUT_DIM, HIDDEN,
        @arena["patch_f32"], stage, projection)
      @condition_plan = WanT5MPSLinearPlan.new(condition_config, LATENT_CHANNELS,
        CONDITION_OUTPUT_DIM, @arena["condition_f32"], stage, projection)
    end

    private def bind_chain!(chain : ML::Metal::MPSCommandChain) : Nil
      if existing = @chain
        unless existing.same?(chain)
          raise ArgumentError.new("PixelUMM stem windows must share one command chain")
        end
      else
        @chain = chain
      end
    end

    private def encode_weight_rounding(e : ML::Metal::ComputeEncoder) : Nil
      encode_rounding(e, @weights.embed_weight, @arena["embed_weight_bf16"], HIDDEN * EMBED_INPUT_DIM)
      encode_rounding(e, @weights.embed_bias, @arena["embed_bias_bf16"], HIDDEN)
      encode_rounding(e, @weights.cond_weight, @arena["cond_weight_bf16"],
        LATENT_CHANNELS * CONDITION_OUTPUT_DIM)
      encode_rounding(e, @weights.cond_bias, @arena["cond_bias_bf16"], CONDITION_OUTPUT_DIM)
    end

    private def encode_rounding(e : ML::Metal::ComputeEncoder,
                                source : ML::MetalBuffer, destination : ML::MetalBuffer,
                                count : Int32) : Nil
      e.set_pipeline(PixelUMMGeneration.pipeline("pixelumm_generation_round_f32_to_bf16"))
      e.set_buffer(source, 0).set_buffer(destination, 1, ML::Metal::BufferAccess::Write)
      e.set_value(count.to_u32, 2)
      e.dispatch_1d(count)
    end

    private def encode_patchify(e : ML::Metal::ComputeEncoder,
                                start_latent_frame : Int32) : Nil
      count = TOKEN_COUNT * EMBED_INPUT_DIM
      e.set_pipeline(PixelUMMGeneration.pipeline("pixelumm_generation_patchify"))
      e.set_buffer(@noise, 0).set_buffer(@arena["patch_f32"], 1, ML::Metal::BufferAccess::Write)
      e.set_value((start_latent_frame * TUBELET_T).to_u32, 2)
      e.set_value(count.to_u32, 3)
      e.dispatch_1d(count)
    end

    private def encode_store_embed(e : ML::Metal::ComputeEncoder) : Nil
      count = TOKEN_COUNT * HIDDEN
      e.set_pipeline(PixelUMMGeneration.pipeline("pixelumm_generation_store_embed"))
      e.set_buffer(@arena["projection_f32"], 0)
        .set_buffer(@arena["tile_major_to_original"], 1)
        .set_buffer(@output_bf16, 2, ML::Metal::BufferAccess::Write)
      e.set_value(count.to_u32, 3)
      e.dispatch_1d(count)
    end

    private def encode_gather_condition(e : ML::Metal::ComputeEncoder,
                                        start_latent_frame : Int32) : Nil
      count = LATENT_TOKEN_COUNT * LATENT_CHANNELS
      e.set_pipeline(PixelUMMGeneration.pipeline("pixelumm_generation_gather_condition"))
      e.set_buffer(@latent, 0).set_buffer(@arena["condition_f32"], 1,
        ML::Metal::BufferAccess::Write)
      e.set_value(start_latent_frame.to_u32, 2)
      e.set_value(count.to_u32, 3)
      e.dispatch_1d(count)
    end

    private def encode_add_condition(e : ML::Metal::ComputeEncoder) : Nil
      count = TOKEN_COUNT * HIDDEN
      e.set_pipeline(PixelUMMGeneration.pipeline("pixelumm_generation_add_condition"))
      e.set_buffer(@arena["projection_f32"], 0)
        .set_buffer(@arena["tile_major_to_original"], 1)
        .set_buffer(@output_bf16, 2, ML::Metal::BufferAccess::ReadWrite)
      e.set_value(count.to_u32, 3)
      e.dispatch_1d(count)
    end
  end

  class AssemblyPlan
    getter config : Config
    getter output_f32 : ML::MetalBuffer
    getter named_bytes_by_name : Hash(String, Int64)
    getter named_bytes : Int64
    getter merged_windows : Int32 = 0
    getter? finalized : Bool = false

    @arena : Arena
    @chain : ML::Metal::MPSCommandChain? = nil
    @failed = false
    @released = false
    @blend_weights_encoded = false

    def initialize(@config : Config = Config.new)
      @config.validate!
      @named_bytes_by_name = PixelUMMGeneration.assembly_named_bytes_by_name(@config)
      @named_bytes = @named_bytes_by_name.values.sum
      @arena = Arena.new
      @arena.alloc("output_f32", @config.output_elements.to_i32)
      @arena.alloc_weight("blend_weights_bf16", WINDOW_OVERLAP_PIXEL_FRAMES,
        WeightDType::BF16)
      @output_f32 = @arena["output_f32"]
    end

    # Merge a head's BF16 [1,3,40,288,512] prediction at one upstream latent
    # start. Five appends create a BF16-equivalent F32 [1,3,120,288,512] clip.
    def encode_merge(chain : ML::Metal::MPSCommandChain,
                     *, window_bf16 : ML::MetalBuffer,
                     start_latent_frame : Int32) : ML::MetalBuffer
      raise ArgumentError.new("PixelUMM assembly plan has been released") if @released
      raise ArgumentError.new("PixelUMM assembly plan cannot continue after an encode error") if @failed
      raise ArgumentError.new("PixelUMM assembly was already finalized") if finalized?
      expected_start = WINDOW_STARTS[@merged_windows]?
      unless expected_start && start_latent_frame == expected_start
        raise ArgumentError.new("PixelUMM assembly windows must use starts 0, 5, 10, 15, 20 exactly once")
      end
      MetalOps.check_storage(window_bf16, @config.window_output_elements, WeightDType::BF16)
      MetalOps.distinct(@output_f32, window_bf16, @arena["blend_weights_bf16"])
      bind_chain!(chain)
      e = chain.encoder
      begin
        unless @blend_weights_encoded
          e.set_pipeline(PixelUMMGeneration.pipeline("pixelumm_generation_blend_weights"))
          e.set_buffer(@arena["blend_weights_bf16"], 0, ML::Metal::BufferAccess::Write)
          e.dispatch_1d(WINDOW_OVERLAP_PIXEL_FRAMES)
          @blend_weights_encoded = true
        end
        e.set_pipeline(PixelUMMGeneration.pipeline("pixelumm_generation_merge_window"))
        e.set_buffer(window_bf16, 0).set_buffer(@output_f32, 1, ML::Metal::BufferAccess::ReadWrite)
          .set_buffer(@arena["blend_weights_bf16"], 2)
        e.set_value((start_latent_frame * TUBELET_T).to_u32, 3)
        e.set_value(@config.window_output_elements.to_u32, 4)
        e.dispatch_1d(@config.window_output_elements.to_i32)
        @merged_windows += 1
        @output_f32
      rescue ex
        @failed = true
        raise ex
      end
    end

    # Convert the fully merged [-1,1] decoder output in place to display-range
    # F32 [0,1] pixels. The caller can export/read it after the terminal wait.
    def encode_finalize(chain : ML::Metal::MPSCommandChain) : ML::MetalBuffer
      raise ArgumentError.new("PixelUMM assembly plan has been released") if @released
      raise ArgumentError.new("PixelUMM assembly plan cannot continue after an encode error") if @failed
      raise ArgumentError.new("PixelUMM assembly requires all five windows") unless @merged_windows == WINDOW_STARTS.size
      raise ArgumentError.new("PixelUMM assembly was already finalized") if finalized?
      bind_chain!(chain)
      e = chain.encoder
      begin
        e.set_pipeline(PixelUMMGeneration.pipeline("pixelumm_generation_finalize_display_range"))
        e.set_buffer(@output_f32, 0, ML::Metal::BufferAccess::ReadWrite)
        e.set_value(@config.output_elements.to_u32, 1)
        e.dispatch_1d(@config.output_elements.to_i32)
        @finalized = true
        @output_f32
      rescue ex
        @failed = true
        raise ex
      end
    end

    def release_after_terminal_wait! : Nil
      return if @released
      if chain = @chain
        raise ArgumentError.new("PixelUMM assembly resources require the terminal chain wait") unless chain.completed?
      end
      @arena.release
      @released = true
    end

    # Match the owning generation runner's lifecycle API. This never waits;
    # callers must first complete the chain that used this plan.
    def release : Nil
      release_after_terminal_wait!
    end

    private def bind_chain!(chain : ML::Metal::MPSCommandChain) : Nil
      if existing = @chain
        unless existing.same?(chain)
          raise ArgumentError.new("PixelUMM assembly windows must share one command chain")
        end
      else
        @chain = chain
      end
    end
  end
end
