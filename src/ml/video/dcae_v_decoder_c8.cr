require "./metal_ops"
require "./metal_mps_command_chain"

module ML::Video::DCAEVDecoderC8
  SOURCE = {{ read_file("#{__DIR__}/../metal/kernels/dcae_v/dcae_v_decoder_c8.metal") }}

  CHANNELS = 512
  FRAMES   =  40
  HEIGHT   =  72
  WIDTH    = 128

  STAGE1_CHANNELS = 256
  STAGE1_HEIGHT   = 144
  STAGE1_WIDTH    = 256

  INPUT_PACK                = "dcae_v_decoder.c8.input_pack"
  WEIGHT_PACK               = "dcae_v_decoder.c8.weight_pack"
  INPUT_PACK_VALUES         = CHANNELS * (FRAMES + 2) * HEIGHT * WIDTH
  WEIGHT_PACK_VALUES        = CHANNELS * CHANNELS * 27
  STAGE1_INPUT_PACK_VALUES  = STAGE1_CHANNELS * (FRAMES + 2) * STAGE1_HEIGHT * STAGE1_WIDTH
  STAGE1_WEIGHT_PACK_VALUES = STAGE1_CHANNELS * STAGE1_CHANNELS * 27
  SHARED_INPUT_PACK_VALUES  = STAGE1_INPUT_PACK_VALUES
  SHARED_WEIGHT_PACK_VALUES = WEIGHT_PACK_VALUES

  def self.eligible?(batches : Int32, cin : Int32, cout : Int32,
                     t : Int32, h : Int32, w : Int32,
                     *, stage : Int32 = 2) : Bool
    shape = {batches, cin, cout, t, h, w}
    case stage
    when 1
      shape == {1, STAGE1_CHANNELS, STAGE1_CHANNELS, FRAMES, STAGE1_HEIGHT, STAGE1_WIDTH}
    when 2
      shape == {1, CHANNELS, CHANNELS, FRAMES, HEIGHT, WIDTH}
    else
      false
    end
  end

  def self.input_pack_values(stage : Int32) : Int32
    case stage
    when 1 then STAGE1_INPUT_PACK_VALUES
    when 2 then INPUT_PACK_VALUES
    else        raise ArgumentError.new("Unsupported DCAE-V C8 stage #{stage}")
    end
  end

  def self.weight_pack_values(stage : Int32) : Int32
    case stage
    when 1 then STAGE1_WEIGHT_PACK_VALUES
    when 2 then WEIGHT_PACK_VALUES
    else        raise ArgumentError.new("Unsupported DCAE-V C8 stage #{stage}")
    end
  end

  def self.scratch_bytes(stage : Int32 = 2) : Int64
    (input_pack_values(stage).to_i64 + weight_pack_values(stage)) * 2
  end

  # Capacity for one serial workspace used by both stage-specific plans.
  def self.shared_scratch_bytes : Int64
    (SHARED_INPUT_PACK_VALUES.to_i64 + SHARED_WEIGHT_PACK_VALUES) * 2
  end

  # The decoder workspace owns the two packs. Each use on the same serial
  # chain overwrites them from the current producer and filter.
  class Plan
    @pack_input : ML::Metal::ComputePipeline
    @pack_weight : ML::Metal::ComputePipeline
    @conv : ML::Metal::ComputePipeline
    @input_threads : Int32
    @weight_threads : Int32
    @conv_slices : Int32
    @conv_threads : Int32

    def initialize(arena : Arena, *, stage : Int32 = 2)
      input_pack_values = DCAEVDecoderC8.input_pack_values(stage)
      weight_pack_values = DCAEVDecoderC8.weight_pack_values(stage)
      @input_threads = input_pack_values // 8
      @weight_threads = weight_pack_values // 4
      channels, height, width = stage == 1 ? {STAGE1_CHANNELS, STAGE1_HEIGHT, STAGE1_WIDTH} : {CHANNELS, HEIGHT, WIDTH}
      @conv_slices = channels // 32
      @conv_threads = 8 * (FRAMES * height * width // 2)
      @input_pack = arena[INPUT_PACK]
      @weight_pack = arena[WEIGHT_PACK]
      MetalOps.check_storage(@input_pack, input_pack_values.to_i64, WeightDType::BF16)
      MetalOps.check_storage(@weight_pack, weight_pack_values.to_i64, WeightDType::BF16)
      prefix = stage == 1 ? "dcae_v_c8_stage1" : "dcae_v_c8"
      @pack_input = pipeline("#{prefix}_pack_input")
      @pack_weight = pipeline("#{prefix}_pack_weight")
      @conv = pipeline("#{prefix}_conv")
    end

    def encode(chain : ML::Metal::MPSCommandChain, input : ML::MetalBuffer,
               cache : ML::MetalBuffer?, weight : ML::MetalBuffer,
               bias : ML::MetalBuffer?, output : ML::MetalBuffer) : Nil
      e = chain.encoder
      e.set_pipeline(@pack_input)
      e.set_buffer(input, 0).set_buffer(cache || input, 1)
      e.set_buffer(@input_pack, 2, ML::Metal::BufferAccess::Write)
      e.set_value(cache ? 1_u32 : 0_u32, 3).dispatch_1d(@input_threads)

      e.set_pipeline(@pack_weight)
      e.set_buffer(weight, 0).set_buffer(@weight_pack, 1, ML::Metal::BufferAccess::Write)
      e.dispatch_1d(@weight_threads)

      e.set_pipeline(@conv)
      e.set_buffer(@input_pack, 0).set_buffer(@weight_pack, 1)
      e.set_buffer(output, 2, ML::Metal::BufferAccess::Write).set_buffer(bias || weight, 3)
      e.set_value(bias ? 1_u32 : 0_u32, 5)
      # Match the measured C8 grid: 32 output channels per dispatch, with
      # one thread for four output channels and two positions.
      @conv_slices.times do |slice|
        e.set_value((slice * 8).to_u32, 4)
        e.dispatch_1d(@conv_threads, 128)
      end
    end

    private def pipeline(name : String) : ML::Metal::ComputePipeline
      ML::Metal::PipelineCache.get(name) { ML::Metal::ComputePipeline.new(name, SOURCE) }
    end
  end
end
