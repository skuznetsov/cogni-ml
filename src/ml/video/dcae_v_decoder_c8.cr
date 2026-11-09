require "./metal_ops"
require "./metal_mps_command_chain"

module ML::Video::DCAEVDecoderC8
  SOURCE = {{ read_file("#{__DIR__}/kernels/dcae_v_decoder_c8.metal") }}

  CHANNELS = 512
  FRAMES   =  40
  HEIGHT   =  72
  WIDTH    = 128

  INPUT_PACK         = "dcae_v_decoder.c8.input_pack"
  WEIGHT_PACK        = "dcae_v_decoder.c8.weight_pack"
  INPUT_PACK_VALUES  = CHANNELS * (FRAMES + 2) * HEIGHT * WIDTH
  WEIGHT_PACK_VALUES = CHANNELS * CHANNELS * 27

  def self.eligible?(batches : Int32, cin : Int32, cout : Int32,
                     t : Int32, h : Int32, w : Int32) : Bool
    {batches, cin, cout, t, h, w} == {1, CHANNELS, CHANNELS, FRAMES, HEIGHT, WIDTH}
  end

  def self.scratch_bytes : Int64
    (INPUT_PACK_VALUES.to_i64 + WEIGHT_PACK_VALUES) * 2
  end

  # The decoder workspace owns the two packs. Each use on the same serial
  # chain overwrites them from the current producer and filter.
  class Plan
    @pack_input : ML::Metal::ComputePipeline
    @pack_weight : ML::Metal::ComputePipeline
    @conv : ML::Metal::ComputePipeline

    def initialize(arena : Arena)
      @input_pack = arena[INPUT_PACK]
      @weight_pack = arena[WEIGHT_PACK]
      MetalOps.check_storage(@input_pack, INPUT_PACK_VALUES.to_i64, WeightDType::BF16)
      MetalOps.check_storage(@weight_pack, WEIGHT_PACK_VALUES.to_i64, WeightDType::BF16)
      @pack_input = pipeline("dcae_v_c8_pack_input")
      @pack_weight = pipeline("dcae_v_c8_pack_weight")
      @conv = pipeline("dcae_v_c8_conv")
    end

    def encode(chain : ML::Metal::MPSCommandChain, input : ML::MetalBuffer,
               cache : ML::MetalBuffer?, weight : ML::MetalBuffer,
               bias : ML::MetalBuffer?, output : ML::MetalBuffer) : Nil
      e = chain.encoder
      e.set_pipeline(@pack_input)
      e.set_buffer(input, 0).set_buffer(cache || input, 1)
      e.set_buffer(@input_pack, 2, ML::Metal::BufferAccess::Write)
      e.set_value(cache ? 1_u32 : 0_u32, 3).dispatch_1d(INPUT_PACK_VALUES // 8)

      e.set_pipeline(@pack_weight)
      e.set_buffer(weight, 0).set_buffer(@weight_pack, 1, ML::Metal::BufferAccess::Write)
      e.dispatch_1d(WEIGHT_PACK_VALUES // 4)

      e.set_pipeline(@conv)
      e.set_buffer(@input_pack, 0).set_buffer(@weight_pack, 1)
      e.set_buffer(output, 2, ML::Metal::BufferAccess::Write).set_buffer(bias || weight, 3)
      e.set_value(bias ? 1_u32 : 0_u32, 5)
      # Match the measured C8 grid: 32 output channels per dispatch, with
      # one thread for four output channels and two positions.
      (CHANNELS // 32).times do |slice|
        e.set_value((slice * 8).to_u32, 4)
        e.dispatch_1d(8 * (FRAMES * HEIGHT * WIDTH // 2), 128)
      end
    end

    private def pipeline(name : String) : ML::Metal::ComputePipeline
      ML::Metal::PipelineCache.get(name) { ML::Metal::ComputePipeline.new(name, SOURCE) }
    end
  end
end
