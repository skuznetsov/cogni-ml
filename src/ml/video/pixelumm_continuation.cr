# GPU-only assembly and output crop for a previous-window PixelUMM continuation.
# This plan owns the joined decoder inputs and current-segment display output;
# all four input histories remain borrowed from the caller.
require "../core/buffer"
require "../metal/device"
require "./metal_mps_command_chain"
require "./metal_ops"
require "./pixelumm_generation"

module ML::Video::PixelUMMContinuation
  SOURCE = {{ read_file("#{__DIR__}/kernels/pixelumm_continuation.metal") }}

  CONTEXT_LATENT_FRAMES = 5
  TUBELET_T             = 4
  CONTEXT_PIXEL_FRAMES  = CONTEXT_LATENT_FRAMES * TUBELET_T

  def self.pipeline(name : String) : ML::Metal::ComputePipeline
    ML::Metal::PipelineCache.get(name) { ML::Metal::ComputePipeline.new(name, SOURCE) }
  end

  struct Config
    getter previous_latent_frames : Int32
    getter current_latent_frames : Int32
    getter latent_height : Int32
    getter latent_width : Int32
    getter decoder_config : PixelUMMGeneration::Config
    getter current_config : PixelUMMGeneration::Config

    def initialize(@previous_latent_frames : Int32,
                   @current_latent_frames : Int32 = PixelUMMGeneration::LATENT_FRAMES,
                   @latent_height : Int32 = PixelUMMGeneration::LATENT_HEIGHT,
                   @latent_width : Int32 = PixelUMMGeneration::LATENT_WIDTH)
      validate_timeline!
      @current_config = PixelUMMGeneration::Config.new(
        total_latent_frames: @current_latent_frames,
        latent_height: @latent_height, latent_width: @latent_width)
      @decoder_config = PixelUMMGeneration::Config.new(
        total_latent_frames: @current_latent_frames + CONTEXT_LATENT_FRAMES,
        latent_height: @latent_height, latent_width: @latent_width)
      validate!
    end

    def validate! : Nil
      validate_timeline!
      @current_config.validate!
      @decoder_config.validate!
      unless @decoder_config.total_latent_frames == @current_latent_frames + CONTEXT_LATENT_FRAMES &&
             @current_config.total_latent_frames == @current_latent_frames
        raise ArgumentError.new("PixelUMM continuation decoder/current timeline geometry is inconsistent")
      end
      # Input shape and byte helpers are also admission checks. The previous
      # timeline has no fixed 180-latent cap, but its native dimensions and
      # resident byte extents must remain representable.
      previous_noise_frames
      previous_latent_bytes
      previous_noise_bytes
      current_latent_bytes
      current_noise_bytes
      named_bytes_by_name
    end

    def previous_latent_shape : Array(Int32)
      [1_i32, PixelUMMGeneration::LATENT_CHANNELS, @previous_latent_frames,
       @latent_height, @latent_width]
    end

    def previous_noise_shape : Array(Int32)
      [1_i32, PixelUMMGeneration::PIXEL_CHANNELS, previous_noise_frames,
       @current_config.pixel_height, @current_config.pixel_width]
    end

    def current_latent_shape : Array(Int32)
      @current_config.latent_shape
    end

    def current_noise_shape : Array(Int32)
      @current_config.noise_shape
    end

    # Plan-owned joined latent input [1,32,current+5,H,W].
    def latent_shape : Array(Int32)
      @decoder_config.latent_shape
    end

    # Plan-owned joined noise input [1,3,(current+5)*4,H*32,W*32].
    def noise_shape : Array(Int32)
      @decoder_config.noise_shape
    end

    # Plan-owned display output has only the requested current segment.
    def output_shape : Array(Int32)
      @current_config.output_shape
    end

    def previous_latent_elements : Int64
      product(PixelUMMGeneration::LATENT_CHANNELS.to_i64,
        @previous_latent_frames.to_i64, @latent_height.to_i64, @latent_width.to_i64)
    end

    def previous_noise_elements : Int64
      product(PixelUMMGeneration::PIXEL_CHANNELS.to_i64, previous_noise_frames.to_i64,
        @current_config.pixel_height.to_i64, @current_config.pixel_width.to_i64)
    end

    def current_latent_elements : Int64
      @current_config.latent_elements
    end

    def current_noise_elements : Int64
      @current_config.output_elements
    end

    def latent_elements : Int64
      @decoder_config.latent_elements
    end

    def noise_elements : Int64
      @decoder_config.output_elements
    end

    def output_elements : Int64
      @current_config.output_elements
    end

    # Exact named plan-owned MetalBuffer payloads. Borrowed source histories
    # are intentionally accounted separately by callers.
    def named_bytes_by_name : Hash(String, Int64)
      {
        "latent_f32" => bytes_for(latent_elements),
        "noise_f32"  => bytes_for(noise_elements),
        "output_f32" => bytes_for(output_elements),
      }
    end

    def named_bytes : Int64
      named_bytes_by_name.values.sum
    end

    # CPU oracle matching the latent GPU gather. The pair is {source buffer,
    # flat source offset} for one flat joined NCTHW destination element.
    def latent_join_source(output_index : Int32) : Tuple(Symbol, Int64)
      unless 0 <= output_index && output_index.to_i64 < latent_elements
        raise ArgumentError.new("PixelUMM joined latent index is out of range")
      end

      spatial = @latent_height.to_i64 * @latent_width
      joined_frames = @current_latent_frames.to_i64 + CONTEXT_LATENT_FRAMES
      joined_channel_stride = joined_frames * spatial
      channel = output_index.to_i64 // joined_channel_stride
      within_channel = output_index.to_i64 % joined_channel_stride
      frame = within_channel // spatial
      cell = within_channel % spatial
      if frame < CONTEXT_LATENT_FRAMES
        source_frame = @previous_latent_frames.to_i64 - CONTEXT_LATENT_FRAMES + frame
        {:previous, channel * @previous_latent_frames.to_i64 * spatial + source_frame * spatial + cell}
      else
        current_frame = frame - CONTEXT_LATENT_FRAMES
        {:current, channel * @current_latent_frames.to_i64 * spatial + current_frame * spatial + cell}
      end
    end

    # CPU oracle matching the noise GPU gather, with the tubelet temporal
    # expansion applied before indexing either full NCTHW history.
    def noise_join_source(output_index : Int32) : Tuple(Symbol, Int64)
      unless 0 <= output_index && output_index.to_i64 < noise_elements
        raise ArgumentError.new("PixelUMM joined noise index is out of range")
      end

      spatial = @current_config.pixel_height.to_i64 * @current_config.pixel_width
      joined_frames = @decoder_config.pixel_frames.to_i64
      previous_frames = previous_noise_frames.to_i64
      current_frames = @current_config.pixel_frames.to_i64
      joined_channel_stride = joined_frames * spatial
      channel = output_index.to_i64 // joined_channel_stride
      within_channel = output_index.to_i64 % joined_channel_stride
      frame = within_channel // spatial
      cell = within_channel % spatial
      if frame < CONTEXT_PIXEL_FRAMES
        source_frame = previous_frames - CONTEXT_PIXEL_FRAMES + frame
        {:previous, channel * previous_frames * spatial + source_frame * spatial + cell}
      else
        current_frame = frame - CONTEXT_PIXEL_FRAMES
        {:current, channel * current_frames * spatial + current_frame * spatial + cell}
      end
    end

    # CPU oracle matching the display GPU crop. It retains full channel
    # strides while skipping only the first four-tubelet (20 pixel) context.
    def output_source_index(output_index : Int32) : Int64
      unless 0 <= output_index && output_index.to_i64 < output_elements
        raise ArgumentError.new("PixelUMM continuation output index is out of range")
      end

      spatial = @current_config.pixel_height.to_i64 * @current_config.pixel_width
      cropped_frames = @current_config.pixel_frames.to_i64
      full_frames = @decoder_config.pixel_frames.to_i64
      output_channel_stride = cropped_frames * spatial
      channel = output_index.to_i64 // output_channel_stride
      within_channel = output_index.to_i64 % output_channel_stride
      frame = within_channel // spatial
      cell = within_channel % spatial
      channel * full_frames * spatial + (frame + CONTEXT_PIXEL_FRAMES) * spatial + cell
    end

    def previous_noise_frames : Int32
      checked_extent(@previous_latent_frames.to_i64 * TUBELET_T, "previous noise frame count")
    end

    private def validate_timeline! : Nil
      unless @previous_latent_frames >= CONTEXT_LATENT_FRAMES &&
             @previous_latent_frames % CONTEXT_LATENT_FRAMES == 0
        raise ArgumentError.new("previous PixelUMM latent frames must be at least five and divisible by five")
      end
      unless @current_latent_frames > 0 &&
             @current_latent_frames <= PixelUMMGeneration::MAX_LATENT_FRAMES - CONTEXT_LATENT_FRAMES
        raise ArgumentError.new("current PixelUMM continuation plus five context latents must not exceed 180")
      end
    end

    private def checked_extent(value : Int64, label : String) : Int32
      unless value > 0 && value <= Int32::MAX
        raise ArgumentError.new("PixelUMM continuation #{label} exceeds native dimensions")
      end
      value.to_i32
    end

    private def product(*factors : Int64) : Int64
      value = 1_i64
      factors.each do |factor|
        unless factor > 0 && value <= Int64::MAX // factor
          raise ArgumentError.new("PixelUMM continuation tensor extent exceeds native byte limits")
        end
        value *= factor
      end
      value
    end

    private def bytes_for(elements : Int64) : Int64
      unless elements > 0 && elements <= Int64::MAX // sizeof(Float32)
        raise ArgumentError.new("PixelUMM continuation tensor byte extent exceeds native limits")
      end
      elements * sizeof(Float32)
    end

    private def previous_latent_bytes : Int64
      bytes_for(previous_latent_elements)
    end

    private def previous_noise_bytes : Int64
      bytes_for(previous_noise_elements)
    end

    private def current_latent_bytes : Int64
      bytes_for(current_latent_elements)
    end

    private def current_noise_bytes : Int64
      bytes_for(current_noise_elements)
    end
  end

  class Plan
    getter config : Config
    getter latent_f32 : ML::MetalBuffer
    getter noise_f32 : ML::MetalBuffer
    getter output_f32 : ML::MetalBuffer
    getter named_bytes_by_name : Hash(String, Int64)
    getter named_bytes : Int64

    @arena : Arena
    @previous_latent : ML::MetalBuffer
    @previous_noise : ML::MetalBuffer
    @current_latent : ML::MetalBuffer
    @current_noise : ML::MetalBuffer
    @chain : ML::Metal::MPSCommandChain? = nil
    @inputs_encoded = false
    @output_encoded = false
    @failed = false
    @released = false

    def initialize(@config : Config, *, previous_latent : ML::MetalBuffer,
                   previous_noise : ML::MetalBuffer, current_latent : ML::MetalBuffer,
                   current_noise : ML::MetalBuffer)
      @config.validate!
      @previous_latent = previous_latent
      @previous_noise = previous_noise
      @current_latent = current_latent
      @current_noise = current_noise
      validate_inputs!
      @named_bytes_by_name = @config.named_bytes_by_name
      @named_bytes = @config.named_bytes
      @arena = Arena.new
      begin
        PixelUMMContinuation.pipeline("pixelumm_continuation_join_latent")
        PixelUMMContinuation.pipeline("pixelumm_continuation_join_noise")
        PixelUMMContinuation.pipeline("pixelumm_continuation_crop_display")
        allocate_buffers!
      rescue ex
        @arena.release
        @released = true
        raise ex
      end
      @latent_f32 = @arena["latent_f32"]
      @noise_f32 = @arena["noise_f32"]
      @output_f32 = @arena["output_f32"]
      begin
        validate_owned_separation!
      rescue ex
        @arena.release
        @released = true
        raise ex
      end
    end

    # GPU-copy the last five previous latent/noise frames followed by every
    # current frame. Setup owns all pipelines and named buffers, so this method
    # only appends two dispatches to the caller's serial chain.
    def encode_inputs(chain : ML::Metal::MPSCommandChain) : Nil
      ensure_usable!
      raise ArgumentError.new("PixelUMM continuation inputs were already encoded") if @inputs_encoded
      bind_chain!(chain)
      e = chain.encoder
      begin
        encode_latent_join(e)
        encode_noise_join(e)
        @inputs_encoded = true
      rescue ex
        @failed = true
        raise ex
      end
    end

    # Crop the joined decoder's finalized display-range F32 output after its
    # twenty-pixel context prefix, retaining only current-segment frames.
    def encode_output(chain : ML::Metal::MPSCommandChain,
                      *, source : ML::MetalBuffer) : ML::MetalBuffer
      ensure_usable!
      raise ArgumentError.new("PixelUMM continuation inputs must be encoded before the output crop") unless @inputs_encoded
      raise ArgumentError.new("PixelUMM continuation output was already encoded") if @output_encoded
      validate_exact_f32!(source, @config.decoder_config.output_elements, "full decoder display output")
      ensure_disjoint!(source, @latent_f32, @noise_f32, @output_f32,
        @previous_latent, @previous_noise, @current_latent, @current_noise)
      bind_chain!(chain)
      e = chain.encoder
      begin
        e.set_pipeline(PixelUMMContinuation.pipeline("pixelumm_continuation_crop_display"))
          .set_buffer(source, 0)
          .set_buffer(@output_f32, 1, ML::Metal::BufferAccess::Write)
          .set_value(@config.current_config.pixel_frames.to_u32, 2)
          .set_value(@config.decoder_config.pixel_frames.to_u32, 3)
          .set_value(@config.current_config.pixel_height.to_u32, 4)
          .set_value(@config.current_config.pixel_width.to_u32, 5)
          .set_value(@config.output_elements.to_u32, 6)
          .dispatch_1d(@config.output_elements.to_i32)
        @output_encoded = true
        @output_f32
      rescue ex
        @failed = true
        raise ex
      end
    end

    # Caller owns the terminal wait. Plan buffers and borrowed sources stay
    # alive until it completes; release never introduces a synchronization.
    def release_after_terminal_wait! : Nil
      return if @released
      if chain = @chain
        raise ArgumentError.new("PixelUMM continuation resources require the terminal chain wait") unless chain.completed?
      end
      @arena.release
      @released = true
    end

    def release : Nil
      release_after_terminal_wait!
    end

    private def validate_inputs! : Nil
      validate_exact_f32!(@previous_latent, @config.previous_latent_elements, "previous latent history")
      validate_exact_f32!(@previous_noise, @config.previous_noise_elements, "previous noise history")
      validate_exact_f32!(@current_latent, @config.current_latent_elements, "current latent input")
      validate_exact_f32!(@current_noise, @config.current_noise_elements, "current noise input")
      ensure_disjoint!(@previous_latent, @previous_noise, @current_latent, @current_noise)
    end

    private def validate_exact_f32!(buffer : ML::MetalBuffer, elements : Int64,
                                    label : String) : Nil
      expected_bytes = elements * sizeof(Float32)
      unless buffer.valid? && elements > 0 && buffer.size == expected_bytes
        raise ArgumentError.new("PixelUMM continuation #{label} must be an exact-size F32 buffer")
      end
    end

    private def ensure_disjoint!(*buffers : ML::MetalBuffer) : Nil
      buffers.each_with_index do |buffer, index|
        other_index = 0
        buffers.each do |other|
          if other_index < index && MetalOps.overlap?(buffer, other)
            raise ArgumentError.new("PixelUMM continuation buffers must not overlap")
          end
          other_index += 1
        end
      end
    end

    private def validate_owned_separation! : Nil
      ensure_disjoint!(@latent_f32, @noise_f32, @output_f32,
        @previous_latent, @previous_noise, @current_latent, @current_noise)
    end

    private def allocate_buffers! : Nil
      @arena.alloc("latent_f32", @config.latent_elements.to_i32)
      @arena.alloc("noise_f32", @config.noise_elements.to_i32)
      @arena.alloc("output_f32", @config.output_elements.to_i32)
    end

    private def ensure_usable! : Nil
      raise ArgumentError.new("PixelUMM continuation plan has been released") if @released
      raise ArgumentError.new("PixelUMM continuation plan cannot continue after an encode error") if @failed
    end

    private def bind_chain!(chain : ML::Metal::MPSCommandChain) : Nil
      if existing = @chain
        unless existing.same?(chain)
          raise ArgumentError.new("PixelUMM continuation encodes must share one command chain")
        end
      else
        @chain = chain
      end
    end

    private def encode_latent_join(e : ML::Metal::ComputeEncoder) : Nil
      e.set_pipeline(PixelUMMContinuation.pipeline("pixelumm_continuation_join_latent"))
        .set_buffer(@previous_latent, 0)
        .set_buffer(@current_latent, 1)
        .set_buffer(@latent_f32, 2, ML::Metal::BufferAccess::Write)
        .set_value(@config.previous_latent_frames.to_u32, 3)
        .set_value(@config.current_latent_frames.to_u32, 4)
        .set_value(@config.latent_height.to_u32, 5)
        .set_value(@config.latent_width.to_u32, 6)
        .set_value(@config.latent_elements.to_u32, 7)
        .dispatch_1d(@config.latent_elements.to_i32)
    end

    private def encode_noise_join(e : ML::Metal::ComputeEncoder) : Nil
      e.set_pipeline(PixelUMMContinuation.pipeline("pixelumm_continuation_join_noise"))
        .set_buffer(@previous_noise, 0)
        .set_buffer(@current_noise, 1)
        .set_buffer(@noise_f32, 2, ML::Metal::BufferAccess::Write)
        .set_value(@config.previous_noise_frames.to_u32, 3)
        .set_value(@config.current_config.pixel_frames.to_u32, 4)
        .set_value(@config.decoder_config.pixel_frames.to_u32, 5)
        .set_value(@config.current_config.pixel_height.to_u32, 6)
        .set_value(@config.current_config.pixel_width.to_u32, 7)
        .set_value(@config.noise_elements.to_u32, 8)
        .dispatch_1d(@config.noise_elements.to_i32)
    end
  end
end
