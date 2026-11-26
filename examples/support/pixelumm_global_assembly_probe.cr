# Synthetic, CPU-checkable fixture and oracle for the global PixelUMM output
# assembler. This does not run neural inference or claim model parity.
require "../../src/ml/video/pixelumm_generation"

module PixelUMMGlobalAssemblyProbe
  SPATIAL_POSITIONS = [0_i32, 73_727_i32, 147_455_i32]
  SPATIAL_SAMPLES   = SPATIAL_POSITIONS.size
  CHANNELS          =   3
  SEAM_START_FRAME  = 120

  def self.fixture_value(start_latent : Int32, local_frame : Int32,
                         channel : Int32, spatial : Int32) : Float32
    mixed = start_latent * 31 + local_frame * 17 + channel * 11 + (spatial % 7) * 5
    numerator = mixed % 257 - 128
    (numerator.to_f32 / 256.0_f32)
  end

  # Round finite Float32 values to BF16 using round-to-nearest, ties-to-even.
  def self.round_bf16(value : Float32) : Float32
    bits = value.unsafe_as(UInt32)
    bias = 0x7fff_u32 + ((bits >> 16) & 1_u32)
    ((bits + bias) & 0xffff0000_u32).unsafe_as(Float32)
  end

  def self.blend(previous : Float32, current : Float32, overlap_index : Int32) : Float32
    position = (overlap_index + 1).to_f32
    angle = (Math::PI.to_f32 * position) / 21.0_f32
    weight = round_bf16(0.5_f32 - 0.5_f32 * Math.cos(angle.to_f64).to_f32)
    inverse = round_bf16(1.0_f32 - weight)
    previous_term = round_bf16(previous * inverse)
    current_term = round_bf16(current * weight)
    round_bf16(previous_term + current_term)
  end

  def self.display_value(value : Float32) : Float32
    scaled = round_bf16(value * 0.5_f32)
    shifted = round_bf16(scaled + 0.5_f32)
    Math.min(1.0_f32, Math.max(0.0_f32, shifted))
  end

  # Evaluate the source assembler's ordered window writes for one output
  # sample. fixture_offset changes only the synthetic input identity; frame
  # coordinates and blending remain local to the supplied timeline.
  def self.assembled_display_value(total_latent_frames : Int32, frame : Int32,
                                   fixture_offset : Int32, channel : Int32,
                                   spatial : Int32) : Float32
    config = ML::Video::PixelUMMGeneration::Config.new(total_latent_frames: total_latent_frames)
    raise ArgumentError.new("sample frame is outside the output timeline") unless 0 <= frame < config.pixel_frames
    raise ArgumentError.new("sample channel is outside RGB output") unless 0 <= channel < CHANNELS
    raise ArgumentError.new("sample spatial index is outside a frame") unless 0 <= spatial < 288 * 512

    output : Float32? = nil
    config.window_starts.each do |start|
      pixel_start = start * ML::Video::PixelUMMGeneration::TUBELET_T
      local_frame = frame - pixel_start
      next unless 0 <= local_frame < ML::Video::PixelUMMGeneration::WINDOW_PIXEL_FRAMES

      current = fixture_value(fixture_offset + start, local_frame, channel, spatial)
      if start != 0 && local_frame < ML::Video::PixelUMMGeneration::WINDOW_OVERLAP_PIXEL_FRAMES
        previous = output || raise ArgumentError.new("overlap sample has no preceding window")
        output = blend(previous, current, local_frame)
      else
        output = current
      end
    end

    display_value(output || raise ArgumentError.new("sample is not covered by a decoder window"))
  end

  def self.global_display_value(frame : Int32, channel : Int32, spatial : Int32) : Float32
    assembled_display_value(60, frame, 0, channel, spatial)
  end

  # Model the forbidden independent-clip-concatenation control: each 30-latent
  # half gets its own local schedule and the second half has a fresh start.
  def self.independent_concat_display_value(frame : Int32, channel : Int32,
                                            spatial : Int32) : Float32
    raise ArgumentError.new("sample frame is outside the concatenated output") unless 0 <= frame < 240
    if frame < SEAM_START_FRAME
      assembled_display_value(30, frame, 0, channel, spatial)
    else
      assembled_display_value(30, frame - SEAM_START_FRAME, 30, channel, spatial)
    end
  end

  def self.sample_count(total_pixel_frames : Int32) : Int32
    total_pixel_frames * CHANNELS * SPATIAL_SAMPLES
  end

  # Interleave [global, concatenated] F32 pairs in the same order as the Metal
  # gather kernel: frame, channel, then SPATIAL_POSITIONS.
  def self.expected_samples : Array(Float32)
    values = Array(Float32).new(sample_count(240) * 2)
    240.times do |frame|
      CHANNELS.times do |channel|
        SPATIAL_POSITIONS.each do |spatial|
          values << global_display_value(frame, channel, spatial)
          values << independent_concat_display_value(frame, channel, spatial)
        end
      end
    end
    values
  end
end
