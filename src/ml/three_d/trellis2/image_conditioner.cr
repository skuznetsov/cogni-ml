# CPU reference for the fixed-resolution image tensor consumed by TRELLIS.2's
# DINOv3 feature extractor. This is preprocessing only: no model configuration,
# weights, embeddings, transformer blocks, accelerator, or Metal execution.

require "../../core/tensor"
require "./image_preprocess"

module ML::ThreeD::Trellis2
  class DinoV3ImageConditionerError < ImagePreprocessError
  end

  record DinoV3ImageCondition,
    resized : RGBImage,
    tensor : Tensor,
    source_revision : String,
    pillow_revision : String

  # Behavioral reimplementation of the RGB 8-bit LANCZOS path described by
  # Pillow 12.2.0 src/libImaging/Resample.c (MIT-CMU). Coefficients and per-pass
  # clipping preserve its 22-bit fixed-point behavior; Pillow is not linked or
  # executed by the native path.
  private module PillowLanczosRGB
    extend self

    PRECISION_BITS    = 22
    COEFFICIENT_SCALE = 1_i64 << PRECISION_BITS
    ROUNDING_BIAS     = 1_i64 << (PRECISION_BITS - 1)
    FILTER_SUPPORT    = 3.0_f64

    private record Coefficients,
      kernel_size : Int32,
      bounds : Array(Int32),
      weights : Array(Int32)

    def resize(input : RGBImage, output_width : Int32, output_height : Int32) : RGBImage
      unless output_width > 0 && output_height > 0
        raise DinoV3ImageConditionerError.new("resize dimensions must be positive")
      end
      if input.width == output_width && input.height == output_height
        return RGBImage.new(input.width, input.height, input.pixels.dup)
      end

      horizontal = precompute(input.width, output_width)
      vertical = precompute(input.height, output_height)
      horizontally_resized = if input.width == output_width
                               input
                             else
                               resize_horizontal(input, output_width, horizontal)
                             end
      if input.height == output_height
        return RGBImage.new(
          horizontally_resized.width,
          horizontally_resized.height,
          horizontally_resized.pixels.dup
        )
      end
      resize_vertical(horizontally_resized, output_height, vertical)
    end

    private def precompute(input_size : Int32, output_size : Int32) : Coefficients
      scale = input_size.to_f64 / output_size.to_f64
      filter_scale = Math.max(scale, 1.0_f64)
      support = FILTER_SUPPORT * filter_scale
      kernel_size = support.ceil.to_i32 * 2 + 1
      bounds = Array(Int32).new(output_size * 2, 0)
      weights = Array(Int32).new(output_size * kernel_size, 0)
      reciprocal_filter_scale = 1.0_f64 / filter_scale

      output_size.times do |output_index|
        center = (output_index.to_f64 + 0.5_f64) * scale
        minimum = (center - support + 0.5_f64).to_i
        minimum = 0 if minimum < 0
        maximum = (center + support + 0.5_f64).to_i
        maximum = input_size if maximum > input_size
        count = maximum - minimum
        bounds[output_index * 2] = minimum
        bounds[output_index * 2 + 1] = count

        double_weights = Array(Float64).new(count, 0.0_f64)
        sum = 0.0_f64
        count.times do |kernel_index|
          position = (
            kernel_index + minimum - center + 0.5_f64
          ) * reciprocal_filter_scale
          weight = lanczos(position)
          double_weights[kernel_index] = weight
          sum += weight
        end
        count.times do |kernel_index|
          normalized = sum == 0.0_f64 ? double_weights[kernel_index] : double_weights[kernel_index] / sum
          scaled = normalized * COEFFICIENT_SCALE.to_f64
          quantized = scaled < 0.0_f64 ? (scaled - 0.5_f64).to_i : (scaled + 0.5_f64).to_i
          weights[output_index * kernel_size + kernel_index] = quantized.to_i32
        end
      end
      Coefficients.new(kernel_size, bounds, weights)
    end

    private def lanczos(value : Float64) : Float64
      return 0.0_f64 unless value >= -3.0_f64 && value < 3.0_f64
      sinc(value) * sinc(value / 3.0_f64)
    end

    private def sinc(value : Float64) : Float64
      return 1.0_f64 if value == 0.0_f64
      angle = value * Math::PI
      Math.sin(angle) / angle
    end

    private def resize_horizontal(
      input : RGBImage,
      output_width : Int32,
      coefficients : Coefficients,
    ) : RGBImage
      output = Bytes.new(output_width * input.height * 3, 0_u8)
      input.height.times do |y|
        output_width.times do |output_x|
          minimum = coefficients.bounds[output_x * 2]
          count = coefficients.bounds[output_x * 2 + 1]
          3.times do |channel|
            accumulator = ROUNDING_BIAS
            count.times do |kernel_index|
              source_offset = (y * input.width + minimum + kernel_index) * 3 + channel
              weight = coefficients.weights[
                output_x * coefficients.kernel_size + kernel_index,
              ]
              accumulator += input.pixels[source_offset].to_i64 * weight.to_i64
            end
            output[(y * output_width + output_x) * 3 + channel] = clip(accumulator)
          end
        end
      end
      RGBImage.new(output_width, input.height, output)
    end

    private def resize_vertical(
      input : RGBImage,
      output_height : Int32,
      coefficients : Coefficients,
    ) : RGBImage
      output = Bytes.new(input.width * output_height * 3, 0_u8)
      output_height.times do |output_y|
        minimum = coefficients.bounds[output_y * 2]
        count = coefficients.bounds[output_y * 2 + 1]
        input.width.times do |x|
          3.times do |channel|
            accumulator = ROUNDING_BIAS
            count.times do |kernel_index|
              source_offset = ((minimum + kernel_index) * input.width + x) * 3 + channel
              weight = coefficients.weights[
                output_y * coefficients.kernel_size + kernel_index,
              ]
              accumulator += input.pixels[source_offset].to_i64 * weight.to_i64
            end
            output[(output_y * input.width + x) * 3 + channel] = clip(accumulator)
          end
        end
      end
      RGBImage.new(input.width, output_height, output)
    end

    private def clip(accumulator : Int64) : UInt8
      (accumulator >> PRECISION_BITS).clamp(0_i64, 255_i64).to_u8
    end
  end

  class DinoV3ImageConditionerCPU
    PILLOW_REVISION = "12.2.0"
    MAX_SOURCE_EDGE = 1024
    MEAN            = {0.485_f32, 0.456_f32, 0.406_f32}
    STD             = {0.229_f32, 0.224_f32, 0.225_f32}

    def prepare(input : RGBImage, target : Int32) : DinoV3ImageCondition
      unless target == 512 || target == 1024
        raise DinoV3ImageConditionerError.new("DINOv3 target must be 512 or 1024")
      end
      unless input.width == input.height
        raise DinoV3ImageConditionerError.new(
          "DINOv3 input must be square after TRELLIS.2 source preprocessing"
        )
      end
      if input.width > MAX_SOURCE_EDGE
        raise DinoV3ImageConditionerError.new(
          "DINOv3 source edge #{input.width} exceeds #{MAX_SOURCE_EDGE}"
        )
      end

      resized = PillowLanczosRGB.resize(input, target, target)
      shape = Shape.new(1_i32, 3_i32, target, target)
      tensor = Tensor.new(shape, device: Tensor::Device::CPU)
      values = tensor.cpu_data.not_nil!
      plane = target * target
      3.times do |channel|
        target.times do |y|
          target.times do |x|
            source = resized.pixels[(y * target + x) * 3 + channel]
            normalized = source.to_f32 / 255.0_f32
            values[channel * plane + y * target + x] = (
              normalized - MEAN[channel]
            ) / STD[channel]
          end
        end
      end

      DinoV3ImageCondition.new(
        resized,
        tensor,
        TRELLIS2_IMAGE_PREPROCESS_REVISION,
        PILLOW_REVISION
      )
    end
  end
end
