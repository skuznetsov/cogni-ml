# Bounded CPU image preprocessing for the TRELLIS.2 DINOv3 conditioning path.
#
# This slice mirrors the alpha/rembg, square-crop, and black-premultiply
# boundary from microsoft/TRELLIS.2 at the pinned revision below. The separate
# Pillow LANCZOS resize and ImageNet normalization boundary is intentionally not
# admitted here yet.

module ML::ThreeD::Trellis2
  TRELLIS2_IMAGE_PREPROCESS_REVISION = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"

  class ImagePreprocessError < Exception
  end

  class BackgroundRemovalRequiredError < ImagePreprocessError
  end

  class BackgroundRemovalGeometryError < ImagePreprocessError
  end

  class EmptyForegroundError < ImagePreprocessError
  end

  class DegenerateForegroundError < ImagePreprocessError
  end

  class ImageResamplingRequiredError < ImagePreprocessError
  end

  private module ImageBufferContract
    extend self

    def validate!(width : Int32, height : Int32, channels : Int32, pixels : Bytes) : Nil
      unless width > 0 && height > 0
        raise ArgumentError.new("image dimensions must be positive")
      end
      expected = width.to_i64 * height.to_i64 * channels.to_i64
      if expected > Int32::MAX
        raise ArgumentError.new("image pixel buffer exceeds addressable Slice size")
      end
      unless pixels.size == expected
        raise ArgumentError.new(
          "image pixel buffer has #{pixels.size} bytes, expected #{expected}"
        )
      end
    end

    def offset(width : Int32, height : Int32, channels : Int32, x : Int32, y : Int32) : Int32
      unless x >= 0 && x < width && y >= 0 && y < height
        raise IndexError.new("pixel coordinate (#{x}, #{y}) is outside #{width}x#{height}")
      end
      (y * width + x) * channels
    end
  end

  # A non-owning interleaved RGBA byte view. The caller must keep the backing
  # storage alive and must not mutate it during preprocessing.
  struct RGBAImage
    getter width : Int32
    getter height : Int32
    getter pixels : Bytes

    def initialize(@width : Int32, @height : Int32, @pixels : Bytes)
      ImageBufferContract.validate!(@width, @height, 4, @pixels)
    end

    def pixel(x : Int32, y : Int32) : {UInt8, UInt8, UInt8, UInt8}
      offset = ImageBufferContract.offset(@width, @height, 4, x, y)
      {
        @pixels[offset],
        @pixels[offset + 1],
        @pixels[offset + 2],
        @pixels[offset + 3],
      }
    end
  end

  # A non-owning interleaved RGB byte view.
  struct RGBImage
    getter width : Int32
    getter height : Int32
    getter pixels : Bytes

    def initialize(@width : Int32, @height : Int32, @pixels : Bytes)
      ImageBufferContract.validate!(@width, @height, 3, @pixels)
    end

    def pixel(x : Int32, y : Int32) : {UInt8, UInt8, UInt8}
      offset = ImageBufferContract.offset(@width, @height, 3, x, y)
      {@pixels[offset], @pixels[offset + 1], @pixels[offset + 2]}
    end
  end

  abstract class BackgroundRemoverCPU
    abstract def remove(image : RGBImage) : RGBAImage
  end

  record CropBox,
    left : Int32,
    top : Int32,
    right : Int32,
    bottom : Int32

  record ImagePreprocessResult,
    image : RGBImage,
    crop : CropBox,
    background_removed : Bool,
    source_revision : String

  class Trellis2ImagePreprocessorCPU
    ALPHA_FOREGROUND_THRESHOLD = 204_u8
    MAX_EDGE_WITHOUT_RESAMPLE  =   1024

    def initialize(@background_remover : BackgroundRemoverCPU? = nil)
    end

    def preprocess(input : RGBAImage) : ImagePreprocessResult
      if input.width > MAX_EDGE_WITHOUT_RESAMPLE || input.height > MAX_EDGE_WITHOUT_RESAMPLE
        raise ImageResamplingRequiredError.new(
          "TRELLIS.2 source images above #{MAX_EDGE_WITHOUT_RESAMPLE}px require the " \
          "not-yet-admitted Pillow LANCZOS parity boundary"
        )
      end

      has_alpha = false
      alpha_offset = 3
      while alpha_offset < input.pixels.size
        if input.pixels[alpha_offset] != 255_u8
          has_alpha = true
          break
        end
        alpha_offset += 4
      end

      background_removed = !has_alpha
      rgba = if has_alpha
               input
             else
               remover = @background_remover || raise BackgroundRemovalRequiredError.new(
                 "fully opaque TRELLIS.2 input requires an explicit background remover"
               )
               removed = remover.remove(to_rgb(input))
               unless removed.width == input.width && removed.height == input.height
                 raise BackgroundRemovalGeometryError.new(
                   "background remover changed image geometry from " \
                   "#{input.width}x#{input.height} to #{removed.width}x#{removed.height}"
                 )
               end
               removed
             end

      min_x = rgba.width
      min_y = rgba.height
      max_x = -1
      max_y = -1
      rgba.height.times do |y|
        rgba.width.times do |x|
          alpha = rgba.pixels[(y * rgba.width + x) * 4 + 3]
          next unless alpha > ALPHA_FOREGROUND_THRESHOLD
          min_x = x if x < min_x
          min_y = y if y < min_y
          max_x = x if x > max_x
          max_y = y if y > max_y
        end
      end

      if max_x < 0
        raise EmptyForegroundError.new(
          "TRELLIS.2 foreground has no alpha values above #{ALPHA_FOREGROUND_THRESHOLD}"
        )
      end

      extent = Math.max(max_x - min_x, max_y - min_y)
      half_extent = extent // 2
      left = round_half_even(min_x + max_x - 2 * half_extent)
      top = round_half_even(min_y + max_y - 2 * half_extent)
      right = round_half_even(min_x + max_x + 2 * half_extent)
      bottom = round_half_even(min_y + max_y + 2 * half_extent)
      crop = CropBox.new(left, top, right, bottom)

      output_width = right - left
      output_height = bottom - top
      unless output_width > 0 && output_height > 0
        raise DegenerateForegroundError.new(
          "TRELLIS.2 foreground extent #{extent} produces a degenerate Pillow crop"
        )
      end

      output = Bytes.new(output_width * output_height * 3, 0_u8)
      output_height.times do |output_y|
        source_y = top + output_y
        next unless source_y >= 0 && source_y < rgba.height
        output_width.times do |output_x|
          source_x = left + output_x
          next unless source_x >= 0 && source_x < rgba.width
          source_offset = (source_y * rgba.width + source_x) * 4
          destination_offset = (output_y * output_width + output_x) * 3
          alpha = rgba.pixels[source_offset + 3]
          output[destination_offset] = premultiply(rgba.pixels[source_offset], alpha)
          output[destination_offset + 1] = premultiply(rgba.pixels[source_offset + 1], alpha)
          output[destination_offset + 2] = premultiply(rgba.pixels[source_offset + 2], alpha)
        end
      end

      ImagePreprocessResult.new(
        RGBImage.new(output_width, output_height, output),
        crop,
        background_removed,
        TRELLIS2_IMAGE_PREPROCESS_REVISION
      )
    end

    private def to_rgb(image : RGBAImage) : RGBImage
      output = Bytes.new(image.width * image.height * 3, 0_u8)
      image.height.times do |y|
        image.width.times do |x|
          source_offset = (y * image.width + x) * 4
          destination_offset = (y * image.width + x) * 3
          output[destination_offset] = image.pixels[source_offset]
          output[destination_offset + 1] = image.pixels[source_offset + 1]
          output[destination_offset + 2] = image.pixels[source_offset + 2]
        end
      end
      RGBImage.new(image.width, image.height, output)
    end

    # Pillow maps floating crop coordinates through Python's ties-to-even round.
    # The caller passes twice the coordinate, so this stays exact for .0/.5.
    private def round_half_even(doubled : Int32) : Int32
      return doubled.tdiv(2) if doubled.even?
      lower = doubled.tdiv(2)
      lower -= 1 if doubled < 0
      lower.even? ? lower : lower + 1
    end

    # Preserve NumPy's float32 operation order from:
    # (rgb / 255) * (alpha / 255) * 255, followed by uint8 truncation.
    private def premultiply(channel : UInt8, alpha : UInt8) : UInt8
      normalized_channel = channel.to_f32 / 255.0_f32
      normalized_alpha = alpha.to_f32 / 255.0_f32
      value = (normalized_channel * normalized_alpha * 255.0_f32).to_i
      value.clamp(0, 255).to_u8
    end
  end
end
