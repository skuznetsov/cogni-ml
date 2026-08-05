# Bounded CPU coordinate layout for the TRELLIS.2 shape-SLat cascade.
#
# This leaf mirrors only the source-pinned coordinate quantization and
# max-token resolution fallback in trellis2_image_to_3d.py. It does not decode
# a latent, run a sparse model, allocate device storage, or claim mesh parity.

module ML::ThreeD::Trellis2
  struct ShapeSlatCoordinate
    getter batch : Int32
    getter x : Int32
    getter y : Int32
    getter z : Int32

    def initialize(@batch : Int32, @x : Int32, @y : Int32, @z : Int32)
    end

    def ==(other : ShapeSlatCoordinate) : Bool
      @batch == other.batch &&
        @x == other.x &&
        @y == other.y &&
        @z == other.z
    end
  end

  struct ShapeSlatCascadeSelection
    getter requested_resolution : Int32
    getter actual_resolution : Int32
    getter max_num_tokens : Int32
    getter coordinates : Array(ShapeSlatCoordinate)

    def initialize(
      @requested_resolution : Int32,
      @actual_resolution : Int32,
      @max_num_tokens : Int32,
      @coordinates : Array(ShapeSlatCoordinate),
    )
    end

    def token_count : Int32
      @coordinates.size.to_i32
    end
  end

  module ShapeSlatCascadeLayoutCPU
    extend self

    # Source default from TRELLIS.2 ImageTo3DPipeline.run.
    DEFAULT_MAX_NUM_TOKENS = 49_152_i32

    # Source fallback loop decreases a 1024/1536 cascade in 128-resolution
    # increments and stops once the actual resolution reaches 1024.
    MIN_CASCADE_RESOLUTION = 1_024_i32
    MAX_CASCADE_RESOLUTION = 1_536_i32
    RESOLUTION_STEP        =   128_i32
    LATENT_GRID_DIVISOR    =    16_i32

    # Mirror the source expression
    # `((coord + 0.5) / lr_resolution * (target_resolution // 16)).int`
    # exactly for non-negative integer coordinates. Integer arithmetic avoids
    # introducing a new floating-point rounding policy at this metadata seam.
    def quantize(
      coordinates : Array(ShapeSlatCoordinate),
      *,
      lr_resolution : Int32,
      target_resolution : Int32,
    ) : Array(ShapeSlatCoordinate)
      validate_low_resolution!(lr_resolution)
      latent_resolution = validate_target_resolution!(target_resolution)

      quantized = Array(ShapeSlatCoordinate).new(coordinates.size) do |index|
        coordinate = coordinates[index]
        validate_coordinate!(coordinate, lr_resolution)
        ShapeSlatCoordinate.new(
          coordinate.batch,
          quantize_axis(coordinate.x, lr_resolution, latent_resolution),
          quantize_axis(coordinate.y, lr_resolution, latent_resolution),
          quantize_axis(coordinate.z, lr_resolution, latent_resolution)
        )
      end

      # torch.unique(dim=0) returns sorted unique rows by default. Preserve
      # that observable order rather than using a hash-dependent set order.
      quantized.sort! { |left, right| compare_coordinates(left, right) }
      unique = Array(ShapeSlatCoordinate).new(quantized.size)
      quantized.each do |coordinate|
        unique << coordinate if unique.empty? || unique.last != coordinate
      end
      unique
    end

    def select(
      coordinates : Array(ShapeSlatCoordinate),
      *,
      lr_resolution : Int32,
      requested_resolution : Int32,
      max_num_tokens : Int32 = DEFAULT_MAX_NUM_TOKENS,
    ) : ShapeSlatCascadeSelection
      validate_low_resolution!(lr_resolution)
      validate_cascade_resolution!(requested_resolution)
      unless max_num_tokens > 0
        raise ArgumentError.new("max_num_tokens must be positive")
      end

      actual_resolution = requested_resolution
      loop do
        quantized = quantize(
          coordinates,
          lr_resolution: lr_resolution,
          target_resolution: actual_resolution
        )
        token_count = quantized.size.to_i32
        # The strict `<` and the 1024 stop are both source-visible behavior.
        if token_count < max_num_tokens || actual_resolution == MIN_CASCADE_RESOLUTION
          return ShapeSlatCascadeSelection.new(
            requested_resolution,
            actual_resolution,
            max_num_tokens,
            quantized
          )
        end
        actual_resolution -= RESOLUTION_STEP
      end
    end

    private def validate_low_resolution!(resolution : Int32) : Nil
      raise ArgumentError.new("lr_resolution must be positive") unless resolution > 0
    end

    private def validate_target_resolution!(resolution : Int32) : Int32
      unless resolution > 0 && resolution % LATENT_GRID_DIVISOR == 0
        raise ArgumentError.new(
          "target_resolution must be positive and divisible by #{LATENT_GRID_DIVISOR}"
        )
      end
      resolution // LATENT_GRID_DIVISOR
    end

    private def validate_cascade_resolution!(resolution : Int32) : Nil
      unless resolution >= MIN_CASCADE_RESOLUTION &&
             resolution <= MAX_CASCADE_RESOLUTION &&
             resolution % RESOLUTION_STEP == 0
        raise ArgumentError.new(
          "requested_resolution must be a multiple of #{RESOLUTION_STEP} " \
          "within [#{MIN_CASCADE_RESOLUTION}, #{MAX_CASCADE_RESOLUTION}]"
        )
      end
    end

    private def validate_coordinate!(coordinate : ShapeSlatCoordinate, lr_resolution : Int32) : Nil
      unless coordinate.batch >= 0
        raise ArgumentError.new("coordinate batch must be non-negative")
      end
      unless coordinate.x >= 0 && coordinate.x < lr_resolution &&
             coordinate.y >= 0 && coordinate.y < lr_resolution &&
             coordinate.z >= 0 && coordinate.z < lr_resolution
        raise ArgumentError.new(
          "coordinate axes must be within [0, #{lr_resolution})"
        )
      end
    end

    private def quantize_axis(coordinate : Int32, lr_resolution : Int32, latent_resolution : Int32) : Int32
      numerator = (coordinate.to_i64 * 2_i64 + 1_i64) * latent_resolution.to_i64
      denominator = 2_i64 * lr_resolution.to_i64
      (numerator // denominator).to_i32
    end

    private def compare_coordinates(left : ShapeSlatCoordinate, right : ShapeSlatCoordinate) : Int32
      comparison = left.batch <=> right.batch
      return comparison unless comparison == 0
      comparison = left.x <=> right.x
      return comparison unless comparison == 0
      comparison = left.y <=> right.y
      return comparison unless comparison == 0
      left.z <=> right.z
    end
  end
end
