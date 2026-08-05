# Exact CPU metadata for the TRELLIS.2 shape-SLat C2S scale transition.
#
# SparseChannel2Spatial's uncached branch changes each spatial scale from `s`
# to `s / factor`. This leaf preserves that value metadata as normalized
# positive rationals. It does not create a sparse tensor, copy coordinates or
# features, inspect/register a spatial cache, or claim decoder/mesh parity.

module ML::ThreeD::Trellis2
  struct ShapeSlatScaleValue
    getter numerator : Int64
    getter denominator : Int64

    def initialize(numerator : Int64, denominator : Int64)
      unless numerator > 0
        raise ArgumentError.new("scale numerator must be positive")
      end
      unless denominator > 0
        raise ArgumentError.new("scale denominator must be positive")
      end

      divisor = greatest_common_divisor(numerator, denominator)
      @numerator = numerator // divisor
      @denominator = denominator // divisor
    end

    def ==(other : ShapeSlatScaleValue) : Bool
      @numerator == other.numerator && @denominator == other.denominator
    end

    def divide_by(factor : Int32) : ShapeSlatScaleValue
      unless factor > 0
        raise ArgumentError.new("scale division factor must be positive")
      end
      factor64 = factor.to_i64
      if @denominator > Int64::MAX // factor64
        raise ArgumentError.new("scale denominator multiplication overflow")
      end
      ShapeSlatScaleValue.new(@numerator, @denominator * factor64)
    end

    private def greatest_common_divisor(left : Int64, right : Int64) : Int64
      a = left
      b = right
      while b != 0
        a, b = b, a % b
      end
      a
    end
  end

  struct ShapeSlatScale
    getter x : ShapeSlatScaleValue
    getter y : ShapeSlatScaleValue
    getter z : ShapeSlatScaleValue

    def initialize(
      @x : ShapeSlatScaleValue,
      @y : ShapeSlatScaleValue,
      @z : ShapeSlatScaleValue,
    )
    end

    def ==(other : ShapeSlatScale) : Bool
      @x == other.x && @y == other.y && @z == other.z
    end
  end

  module ShapeSlatC2SScaleCPU
    extend self

    FACTOR = 2_i32

    # Mirror SparseChannel2Spatial's uncached `_scale = s / factor` metadata
    # transition without introducing floating-point rounding or payload work.
    def transition(
      input : ShapeSlatScale,
      *,
      factor : Int32 = FACTOR,
    ) : ShapeSlatScale
      unless factor == FACTOR
        raise ArgumentError.new("C2S scale transition requires factor #{FACTOR}")
      end

      ShapeSlatScale.new(
        input.x.divide_by(factor),
        input.y.divide_by(factor),
        input.z.divide_by(factor)
      )
    end
  end
end
