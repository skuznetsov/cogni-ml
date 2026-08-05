# Bounded CPU coordinate layout for the TRELLIS.2 shape-SLat upsample seam.
#
# This leaf mirrors only SparseUpsample's 2x2x2 coordinate expansion and
# parent/subdivision ordering. It does not move feature values, run a sparse
# convolution, decode a latent, allocate device storage, or claim mesh parity.

module ML::ThreeD::Trellis2
  struct ShapeSlatUpsampleLayout
    getter factor : Int32
    getter coordinates : Array(ShapeSlatCoordinate)
    getter parent_indices : Array(Int32)
    getter subindices : Array(Int32)

    def initialize(
      @factor : Int32,
      coordinates : Array(ShapeSlatCoordinate),
      parent_indices : Array(Int32),
      subindices : Array(Int32),
    )
      unless coordinates.size == parent_indices.size &&
             coordinates.size == subindices.size
        raise ArgumentError.new(
          "upsample layout arrays must have equal lengths"
        )
      end
      @coordinates = coordinates.dup
      @parent_indices = parent_indices.dup
      @subindices = subindices.dup
    end

    def child_count : Int32
      @coordinates.size.to_i32
    end
  end

  module ShapeSlatUpsampleLayoutCPU
    extend self

    # TRELLIS.2 SparseUpsample is used for a 3D factor-2 decoder hierarchy.
    FACTOR                 =       2_i32
    SUBDIVISION_SLOTS      =       8_i32
    MAX_INPUT_COORDINATES  =  49_152_i32
    MAX_OUTPUT_COORDINATES = 393_216_i32

    # Mirror SparseUpsample's `sub.nonzero()` order: parent rows first, then
    # the little-endian 2x2x2 slot index from 0 through 7.
    def expand(
      coordinates : Array(ShapeSlatCoordinate),
      subdivisions : Array(Array(Bool)),
      *,
      max_input_coordinates : Int32 = MAX_INPUT_COORDINATES,
      max_output_coordinates : Int32 = MAX_OUTPUT_COORDINATES,
    ) : ShapeSlatUpsampleLayout
      unless max_input_coordinates > 0
        raise ArgumentError.new("input coordinate budget must be positive")
      end
      unless max_output_coordinates > 0
        raise ArgumentError.new("output coordinate budget must be positive")
      end
      if coordinates.size.to_i64 > max_input_coordinates.to_i64
        raise ArgumentError.new(
          "input coordinates exceed the upsample coordinate budget"
        )
      end
      unless subdivisions.size == coordinates.size
        raise ArgumentError.new("one mask per coordinate is required")
      end

      coordinates.each { |coordinate| validate_coordinate!(coordinate) }

      active_count = 0_i64
      subdivisions.each do |mask|
        unless mask.size == SUBDIVISION_SLOTS
          raise ArgumentError.new("each subdivision mask must contain exactly 8 slots")
        end
        mask.each { |active| active_count += 1_i64 if active }
      end
      if active_count > max_output_coordinates.to_i64
        raise ArgumentError.new("output coordinate budget exceeded")
      end

      child_coordinates = Array(ShapeSlatCoordinate).new(active_count.to_i)
      parent_indices = Array(Int32).new(active_count.to_i)
      subindices = Array(Int32).new(active_count.to_i)

      coordinates.each_with_index do |parent, parent_index|
        subdivisions[parent_index].each_with_index do |active, subindex|
          next unless active

          child_coordinates << ShapeSlatCoordinate.new(
            parent.batch,
            child_axis(parent.x, subindex.to_i32, 0_i32),
            child_axis(parent.y, subindex.to_i32, 1_i32),
            child_axis(parent.z, subindex.to_i32, 2_i32)
          )
          parent_indices << parent_index.to_i32
          subindices << subindex.to_i32
        end
      end

      ShapeSlatUpsampleLayout.new(
        FACTOR,
        child_coordinates,
        parent_indices,
        subindices
      )
    end

    private def validate_coordinate!(coordinate : ShapeSlatCoordinate) : Nil
      unless coordinate.batch >= 0 && coordinate.x >= 0 &&
             coordinate.y >= 0 && coordinate.z >= 0
        raise ArgumentError.new("coordinate batch and axes must be non-negative")
      end
    end

    private def child_axis(
      coordinate : Int32,
      subindex : Int32,
      axis : Int32,
    ) : Int32
      offset = case axis
               when 0 then subindex % FACTOR
               when 1 then (subindex // FACTOR) % FACTOR
               else        (subindex // (FACTOR * FACTOR)) % FACTOR
               end
      value = coordinate.to_i64 * FACTOR.to_i64 + offset.to_i64
      if value > Int32::MAX
        raise ArgumentError.new("upsampled coordinate exceeds Int32 range")
      end
      value.to_i32
    end
  end
end
