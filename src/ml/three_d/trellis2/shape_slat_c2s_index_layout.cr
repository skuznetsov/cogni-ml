# Bounded CPU metadata for the TRELLIS.2 shape-SLat channel-to-spatial seam.
#
# This leaf mirrors only SparseChannel2Spatial's packed row selection:
# `x.feats.reshape(N * 8, -1)[idx * 8 + subidx]`. It does not read feature
# values, allocate a sparse carrier, execute a convolution, inspect caches, or
# claim decoder/mesh parity.

module ML::ThreeD::Trellis2
  struct ShapeSlatC2SIndexLayout
    getter factor : Int32
    getter slots_per_parent : Int32
    getter output_channels : Int32
    getter packed_source_indices : Array(Int32)

    def initialize(
      @factor : Int32,
      @slots_per_parent : Int32,
      @output_channels : Int32,
      packed_source_indices : Array(Int32),
    )
      @packed_source_indices = packed_source_indices.dup
    end

    def child_count : Int32
      @packed_source_indices.size.to_i32
    end
  end

  module ShapeSlatC2SIndexLayoutCPU
    extend self

    FACTOR                 =       2_i32
    SUBDIVISION_SLOTS      =       8_i32
    MAX_INPUT_COORDINATES  =  49_152_i32
    MAX_OUTPUT_COORDINATES = 393_216_i32
    MAX_OUTPUT_CHANNELS    =   1_536_i32
    MAX_PACKED_CHANNELS    = MAX_OUTPUT_CHANNELS * SUBDIVISION_SLOTS
    MAX_METADATA_BYTES     = 64_i64 * 1024_i64 * 1024_i64

    # Build only the source-row metadata needed by SparseChannel2Spatial.
    # `layout` already owns the source-order coordinates, parent rows, and
    # little-endian subdivision slots established by ShapeSlatUpsampleLayout.
    def derive(
      layout : ShapeSlatUpsampleLayout,
      *,
      input_point_count : Int32,
      packed_channels : Int32,
      max_output_coordinates : Int32 = MAX_OUTPUT_COORDINATES,
      max_metadata_bytes : Int64 = MAX_METADATA_BYTES,
    ) : ShapeSlatC2SIndexLayout
      validate_metadata_budget!(max_metadata_bytes)
      validate_input_points!(input_point_count)
      validate_output_points!(max_output_coordinates)
      unless layout.factor == FACTOR
        raise ArgumentError.new(
          "C2S layout requires factor #{FACTOR}, got #{layout.factor}"
        )
      end
      unless layout.parent_indices.size == layout.child_count &&
             layout.subindices.size == layout.child_count
        raise ArgumentError.new("C2S layout mappings must match child count")
      end
      layout.coordinates.each { |coordinate| validate_coordinate!(coordinate) }
      unless 1 <= packed_channels <= MAX_PACKED_CHANNELS
        raise ArgumentError.new(
          "packed channel count must be in 1..#{MAX_PACKED_CHANNELS}"
        )
      end
      unless packed_channels % SUBDIVISION_SLOTS == 0
        raise ArgumentError.new(
          "packed channel count must be a multiple of #{SUBDIVISION_SLOTS}"
        )
      end
      if layout.child_count > max_output_coordinates
        raise ArgumentError.new("output coordinate budget exceeded")
      end

      metadata_bytes = checked_metadata_bytes(layout.child_count)
      if metadata_bytes > max_metadata_bytes
        raise ArgumentError.new(
          "metadata byte budget requires #{metadata_bytes} bytes, " \
          "limit is #{max_metadata_bytes}"
        )
      end

      # Validate every mapping before allocating the fresh index array.
      layout.parent_indices.each_with_index do |parent_index, child_index|
        unless 0 <= parent_index < input_point_count
          raise ArgumentError.new(
            "parent index #{parent_index} for child #{child_index} is out of range"
          )
        end
        subindex = layout.subindices[child_index]
        unless 0 <= subindex < SUBDIVISION_SLOTS
          raise ArgumentError.new(
            "subdivision slot #{subindex} for child #{child_index} is out of range"
          )
        end
      end

      packed_source_indices = Array(Int32).new(layout.child_count)
      layout.parent_indices.each_with_index do |parent_index, child_index|
        subindex = layout.subindices[child_index]
        packed_index = parent_index.to_i64 * SUBDIVISION_SLOTS.to_i64 + subindex.to_i64
        unless packed_index <= Int32::MAX.to_i64
          raise ArgumentError.new("packed source index exceeds Int32 range")
        end
        packed_source_indices << packed_index.to_i32
      end

      ShapeSlatC2SIndexLayout.new(
        FACTOR,
        SUBDIVISION_SLOTS,
        packed_channels // SUBDIVISION_SLOTS,
        packed_source_indices
      )
    end

    private def validate_input_points!(input_point_count : Int32) : Nil
      unless 0 <= input_point_count <= MAX_INPUT_COORDINATES
        raise ArgumentError.new(
          "input point count must be in 0..#{MAX_INPUT_COORDINATES}"
        )
      end
    end

    private def validate_coordinate!(coordinate : ShapeSlatCoordinate) : Nil
      unless coordinate.batch >= 0 && coordinate.x >= 0 &&
             coordinate.y >= 0 && coordinate.z >= 0
        raise ArgumentError.new("C2S coordinates must be non-negative")
      end
    end

    private def validate_output_points!(max_output_coordinates : Int32) : Nil
      unless 1 <= max_output_coordinates <= MAX_OUTPUT_COORDINATES
        raise ArgumentError.new(
          "output coordinate budget must be in 1..#{MAX_OUTPUT_COORDINATES}"
        )
      end
    end

    private def validate_metadata_budget!(max_metadata_bytes : Int64) : Nil
      unless 1_i64 <= max_metadata_bytes <= MAX_METADATA_BYTES
        raise ArgumentError.new(
          "metadata byte budget must be in 1..#{MAX_METADATA_BYTES}"
        )
      end
    end

    private def checked_metadata_bytes(child_count : Int32) : Int64
      bytes = child_count.to_i64 * 4_i64
      unless bytes >= child_count.to_i64
        raise ArgumentError.new("metadata byte count overflow")
      end
      bytes
    end
  end
end
