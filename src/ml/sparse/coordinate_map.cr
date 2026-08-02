require "digest/sha256"

module ML::Sparse
  class CoordinateMapError < Exception
  end

  class CoordinateMapBudgetError < CoordinateMapError
  end

  record BatchSlice, start : Int32, stop : Int32 do
    def size : Int32
      stop - start
    end
  end

  # Immutable, order-preserving CPU coordinate identity for sparse 3D values.
  #
  # Rows are [batch, x, y, z]. Batches are grouped but spatial rows retain the
  # caller's order because features are row-aligned and upstream model parity
  # must not depend on an implicit lexicographic or Morton permutation.
  class CoordinateMap3D
    MAX_POINTS         = 65_536_i32
    MAX_BATCH_SIZE     =     64_i32
    MAX_SPATIAL_EXTENT =  1_024_i32
    COORDINATE_WIDTH   =      4_i32

    getter batch_size : Int32
    getter point_count : Int32
    getter spatial_shape : Tuple(Int32, Int32, Int32)
    getter occupied_spatial_shape : Tuple(Int32, Int32, Int32)
    getter digest : String

    @coordinates : Array(Int32)
    @layout : Array(BatchSlice)
    @sequence_lengths : Array(Int32)
    @cumulative_sequence_lengths : Array(Int32)
    @batch_broadcast_map : Array(Int32)
    @lookup : Hash(Int64, Int32)

    def initialize(
      coordinates : Indexable(Int32),
      @batch_size : Int32,
      @spatial_shape : Tuple(Int32, Int32, Int32),
    )
      validate_geometry!
      unless coordinates.size % COORDINATE_WIDTH == 0
        raise CoordinateMapError.new(
          "sparse coordinate element count must be a multiple of 4"
        )
      end
      @point_count = coordinates.size // COORDINATE_WIDTH
      if @point_count > MAX_POINTS
        raise CoordinateMapBudgetError.new(
          "sparse coordinate point count #{@point_count} exceeds #{MAX_POINTS}"
        )
      end

      @coordinates = Array(Int32).new(coordinates.size) { |index| coordinates[index] }
      @sequence_lengths = Array(Int32).new(@batch_size, 0_i32)
      @batch_broadcast_map = Array(Int32).new(@point_count, 0_i32)
      @lookup = Hash(Int64, Int32).new
      max_x = -1_i32
      max_y = -1_i32
      max_z = -1_i32
      previous_batch = -1_i32

      @point_count.times do |row|
        offset = row * COORDINATE_WIDTH
        batch = @coordinates[offset]
        x = @coordinates[offset + 1]
        y = @coordinates[offset + 2]
        z = @coordinates[offset + 3]
        validate_coordinate!(row, batch, x, y, z)
        if batch < previous_batch
          raise CoordinateMapError.new(
            "sparse coordinate rows must use nondecreasing batch groups"
          )
        end
        previous_batch = batch

        key = encode_key(batch, x, y, z)
        if @lookup.has_key?(key)
          raise CoordinateMapError.new(
            "duplicate sparse coordinate [#{batch}, #{x}, #{y}, #{z}]"
          )
        end
        @lookup[key] = row
        @sequence_lengths[batch] += 1
        @batch_broadcast_map[row] = batch
        max_x = x if x > max_x
        max_y = y if y > max_y
        max_z = z if z > max_z
      end

      @occupied_spatial_shape = if @point_count == 0
                                  {0_i32, 0_i32, 0_i32}
                                else
                                  {max_x + 1, max_y + 1, max_z + 1}
                                end
      @layout = [] of BatchSlice
      @cumulative_sequence_lengths = Array(Int32).new(@batch_size + 1, 0_i32)
      start = 0_i32
      @batch_size.times do |batch|
        stop = start + @sequence_lengths[batch]
        @layout << BatchSlice.new(start, stop)
        @cumulative_sequence_lengths[batch + 1] = stop
        start = stop
      end
      @digest = build_digest
    end

    def layout : Array(BatchSlice)
      @layout.dup
    end

    def sequence_lengths : Array(Int32)
      @sequence_lengths.dup
    end

    def cumulative_sequence_lengths : Array(Int32)
      @cumulative_sequence_lengths.dup
    end

    def batch_broadcast_map : Array(Int32)
      @batch_broadcast_map.dup
    end

    # Kernel authority reads over CoordinateMap3D-owned immutable state. These
    # class methods deliberately bypass virtual getters while exposing neither
    # the coordinate array nor the cached batch-map storage to callers.
    def self.kernel_layout(map : CoordinateMap3D) : Tuple(Int32, Int32)
      {map.@batch_size, map.@point_count}
    end

    def self.kernel_batch_index(map : CoordinateMap3D, row : Int32) : Int32
      unless 0 <= row < map.@point_count
        raise IndexError.new("sparse coordinate row #{row} is out of bounds")
      end
      map.@batch_broadcast_map[row]
    end

    def coordinates_copy : Array(Int32)
      @coordinates.dup
    end

    def coordinate(row : Int32, axis : Int32) : Int32
      unless 0 <= row < @point_count
        raise IndexError.new("sparse coordinate row #{row} is out of bounds")
      end
      unless 0 <= axis < COORDINATE_WIDTH
        raise IndexError.new("sparse coordinate axis #{axis} is out of bounds")
      end
      @coordinates[row * COORDINATE_WIDTH + axis]
    end

    def index_of(batch : Int32, x : Int32, y : Int32, z : Int32) : Int32?
      return nil unless 0 <= batch < @batch_size
      return nil unless coordinate_inside?(x, y, z)
      @lookup[encode_key(batch, x, y, z)]?
    end

    private def validate_geometry! : Nil
      unless 1 <= @batch_size <= MAX_BATCH_SIZE
        raise CoordinateMapError.new(
          "sparse coordinate batch size must be in 1..#{MAX_BATCH_SIZE}"
        )
      end
      @spatial_shape.each do |extent|
        unless extent > 0
          raise CoordinateMapError.new("sparse spatial extent must be positive")
        end
        if extent > MAX_SPATIAL_EXTENT
          raise CoordinateMapError.new(
            "sparse spatial extent must fit the 10-bit 0..1023 coordinate domain"
          )
        end
      end
    end

    private def validate_coordinate!(
      row : Int32,
      batch : Int32,
      x : Int32,
      y : Int32,
      z : Int32,
    ) : Nil
      unless 0 <= batch < @batch_size
        raise CoordinateMapError.new(
          "sparse coordinate row #{row} batch #{batch} is outside range 0...#{@batch_size}"
        )
      end
      { {x, "x", @spatial_shape[0]}, {y, "y", @spatial_shape[1]}, {z, "z", @spatial_shape[2]} }.each do |entry|
        value = entry[0]
        axis = entry[1]
        extent = entry[2]
        if value < 0
          raise CoordinateMapError.new(
            "sparse #{axis} coordinate must be non-negative"
          )
        end
        if value >= extent
          raise CoordinateMapError.new(
            "sparse #{axis} coordinate #{value} exceeds declared extent #{extent}"
          )
        end
      end
    end

    private def coordinate_inside?(x : Int32, y : Int32, z : Int32) : Bool
      0 <= x < @spatial_shape[0] &&
        0 <= y < @spatial_shape[1] &&
        0 <= z < @spatial_shape[2]
    end

    private def encode_key(batch : Int32, x : Int32, y : Int32, z : Int32) : Int64
      # Declared caps bound the maximum key below 2^36, so every operation here
      # is collision-free and far inside Int64 without unchecked wraparound.
      sx = @spatial_shape[0].to_i64
      sy = @spatial_shape[1].to_i64
      sz = @spatial_shape[2].to_i64
      (((batch.to_i64 * sx + x.to_i64) * sy + y.to_i64) * sz + z.to_i64)
    end

    private def build_digest : String
      digest = Digest::SHA256.new
      bytes = Bytes.new(4, 0_u8)
      {@batch_size, @spatial_shape[0], @spatial_shape[1], @spatial_shape[2]}.each do |value|
        IO::ByteFormat::LittleEndian.encode(value, bytes)
        digest.update(bytes)
      end
      @coordinates.each do |value|
        IO::ByteFormat::LittleEndian.encode(value, bytes)
        digest.update(bytes)
      end
      digest.final.hexstring
    end
  end
end
