module ML::ThreeD::Trellis2
  class LayoutError < Exception
  end

  struct RawTensor
    MAX_BYTES = 64_i64 * 1024_i64 * 1024_i64

    getter name : String
    getter dtype : String
    getter shape : Array(Int64)
    getter bytes : Bytes

    def initialize(
      name : String,
      dtype : String,
      shape : Array(Int64),
      bytes : Bytes,
    )
      @name = name
      @dtype = dtype
      @shape = shape.dup
      @bytes = bytes.dup
      validate!
    end

    def bytes_per_element : Int32
      case @dtype
      when "BF16", "F16" then 2
      when "F32"         then 4
      else
        raise LayoutError.new("unsupported dtype #{@dtype.inspect}")
      end
    end

    def n_elements : Int64
      checked_elements(@shape)
    end

    private def validate! : Nil
      raise LayoutError.new("tensor name must not be empty") if @name.empty?
      elements = n_elements
      element_bytes = bytes_per_element.to_i64
      if elements > Int64::MAX // element_bytes
        raise LayoutError.new("tensor #{@name.inspect} shape byte size overflow")
      end
      expected = elements * element_bytes
      if expected > MAX_BYTES
        raise LayoutError.new(
          "tensor #{@name.inspect} exceeds T2N1 byte limit #{MAX_BYTES}"
        )
      end
      unless @bytes.size.to_i64 == expected
        raise LayoutError.new(
          "tensor #{@name.inspect} has #{@bytes.size} bytes; expected #{expected}"
        )
      end
    end

    private def checked_elements(shape : Array(Int64)) : Int64
      shape.each do |dimension|
        if dimension < 0
          raise LayoutError.new(
            "tensor #{@name.inspect} has negative shape dimension #{dimension}"
          )
        end
      end
      return 0_i64 if shape.any?(&.zero?)

      shape.reduce(1_i64) do |product, dimension|
        if product > Int64::MAX // dimension
          raise LayoutError.new("tensor #{@name.inspect} shape overflow")
        end
        product * dimension
      end
    end
  end

  module Layout
    extend self

    def identity(source : RawTensor, destination : String) : RawTensor
      RawTensor.new(destination, source.dtype, source.shape, source.bytes)
    end

    def transpose(
      source : RawTensor,
      destination : String,
      axis_a : Int32,
      axis_b : Int32,
    ) : RawTensor
      rank = source.shape.size
      a = checked_axis(axis_a, rank, "transpose")
      b = checked_axis(axis_b, rank, "transpose")
      if a == b
        raise LayoutError.new("transpose axes must be distinct")
      end
      axes = (0...rank).to_a
      axes[a], axes[b] = axes[b], axes[a]
      permute(source, destination, axes)
    end

    def permute(
      source : RawTensor,
      destination : String,
      axes : Array(Int32),
    ) : RawTensor
      rank = source.shape.size
      unless axes.size == rank &&
             axes.all? { |axis| 0 <= axis < rank } &&
             axes.to_set.size == rank
        raise LayoutError.new(
          "permutation must contain each source axis exactly once"
        )
      end

      output_shape = axes.map { |axis| source.shape[axis] }
      output_bytes = Bytes.new(source.bytes.size)
      return RawTensor.new(
        destination,
        source.dtype,
        output_shape,
        output_bytes
      ) if source.n_elements == 0

      input_strides = strides(source.shape)
      element_bytes = source.bytes_per_element
      coordinates = Array(Int64).new(rank, 0_i64)
      source_coordinates = Array(Int64).new(rank, 0_i64)

      source.n_elements.to_i.times do |output_index|
        decode_coordinates(output_index.to_i64, output_shape, coordinates)
        rank.times do |output_axis|
          source_coordinates[axes[output_axis]] = coordinates[output_axis]
        end
        source_index = linear_index(source_coordinates, input_strides)
        copy_element!(
          source.bytes,
          source_index,
          output_bytes,
          output_index.to_i64,
          element_bytes
        )
      end

      RawTensor.new(destination, source.dtype, output_shape, output_bytes)
    end

    def split(
      source : RawTensor,
      destinations : Array(String),
      axis : Int32,
      sizes : Array(Int64),
    ) : Array(RawTensor)
      normalized_axis = checked_axis(axis, source.shape.size, "split")
      unless !destinations.empty? &&
             destinations.size == sizes.size &&
             sizes.all? { |size| size >= 0 } &&
             checked_sum(sizes, "split sizes") == source.shape[normalized_axis]
        raise LayoutError.new(
          "split sizes and destinations must exactly partition the axis"
        )
      end
      if destinations.to_set.size != destinations.size
        raise LayoutError.new("split destinations must be unique")
      end

      start = 0_i64
      destinations.zip(sizes).map do |destination, size|
        result = slice(source, destination, normalized_axis, start, size)
        start += size
        result
      end
    end

    def concat(
      sources : Array(RawTensor),
      destination : String,
      axis : Int32,
    ) : RawTensor
      raise LayoutError.new("concat requires at least one source") if sources.empty?
      first = sources.first
      rank = first.shape.size
      normalized_axis = checked_axis(axis, rank, "concat")

      sources.each do |source|
        unless source.dtype == first.dtype
          raise LayoutError.new("concat sources must have the same dtype")
        end
        unless source.shape.size == rank
          raise LayoutError.new("concat sources must have the same rank")
        end
        rank.times do |dimension|
          next if dimension == normalized_axis
          unless source.shape[dimension] == first.shape[dimension]
            raise LayoutError.new(
              "concat source shape differs outside the concat axis"
            )
          end
        end
      end

      output_shape = first.shape.dup
      output_shape[normalized_axis] = checked_sum(
        sources.map { |source| source.shape[normalized_axis] },
        "concat axis"
      )
      output_elements = checked_elements(output_shape, "concat output")
      element_bytes = first.bytes_per_element
      if output_elements > RawTensor::MAX_BYTES // element_bytes
        raise LayoutError.new(
          "concat output exceeds T2N1 byte limit #{RawTensor::MAX_BYTES}"
        )
      end
      output_bytes = Bytes.new((output_elements * element_bytes).to_i)
      return RawTensor.new(
        destination,
        first.dtype,
        output_shape,
        output_bytes
      ) if output_elements == 0

      output_coordinates = Array(Int64).new(rank, 0_i64)
      source_coordinates = Array(Int64).new(rank, 0_i64)
      source_strides = sources.map { |source| strides(source.shape) }

      output_elements.to_i.times do |output_index|
        decode_coordinates(
          output_index.to_i64,
          output_shape,
          output_coordinates
        )
        axis_coordinate = output_coordinates[normalized_axis]
        source_index = 0
        prefix = 0_i64
        sources.each_with_index do |source, index|
          width = source.shape[normalized_axis]
          if axis_coordinate < prefix + width
            source_index = index
            break
          end
          prefix += width
        end

        rank.times { |dimension| source_coordinates[dimension] = output_coordinates[dimension] }
        source_coordinates[normalized_axis] = axis_coordinate - prefix
        input_index = linear_index(
          source_coordinates,
          source_strides[source_index]
        )
        copy_element!(
          sources[source_index].bytes,
          input_index,
          output_bytes,
          output_index.to_i64,
          element_bytes
        )
      end

      RawTensor.new(destination, first.dtype, output_shape, output_bytes)
    end

    private def slice(
      source : RawTensor,
      destination : String,
      axis : Int32,
      start : Int64,
      length : Int64,
    ) : RawTensor
      output_shape = source.shape.dup
      output_shape[axis] = length
      output_elements = checked_elements(output_shape, "split output")
      element_bytes = source.bytes_per_element
      output_bytes = Bytes.new((output_elements * element_bytes).to_i)
      return RawTensor.new(
        destination,
        source.dtype,
        output_shape,
        output_bytes
      ) if output_elements == 0

      input_strides = strides(source.shape)
      output_coordinates = Array(Int64).new(output_shape.size, 0_i64)
      source_coordinates = Array(Int64).new(output_shape.size, 0_i64)
      output_elements.to_i.times do |output_index|
        decode_coordinates(
          output_index.to_i64,
          output_shape,
          output_coordinates
        )
        output_shape.size.times do |dimension|
          source_coordinates[dimension] = output_coordinates[dimension]
        end
        source_coordinates[axis] += start
        input_index = linear_index(source_coordinates, input_strides)
        copy_element!(
          source.bytes,
          input_index,
          output_bytes,
          output_index.to_i64,
          element_bytes
        )
      end

      RawTensor.new(destination, source.dtype, output_shape, output_bytes)
    end

    private def checked_axis(axis : Int32, rank : Int32, context : String) : Int32
      unless 0 <= axis < rank
        raise LayoutError.new(
          "#{context} axis #{axis} is outside rank #{rank}"
        )
      end
      axis
    end

    private def checked_sum(values : Array(Int64), context : String) : Int64
      values.reduce(0_i64) do |sum, value|
        if value > Int64::MAX - sum
          raise LayoutError.new("#{context} overflow")
        end
        sum + value
      end
    end

    private def checked_elements(shape : Array(Int64), context : String) : Int64
      return 0_i64 if shape.any?(&.zero?)
      shape.reduce(1_i64) do |product, dimension|
        if dimension < 0 || product > Int64::MAX // dimension
          raise LayoutError.new("#{context} shape overflow")
        end
        product * dimension
      end
    end

    private def strides(shape : Array(Int64)) : Array(Int64)
      result = Array(Int64).new(shape.size, 1_i64)
      if shape.size > 1
        (shape.size - 2).downto(0) do |index|
          result[index] = result[index + 1] * shape[index + 1]
        end
      end
      result
    end

    private def decode_coordinates(
      linear : Int64,
      shape : Array(Int64),
      coordinates : Array(Int64),
    ) : Nil
      remaining = linear
      (shape.size - 1).downto(0) do |axis|
        dimension = shape[axis]
        coordinates[axis] = remaining % dimension
        remaining //= dimension
      end
    end

    private def linear_index(
      coordinates : Array(Int64),
      strides : Array(Int64),
    ) : Int64
      coordinates.each_with_index.reduce(0_i64) do |index, (coordinate, axis)|
        index + coordinate * strides[axis]
      end
    end

    private def copy_element!(
      source : Bytes,
      source_index : Int64,
      destination : Bytes,
      destination_index : Int64,
      element_bytes : Int32,
    ) : Nil
      source_offset = (source_index * element_bytes).to_i
      destination_offset = (destination_index * element_bytes).to_i
      destination[destination_offset, element_bytes].copy_from(
        source[source_offset, element_bytes]
      )
    end
  end
end
