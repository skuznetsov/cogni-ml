# Shape and stride utilities for tensor operations
# Supports up to 8 dimensions (sufficient for most ML workloads)

module ML
  # Immutable shape descriptor
  struct Shape
    MAX_DIMS = 8

    getter dims : StaticArray(Int32, 8)
    getter ndim : Int32

    def initialize(dimensions : Array(Int32))
      raise ArgumentError.new("Too many dimensions: #{dimensions.size} > #{MAX_DIMS}") if dimensions.size > MAX_DIMS
      raise ArgumentError.new("Shape cannot be empty") if dimensions.empty?

      @ndim = dimensions.size
      @dims = StaticArray(Int32, 8).new(1)
      dimensions.each_with_index { |d, i| @dims[i] = d }
    end

    def initialize(*dimensions : Int32)
      initialize(dimensions.to_a)
    end

    # Total number of elements
    def numel : Int32
      result = 1
      @ndim.times { |i| result *= @dims[i] }
      result
    end

    # Access dimension by index
    def [](index : Int32) : Int32
      raise IndexError.new("Dimension index #{index} out of range for #{@ndim}D tensor") if index < 0 || index >= @ndim
      @dims[index]
    end

    # Iterate over dimensions
    def each(&block : Int32 -> Nil) : Nil
      @ndim.times { |i| yield @dims[i] }
    end

    def each_with_index(&block : Int32, Int32 -> Nil) : Nil
      @ndim.times { |i| yield @dims[i], i }
    end

    # Convert to array
    def to_a : Array(Int32)
      Array(Int32).new(@ndim) { |i| @dims[i] }
    end

    # Shape equality
    def ==(other : Shape) : Bool
      return false unless @ndim == other.ndim
      @ndim.times { |i| return false unless @dims[i] == other.dims[i] }
      true
    end

    # Check if shapes are broadcastable
    def broadcastable_with?(other : Shape) : Bool
      max_ndim = Math.max(@ndim, other.ndim)
      max_ndim.times do |i|
        d1 = i < @ndim ? @dims[@ndim - 1 - i] : 1
        d2 = i < other.ndim ? other.dims[other.ndim - 1 - i] : 1
        return false unless d1 == d2 || d1 == 1 || d2 == 1
      end
      true
    end

    # Compute broadcast result shape
    def broadcast_with(other : Shape) : Shape
      raise ArgumentError.new("Shapes not broadcastable: #{self} and #{other}") unless broadcastable_with?(other)

      max_ndim = Math.max(@ndim, other.ndim)
      result = Array(Int32).new(max_ndim, 1)

      max_ndim.times do |i|
        d1 = i < @ndim ? @dims[@ndim - 1 - i] : 1
        d2 = i < other.ndim ? other.dims[other.ndim - 1 - i] : 1
        result[max_ndim - 1 - i] = Math.max(d1, d2)
      end

      Shape.new(result)
    end

    # Reshape validation
    def can_reshape_to?(new_shape : Shape) : Bool
      numel == new_shape.numel
    end

    # String representation
    def to_s(io : IO) : Nil
      io << "Shape("
      @ndim.times do |i|
        io << ", " if i > 0
        io << @dims[i]
      end
      io << ")"
    end

    def inspect(io : IO) : Nil
      to_s(io)
    end
  end

  # Stride descriptor for memory layout
  struct Strides
    getter strides : StaticArray(Int32, 8)
    getter ndim : Int32

    def initialize(shape : Shape, contiguous : Bool = true)
      @ndim = shape.ndim
      @strides = StaticArray(Int32, 8).new(0)

      if contiguous
        # Row-major (C-style) strides
        @strides[@ndim - 1] = 1
        (@ndim - 2).downto(0) do |i|
          @strides[i] = @strides[i + 1] * shape.dims[i + 1]
        end
      end
    end

    def initialize(strides_arr : Array(Int32))
      raise ArgumentError.new("Too many strides") if strides_arr.size > 8
      @ndim = strides_arr.size
      @strides = StaticArray(Int32, 8).new(0)
      strides_arr.each_with_index { |s, i| @strides[i] = s }
    end

    def [](index : Int32) : Int32
      raise IndexError.new("Stride index #{index} out of range") if index < 0 || index >= @ndim
      @strides[index]
    end

    # Check if memory is contiguous (row-major)
    def contiguous?(shape : Shape) : Bool
      return false unless @ndim == shape.ndim
      expected = 1
      (@ndim - 1).downto(0) do |i|
        return false unless @strides[i] == expected
        expected *= shape.dims[i]
      end
      true
    end

    # Compute flat index from multi-dimensional indices
    def flat_index(indices : Array(Int32)) : Int32
      raise ArgumentError.new("Index count mismatch") unless indices.size == @ndim
      result = 0
      @ndim.times { |i| result += indices[i] * @strides[i] }
      result
    end

    def flat_index(*indices : Int32) : Int32
      flat_index(indices.to_a)
    end

    def to_a : Array(Int32)
      Array(Int32).new(@ndim) { |i| @strides[i] }
    end

    def to_s(io : IO) : Nil
      io << "Strides("
      @ndim.times do |i|
        io << ", " if i > 0
        io << @strides[i]
      end
      io << ")"
    end
  end

  # Utility functions for shape operations
  module ShapeOps
    extend self

    # Compute output shape for matmul: [M, K] @ [K, N] -> [M, N]
    def matmul_shape(a : Shape, b : Shape) : Shape
      raise ArgumentError.new("matmul requires 2D tensors, got #{a.ndim}D and #{b.ndim}D") unless a.ndim == 2 && b.ndim == 2
      raise ArgumentError.new("matmul shape mismatch: #{a}[1]=#{a[1]} != #{b}[0]=#{b[0]}") unless a[1] == b[0]
      Shape.new(a[0], b[1])
    end

    # Compute output shape for batched matmul: [..., M, K] @ [..., K, N] -> [..., M, N]
    def batched_matmul_shape(a : Shape, b : Shape) : Shape
      raise ArgumentError.new("batched_matmul requires at least 2D tensors") unless a.ndim >= 2 && b.ndim >= 2
      raise ArgumentError.new("matmul shape mismatch on inner dims") unless a[-1] == b[-2]

      # Broadcast batch dimensions
      batch_a = a.ndim > 2 ? Shape.new(a.to_a[0...-2]) : Shape.new(1)
      batch_b = b.ndim > 2 ? Shape.new(b.to_a[0...-2]) : Shape.new(1)
      batch_out = batch_a.broadcast_with(batch_b)

      Shape.new(batch_out.to_a + [a[-2], b[-1]])
    end

    # Transpose last two dimensions
    def transpose_shape(shape : Shape) : Shape
      raise ArgumentError.new("transpose requires at least 2D tensor") unless shape.ndim >= 2
      dims = shape.to_a
      dims[-1], dims[-2] = dims[-2], dims[-1]
      Shape.new(dims)
    end

    # Squeeze: remove dimensions of size 1
    def squeeze_shape(shape : Shape, dim : Int32? = nil) : Shape
      if dim
        raise ArgumentError.new("Cannot squeeze dim #{dim}: size is #{shape[dim]}, not 1") unless shape[dim] == 1
        dims = shape.to_a
        dims.delete_at(dim)
        dims = [1] if dims.empty?
        Shape.new(dims)
      else
        dims = shape.to_a.reject { |d| d == 1 }
        dims = [1] if dims.empty?
        Shape.new(dims)
      end
    end

    # Unsqueeze: add dimension of size 1 at position
    def unsqueeze_shape(shape : Shape, dim : Int32) : Shape
      dim = shape.ndim + dim + 1 if dim < 0
      raise ArgumentError.new("Invalid unsqueeze dim #{dim}") if dim < 0 || dim > shape.ndim
      dims = shape.to_a
      dims.insert(dim, 1)
      Shape.new(dims)
    end

    # Flatten shape from start_dim to end_dim
    def flatten_shape(shape : Shape, start_dim : Int32 = 0, end_dim : Int32 = -1) : Shape
      end_dim = shape.ndim + end_dim if end_dim < 0
      raise ArgumentError.new("Invalid flatten dims") if start_dim > end_dim

      before = shape.to_a[0...start_dim]
      middle = shape.to_a[start_dim..end_dim].reduce(1) { |acc, d| acc * d }
      after = shape.to_a[(end_dim + 1)..]

      Shape.new(before + [middle] + after)
    end
  end
end
