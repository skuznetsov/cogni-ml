# Tensor: multi-dimensional array with GPU buffer backend
# NO autograd logic here - that lives in autograd/variable.cr

require "./buffer"
require "./dtype"
require "./shape"

module ML
  # Tensor: shape + strides + data buffer
  # Immutable shape, mutable data
  class Tensor
    getter shape : Shape
    getter strides : Strides
    getter dtype : DType
    getter buffer : MetalBuffer?
    getter cpu_data : Array(Float32)?

    # Device location
    enum Device
      CPU
      GPU
    end

    getter device : Device

    # Default device (compile-time selectable)
    def self.default_device : Device
      {% if flag?(:cpu_only) %}
        Device::CPU
      {% else %}
        Device::GPU
      {% end %}
    end

    # Internal constructor for views (doesn't allocate)
    protected def initialize(
      @shape : Shape,
      @strides : Strides,
      @dtype : DType,
      @device : Device,
      @buffer : MetalBuffer?,
      @cpu_data : Array(Float32)?,
    )
      validate_storage!
    end

    # Create empty tensor on GPU
    def initialize(@shape : Shape, @dtype : DType = DType::F32, @device : Device = Tensor.default_device)
      raise ArgumentError.new("Only F32 dtype is supported for now") unless @dtype.f32?
      @strides = Strides.new(@shape)
      if @device.gpu? && @shape.numel == 0
        raise ArgumentError.new("Cannot allocate a zero-element GPU tensor")
      end

      case @device
      in .gpu?
        byte_size = @shape.numel.to_i64 * @dtype.byte_size
        @buffer = MetalBuffer.new(byte_size)
        @cpu_data = nil
      in .cpu?
        @buffer = nil
        @cpu_data = Array(Float32).new(@shape.numel, 0.0_f32)
      end
      validate_storage!
    end

    # Create from shape tuple
    def self.new(*dims : Int32, dtype : DType = DType::F32, device : Device = Tensor.default_device) : Tensor
      new(Shape.new(dims.to_a), dtype, device)
    end

    # Create from existing data (CPU)
    def self.from_array(data : Array(Float32), shape : Shape) : Tensor
      raise ArgumentError.new("Data size #{data.size} doesn't match shape #{shape.numel}") unless data.size == shape.numel
      # Use protected constructor directly with copied data
      Tensor.new(
        shape,
        Strides.new(shape),
        DType::F32,
        Device::CPU,
        nil,
        data.dup
      )
    end

    # Create from nested array (infers shape)
    def self.from_array(data : Array(Array(Float32))) : Tensor
      rows = data.size
      cols = data.first?.try(&.size) || 0
      flat = data.flatten
      from_array(flat, Shape.new(rows, cols))
    end

    # Factory methods
    def self.zeros(*dims : Int32, device : Device = Tensor.default_device) : Tensor
      tensor = new(*dims, device: device)
      tensor.fill!(0.0_f32)
      tensor
    end

    def self.ones(*dims : Int32, device : Device = Tensor.default_device) : Tensor
      tensor = new(*dims, device: device)
      tensor.fill!(1.0_f32)
      tensor
    end

    def self.full(*dims : Int32, value : Float32, device : Device = Tensor.default_device) : Tensor
      tensor = new(*dims, device: device)
      tensor.fill!(value)
      tensor
    end

    def self.rand(*dims : Int32, device : Device = Tensor.default_device) : Tensor
      tensor = new(*dims, device: Device::CPU)
      tensor.cpu_data.not_nil!.map_with_index! { |_, _| Random.rand.to_f32 }
      device.gpu? ? tensor.to_gpu : tensor
    end

    def self.randn(*dims : Int32, device : Device = Tensor.default_device) : Tensor
      # Box-Muller transform for normal distribution
      tensor = new(*dims, device: Device::CPU)
      data = tensor.cpu_data.not_nil!
      i = 0
      while i < data.size
        u1 = Random.rand.to_f32
        u2 = Random.rand.to_f32
        u1 = 1e-10_f32 if u1 < 1e-10_f32 # Avoid log(0)
        mag = Math.sqrt(-2.0_f32 * Math.log(u1))
        z0 = mag * Math.cos(2.0_f32 * Math::PI * u2)
        z1 = mag * Math.sin(2.0_f32 * Math::PI * u2)
        data[i] = z0.to_f32
        data[i + 1] = z1.to_f32 if i + 1 < data.size
        i += 2
      end
      device.gpu? ? tensor.to_gpu : tensor
    end

    # Identity matrix
    def self.eye(n : Int32, device : Device = Tensor.default_device) : Tensor
      tensor = zeros(n, n, device: Device::CPU)
      data = tensor.cpu_data.not_nil!
      n.times { |i| data[i * n + i] = 1.0_f32 }
      device.gpu? ? tensor.to_gpu : tensor
    end

    # Arange
    def self.arange(start : Float32, stop : Float32, step : Float32 = 1.0_f32, device : Device = Tensor.default_device) : Tensor
      unless start.finite? && stop.finite?
        raise ArgumentError.new("arange start and stop must be finite")
      end
      unless step.finite? && step != 0.0_f32
        raise ArgumentError.new("arange step must be finite and non-zero")
      end

      count_f64 = if (step > 0.0_f32 && start < stop) || (step < 0.0_f32 && start > stop)
                    ((stop.to_f64 - start.to_f64) / step.to_f64).ceil
                  else
                    0.0
                  end
      unless count_f64 <= Int32::MAX
        raise ArgumentError.new("arange element count overflow: #{count_f64}")
      end
      count = count_f64.to_i32
      tensor = new(count, device: Device::CPU)
      data = tensor.cpu_data.not_nil!
      count.times do |i|
        data[i] = (start.to_f64 + i.to_f64 * step.to_f64).to_f32
      end
      device.gpu? ? tensor.to_gpu : tensor
    end

    # Linspace
    def self.linspace(start : Float32, stop : Float32, count : Int32, device : Device = Tensor.default_device) : Tensor
      unless start.finite? && stop.finite?
        raise ArgumentError.new("linspace start and stop must be finite")
      end
      if count < 0
        raise ArgumentError.new("linspace count must be non-negative")
      end

      tensor = new(count, device: Device::CPU)
      data = tensor.cpu_data.not_nil!
      if count == 1
        data[0] = start
      elsif count > 1
        step = (stop.to_f64 - start.to_f64) / (count - 1).to_f64
        count.times do |i|
          data[i] = (start.to_f64 + i.to_f64 * step).to_f32
        end
        data[-1] = stop
      end
      device.gpu? ? tensor.to_gpu : tensor
    end

    # Properties
    def numel : Int32
      @shape.numel
    end

    def ndim : Int32
      @shape.ndim
    end

    def contiguous? : Bool
      @strides.contiguous?(@shape)
    end

    def on_gpu? : Bool
      @device.gpu?
    end

    def on_cpu? : Bool
      @device.cpu?
    end

    # Data access
    def to_a : Array(Float32)
      result = Array(Float32).new(@shape.numel, 0.0_f32)
      copy_logical_cpu_data_to!(result)
      result
    end

    def to_flat_array : Array(Float32)
      to_a
    end

    # Fill with value
    def fill!(value : Float32) : self
      case @device
      in .cpu?
        @cpu_data.not_nil!.fill(value)
      in .gpu?
        # TODO: GPU fill kernel
        # For now, use CPU then transfer
        temp = Array(Float32).new(@shape.numel, value)
        @buffer.not_nil!.write(temp)
      end
      self
    end

    # Device transfer
    def to_gpu : Tensor
      return self if @device.gpu?

      gpu_tensor = Tensor.new(@shape, @dtype, Device::GPU)
      gpu_tensor.buffer.not_nil!.write(@cpu_data.not_nil!)
      gpu_tensor
    end

    def to_cpu : Tensor
      return self if @device.cpu?

      # Create new CPU tensor and copy data from GPU buffer
      buf = @buffer.not_nil!
      data = buf.read(@shape.numel)

      Tensor.new(
        @shape,
        Strides.new(@shape),
        @dtype,
        Device::CPU,
        nil,
        data
      )
    end

    def to_cpu! : self
      return self if @device.cpu?

      @cpu_data = @buffer.not_nil!.read(@shape.numel)
      @buffer.not_nil!.release
      @buffer = nil
      @device = Device::CPU
      validate_storage!
      self
    end

    def to_gpu! : self
      return self if @device.gpu?
      if @shape.numel == 0
        raise ArgumentError.new("Cannot allocate a zero-element GPU tensor")
      end

      byte_size = @shape.numel.to_i64 * @dtype.byte_size
      @buffer = MetalBuffer.new(byte_size)
      @buffer.not_nil!.write(@cpu_data.not_nil!)
      @cpu_data = nil
      @device = Device::GPU
      validate_storage!
      self
    end

    # Safe CPU data access - converts from GPU if needed, handles empty tensors
    # Returns empty array for empty tensors instead of crashing
    def safe_cpu_data : Array(Float32)
      # Empty tensor case
      return Array(Float32).new if @shape.numel == 0

      if @device.gpu?
        # GPU tensor - read from buffer
        if buf = @buffer
          buf.read(@shape.numel)
        else
          raise "GPU tensor has no buffer allocated"
        end
      else
        # CPU tensor - return data
        if data = @cpu_data
          data
        else
          raise "CPU tensor has no data allocated"
        end
      end
    end

    # Ensure tensor is on CPU for data access
    private def ensure_cpu! : Nil
      if @device.gpu?
        @cpu_data = @buffer.not_nil!.read(@shape.numel)
      end
    end

    # Element access (for debugging, copies to CPU if needed)
    def [](indices : Array(Int32)) : Float32
      flat_idx = checked_flat_index(indices)
      ensure_cpu!
      @cpu_data.not_nil![flat_idx]
    end

    def [](*indices : Int32) : Float32
      self[indices.to_a]
    end

    def []=(indices : Array(Int32), value : Float32) : Float32
      flat_idx = checked_flat_index(indices)
      ensure_cpu!
      @cpu_data.not_nil![flat_idx] = value
      # Mark as dirty if on GPU
      if @device.gpu?
        @buffer.not_nil!.write(@cpu_data.not_nil!)
      end
      value
    end

    def []=(*indices_and_value) : Float32
      indices = indices_and_value[0...-1].map(&.as(Int32)).to_a
      value = indices_and_value[-1].as(Float32)
      self[indices] = value
    end

    # Reshape (returns view if contiguous, copy otherwise)
    def reshape(new_shape : Shape) : Tensor
      raise ArgumentError.new("Cannot reshape #{@shape} to #{new_shape}: element count mismatch") unless @shape.numel == new_shape.numel

      if contiguous?
        Tensor.new(
          new_shape,
          Strides.new(new_shape),
          @dtype,
          @device,
          @buffer,
          @cpu_data
        )
      else
        # Need to copy to make contiguous
        contiguous_copy.reshape(new_shape)
      end
    end

    def reshape(*dims : Int32) : Tensor
      reshape(Shape.new(dims.to_a))
    end

    # Make contiguous copy
    def contiguous : Tensor
      return self if contiguous?
      contiguous_copy
    end

    private def contiguous_copy : Tensor
      result = Tensor.new(@shape, @dtype, @device)
      if @device.gpu?
        logical_data = Array(Float32).new(@shape.numel, 0.0_f32)
        copy_logical_cpu_data_to!(logical_data)
        result.buffer.not_nil!.write(logical_data)
      else
        copy_logical_cpu_data_to!(result.cpu_data.not_nil!)
      end
      result
    end

    # Transpose (swap last two dims)
    def transpose : Tensor
      raise ArgumentError.new("transpose requires at least 2D tensor") unless @shape.ndim >= 2

      new_shape = ShapeOps.transpose_shape(@shape)
      new_strides_arr = @strides.to_a
      new_strides_arr[-1], new_strides_arr[-2] = new_strides_arr[-2], new_strides_arr[-1]

      Tensor.new(
        new_shape,
        Strides.new(new_strides_arr),
        @dtype,
        @device,
        @buffer,
        @cpu_data
      )
    end

    def t : Tensor
      transpose
    end

    # Squeeze / Unsqueeze
    def squeeze(dim : Int32? = nil) : Tensor
      new_shape = ShapeOps.squeeze_shape(@shape, dim)
      reshape(new_shape)
    end

    def unsqueeze(dim : Int32) : Tensor
      new_shape = ShapeOps.unsqueeze_shape(@shape, dim)
      reshape(new_shape)
    end

    # Flatten
    def flatten(start_dim : Int32 = 0, end_dim : Int32 = -1) : Tensor
      new_shape = ShapeOps.flatten_shape(@shape, start_dim, end_dim)
      reshape(new_shape)
    end

    # Clone (deep copy)
    def clone : Tensor
      return contiguous_copy unless contiguous?

      result = Tensor.new(@shape, @dtype, @device)
      case @device
      in .cpu?
        src = @cpu_data.not_nil!
        dst = result.cpu_data.not_nil!
        @shape.numel.times { |i| dst[i] = src[i] }
      in .gpu?
        # Copy via CPU for now
        # TODO: GPU memcpy kernel
        data = @buffer.not_nil!.read(@shape.numel)
        result.buffer.not_nil!.write(data)
      end
      result
    end

    # Gather the logical row-major sequence from the current dense layout.
    private def copy_logical_cpu_data_to!(result : Array(Float32)) : Nil
      ensure_cpu!
      src = @cpu_data.not_nil!
      unless result.size == @shape.numel
        raise ArgumentError.new(
          "Logical copy destination length #{result.size} doesn't match shape #{@shape.numel}"
        )
      end
      if contiguous?
        @shape.numel.times { |i| result[i] = src[i] }
        return
      end

      @shape.numel.times do |logical_index|
        remaining = logical_index
        storage_index = 0_i64
        axis = @shape.ndim - 1
        while axis >= 0
          dimension = @shape[axis]
          coordinate = remaining % dimension
          remaining //= dimension
          storage_index += coordinate.to_i64 * @strides[axis]
          axis -= 1
        end
        result[logical_index] = src[storage_index.to_i]
      end
    end

    # String representation
    def to_s(io : IO) : Nil
      io << "Tensor(shape=#{@shape}, dtype=#{@dtype}, device=#{@device})"
    end

    def inspect(io : IO) : Nil
      ensure_cpu!
      io << "Tensor(\n"
      print_recursive(io, 0, 0, "  ")
      io << ", shape=#{@shape}, dtype=#{@dtype})"
    end

    private def print_recursive(io : IO, dim : Int32, offset : Int32, indent : String) : Int32
      if dim == @shape.ndim - 1
        # Last dimension: print elements
        io << "["
        @shape[dim].times do |i|
          io << ", " if i > 0
          val = @cpu_data.not_nil![offset + i * @strides[dim]]
          io << sprintf("%.4f", val)
        end
        io << "]"
        offset + @shape[dim] * @strides[dim]
      else
        io << "["
        new_offset = offset
        @shape[dim].times do |i|
          io << ",\n#{indent} " if i > 0
          new_offset = print_recursive(io, dim + 1, new_offset, indent + " ")
        end
        io << "]"
        new_offset
      end
    end

    # Get underlying buffer handle for kernel dispatch
    def buffer_handle : Pointer(Void)
      raise "Tensor not on GPU" unless @device.gpu?
      @buffer.not_nil!.handle
    end

    # Get raw data pointer (CPU only)
    def data_ptr : Pointer(Float32)
      raise "Tensor not on CPU" unless @device.cpu?
      @cpu_data.not_nil!.to_unsafe
    end

    private def validate_storage! : Nil
      unless @dtype.f32?
        raise ArgumentError.new("Only F32 dtype is supported for now")
      end
      unless @strides.ndim == @shape.ndim
        raise ArgumentError.new(
          "Tensor stride rank #{@strides.ndim} doesn't match shape rank #{@shape.ndim}"
        )
      end
      validate_dense_strides!

      case @device
      in .cpu?
        raise ArgumentError.new("CPU tensor cannot own a GPU buffer") if @buffer
        data = @cpu_data
        raise ArgumentError.new("CPU tensor requires data") unless data
        unless data.size == @shape.numel
          raise ArgumentError.new(
            "CPU tensor data length #{data.size} doesn't match shape #{@shape.numel}"
          )
        end
      in .gpu?
        if @shape.numel == 0
          raise ArgumentError.new("Cannot allocate a zero-element GPU tensor")
        end
        buffer = @buffer
        raise ArgumentError.new("GPU tensor requires a buffer") unless buffer
        required_bytes = @shape.numel.to_i64 * sizeof(Float32)
        if buffer.size < required_bytes
          raise ArgumentError.new(
            "GPU tensor buffer length #{buffer.size} is smaller than #{required_bytes}"
          )
        end
        if data = @cpu_data
          unless data.size == @shape.numel
            raise ArgumentError.new(
              "GPU tensor CPU cache length #{data.size} doesn't match shape #{@shape.numel}"
            )
          end
        end
      end
    end

    private def validate_dense_strides! : Nil
      @shape.ndim.times do |axis|
        if @strides[axis] < 0
          raise ArgumentError.new(
            "Tensor strides must describe a dense non-overlapping layout"
          )
        end
      end
      return if @shape.numel == 0

      axes = Array(Int32).new
      @shape.ndim.times do |axis|
        axes << axis if @shape[axis] > 1
      end
      axes.sort_by! { |axis| @strides[axis] }

      expected_stride = 1_i64
      axes.each do |axis|
        unless @strides[axis].to_i64 == expected_stride
          raise ArgumentError.new(
            "Tensor strides must describe a dense non-overlapping layout"
          )
        end
        expected_stride *= @shape[axis]
      end
      unless expected_stride == @shape.numel
        raise ArgumentError.new(
          "Tensor strides must describe a dense non-overlapping layout"
        )
      end
    end

    private def checked_flat_index(indices : Array(Int32)) : Int32
      unless indices.size == @shape.ndim
        raise ArgumentError.new(
          "Index count #{indices.size} doesn't match tensor rank #{@shape.ndim}"
        )
      end

      flat_index = 0_i64
      @shape.ndim.times do |axis|
        index = indices[axis]
        dimension = @shape[axis]
        unless 0 <= index < dimension
          raise IndexError.new(
            "Tensor index #{index} is outside dimension #{axis} of size #{dimension}"
          )
        end
        flat_index += index.to_i64 * @strides[axis]
      end
      if flat_index > Int32::MAX
        raise IndexError.new("Tensor flat index overflow: #{flat_index}")
      end
      flat_index.to_i32
    end
  end
end
