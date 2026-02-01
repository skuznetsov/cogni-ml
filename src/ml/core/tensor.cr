# Tensor: multi-dimensional array with GPU buffer backend
# NO autograd logic here - that lives in autograd/variable.cr

require "./buffer"
require "./shape"

module ML
  # Data type for tensor elements
  enum DType
    F32
    F16 # Future support
    I32
    I64
    U8  # For Bool

    def byte_size : ::Int32
      case self
      in .f32? then 4
      in .f16? then 2
      in .i32? then 4
      in .i64? then 8
      in .u8?  then 1
      end
    end
  end

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

    # Internal constructor for views (doesn't allocate)
    protected def initialize(
      @shape : Shape,
      @strides : Strides,
      @dtype : DType,
      @device : Device,
      @buffer : MetalBuffer?,
      @cpu_data : Array(Float32)?
    )
    end

    # Create empty tensor on GPU
    def initialize(@shape : Shape, @dtype : DType = DType::F32, @device : Device = Device::GPU)
      @strides = Strides.new(@shape)

      case @device
      in .gpu?
        byte_size = @shape.numel.to_i64 * @dtype.byte_size
        @buffer = MetalBuffer.new(byte_size)
        @cpu_data = nil
      in .cpu?
        @buffer = nil
        @cpu_data = Array(Float32).new(@shape.numel, 0.0_f32)
      end
    end

    # Create from shape tuple
    def self.new(*dims : Int32, dtype : DType = DType::F32, device : Device = Device::GPU) : Tensor
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
    def self.zeros(*dims : Int32, device : Device = Device::GPU) : Tensor
      tensor = new(*dims, device: device)
      tensor.fill!(0.0_f32)
      tensor
    end

    def self.ones(*dims : Int32, device : Device = Device::GPU) : Tensor
      tensor = new(*dims, device: device)
      tensor.fill!(1.0_f32)
      tensor
    end

    def self.full(*dims : Int32, value : Float32, device : Device = Device::GPU) : Tensor
      tensor = new(*dims, device: device)
      tensor.fill!(value)
      tensor
    end

    def self.rand(*dims : Int32, device : Device = Device::GPU) : Tensor
      tensor = new(*dims, device: Device::CPU)
      tensor.cpu_data.not_nil!.map_with_index! { |_, _| Random.rand.to_f32 }
      device.gpu? ? tensor.to_gpu : tensor
    end

    def self.randn(*dims : Int32, device : Device = Device::GPU) : Tensor
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
    def self.eye(n : Int32, device : Device = Device::GPU) : Tensor
      tensor = zeros(n, n, device: Device::CPU)
      data = tensor.cpu_data.not_nil!
      n.times { |i| data[i * n + i] = 1.0_f32 }
      device.gpu? ? tensor.to_gpu : tensor
    end

    # Arange
    def self.arange(start : Float32, stop : Float32, step : Float32 = 1.0_f32, device : Device = Device::GPU) : Tensor
      count = ((stop - start) / step).ceil.to_i32
      tensor = new(count, device: Device::CPU)
      data = tensor.cpu_data.not_nil!
      count.times { |i| data[i] = start + i * step }
      device.gpu? ? tensor.to_gpu : tensor
    end

    # Linspace
    def self.linspace(start : Float32, stop : Float32, count : Int32, device : Device = Device::GPU) : Tensor
      tensor = new(count, device: Device::CPU)
      data = tensor.cpu_data.not_nil!
      step = (stop - start) / (count - 1).to_f32
      count.times { |i| data[i] = start + i * step }
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
      ensure_cpu!
      @cpu_data.not_nil!.dup
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
      self
    end

    def to_gpu! : self
      return self if @device.gpu?

      byte_size = @shape.numel.to_i64 * @dtype.byte_size
      @buffer = MetalBuffer.new(byte_size)
      @buffer.not_nil!.write(@cpu_data.not_nil!)
      @cpu_data = nil
      @device = Device::GPU
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
      ensure_cpu!
      flat_idx = @strides.flat_index(indices)
      @cpu_data.not_nil![flat_idx]
    end

    def [](*indices : Int32) : Float32
      self[indices.to_a]
    end

    def []=(indices : Array(Int32), value : Float32) : Float32
      ensure_cpu!
      flat_idx = @strides.flat_index(indices)
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
        # Create view with new shape
        view = Tensor.allocate
        view.initialize_as_view(self, new_shape)
        view
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
      # TODO: implement strided copy
      # For now, go through CPU
      ensure_cpu!
      if @device.gpu?
        result.buffer.not_nil!.write(@cpu_data.not_nil!)
      else
        src = @cpu_data.not_nil!
        dst = result.cpu_data.not_nil!
        @shape.numel.times { |i| dst[i] = src[i] }
      end
      result
    end

    # View initialization (internal)
    protected def initialize_as_view(source : Tensor, new_shape : Shape)
      @shape = new_shape
      @strides = Strides.new(new_shape)
      @dtype = source.dtype
      @device = source.device
      @buffer = source.buffer
      @cpu_data = source.cpu_data
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
  end
end
