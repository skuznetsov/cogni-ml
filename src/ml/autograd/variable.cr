# Variable: Tensor with automatic gradient tracking
# Wraps Tensor and builds computation graph for backprop

require "../core/tensor"
require "./grad_fn"

module ML
  module Autograd
    # Variable wraps a Tensor and tracks gradients
    class Variable
      getter data : Tensor
      property grad : Tensor?
      property grad_fn : GradFn?
      getter? requires_grad : Bool
      getter? is_leaf : Bool

      # Protected setter for is_leaf (used by operations)
      protected def is_leaf=(value : Bool)
        @is_leaf = value
      end

      # Allow toggling grad tracking (useful for inference)
      def requires_grad=(value : Bool)
        @requires_grad = value
        if !value
          @grad = nil
          @grad_fn = nil
        end
      end

      # Create variable from tensor
      def initialize(@data : Tensor, @requires_grad : Bool = false)
        @grad = nil
        @grad_fn = nil
        @is_leaf = true  # Leaf until used in an operation
      end

      # Create from existing data
      def self.from_tensor(tensor : Tensor, requires_grad : Bool = false) : Variable
        new(tensor, requires_grad)
      end

      # Factory methods that mirror Tensor
      def self.zeros(*dims : Int32, requires_grad : Bool = false, device : Tensor::Device = Tensor::Device::GPU) : Variable
        new(Tensor.zeros(*dims, device: device), requires_grad)
      end

      def self.ones(*dims : Int32, requires_grad : Bool = false, device : Tensor::Device = Tensor::Device::GPU) : Variable
        new(Tensor.ones(*dims, device: device), requires_grad)
      end

      def self.rand(*dims : Int32, requires_grad : Bool = false, device : Tensor::Device = Tensor::Device::GPU) : Variable
        new(Tensor.rand(*dims, device: device), requires_grad)
      end

      def self.randn(*dims : Int32, requires_grad : Bool = false, device : Tensor::Device = Tensor::Device::GPU) : Variable
        new(Tensor.randn(*dims, device: device), requires_grad)
      end

      # Shape delegation
      def shape : Shape
        @data.shape
      end

      def numel : Int32
        @data.numel
      end

      def ndim : Int32
        @data.ndim
      end

      # Detach from computation graph (returns new variable with same data, no grad tracking)
      def detach : Variable
        v = Variable.new(@data, false)
        v
      end

      # Clone with gradient tracking
      def clone : Variable
        v = Variable.new(@data.clone, @requires_grad)
        v
      end

      # Transpose (swap last two dims)
      def transpose : Variable
        Variable.new(@data.transpose, @requires_grad)
      end

      def t : Variable
        transpose
      end

      # Zero gradient
      def zero_grad! : Nil
        @grad = nil
      end

      # Backward pass - compute gradients
      def backward(grad_output : Tensor? = nil) : Nil
        raise "Cannot call backward on non-scalar without grad_output" if grad_output.nil? && @data.numel != 1
        raise "backward() requires requires_grad=true" unless @requires_grad

        # Default gradient is 1 for scalar
        upstream_grad = grad_output || Tensor.ones(1, device: @data.device)

        # Build topological order
        topo_order = Array(Variable).new
        visited = Set(UInt64).new

        build_topo(self, topo_order, visited)

        # Initialize gradient for this variable
        @grad = upstream_grad.clone

        # Backpropagate in reverse topological order
        topo_order.reverse_each do |var|
          next unless var.grad_fn
          next unless var.grad

          # Compute gradients for inputs
          input_grads = var.grad_fn.not_nil!.backward(var.grad.not_nil!)

          # Accumulate gradients to inputs
          var.grad_fn.not_nil!.inputs.each_with_index do |input_var, i|
            next unless input_var.requires_grad?
            next unless input_grads[i]?

            input_grad = input_grads[i]
            next unless input_grad

            if input_var.grad
              # Accumulate
              input_var.grad = add_tensors(input_var.grad.not_nil!, input_grad)
            else
              input_var.grad = input_grad.clone
            end
          end
        end
      end

      private def build_topo(var : Variable, order : Array(Variable), visited : Set(UInt64)) : Nil
        id = var.object_id
        return if visited.includes?(id)
        visited.add(id)

        if gf = var.grad_fn
          gf.inputs.each do |input|
            build_topo(input, order, visited)
          end
        end

        order << var
      end

      private def add_tensors(a : Tensor, b : Tensor) : Tensor
        # CPU fallback
        a_cpu = a.on_cpu? ? a : a.to_cpu
        b_cpu = b.on_cpu? ? b : b.to_cpu

        result = Tensor.new(a.shape, a.dtype, Tensor::Device::CPU)
        a_data = a_cpu.cpu_data.not_nil!
        b_data = b_cpu.cpu_data.not_nil!
        result_data = result.cpu_data.not_nil!

        a.numel.times { |i| result_data[i] = a_data[i] + b_data[i] }

        a.on_gpu? ? result.to_gpu : result
      end

      # ========================================================================
      # Operations that build computation graph
      # ========================================================================

      # Addition
      def +(other : Variable) : Variable
        result_data = add_tensors(@data, other.data)
        result = Variable.new(result_data, @requires_grad || other.requires_grad?)

        if result.requires_grad?
          result.is_leaf = false
          grad_fn = AddBackward.new
          grad_fn.inputs = [self, other]
          result.grad_fn = grad_fn
        end

        result
      end

      def +(scalar : Float32) : Variable
        result_data = @data.clone
        result_data.to_cpu!
        result_data.cpu_data.not_nil!.map! { |x| x + scalar }
        result_data.to_gpu! if @data.on_gpu?

        result = Variable.new(result_data, @requires_grad)

        if result.requires_grad?
          result.is_leaf = false
          grad_fn = AddBackward.new
          grad_fn.inputs = [self]  # scalar doesn't need grad
          result.grad_fn = grad_fn
        end

        result
      end

      # Subtraction
      def -(other : Variable) : Variable
        result_data = sub_tensors(@data, other.data)
        result = Variable.new(result_data, @requires_grad || other.requires_grad?)

        if result.requires_grad?
          result.is_leaf = false
          grad_fn = SubBackward.new
          grad_fn.inputs = [self, other]
          result.grad_fn = grad_fn
        end

        result
      end

      private def sub_tensors(a : Tensor, b : Tensor) : Tensor
        a_cpu = a.on_cpu? ? a : a.to_cpu
        b_cpu = b.on_cpu? ? b : b.to_cpu

        result = Tensor.new(a.shape, a.dtype, Tensor::Device::CPU)
        a_data = a_cpu.cpu_data.not_nil!
        b_data = b_cpu.cpu_data.not_nil!
        result_data = result.cpu_data.not_nil!

        a.numel.times { |i| result_data[i] = a_data[i] - b_data[i] }

        a.on_gpu? ? result.to_gpu : result
      end

      # Multiplication
      def *(other : Variable) : Variable
        result_data = mul_tensors(@data, other.data)
        result = Variable.new(result_data, @requires_grad || other.requires_grad?)

        if result.requires_grad?
          result.is_leaf = false
          grad_fn = MulBackward.new(@data.clone, other.data.clone)
          grad_fn.inputs = [self, other]
          result.grad_fn = grad_fn
        end

        result
      end

      def *(scalar : Float32) : Variable
        result_data = @data.clone
        result_data.to_cpu!
        result_data.cpu_data.not_nil!.map! { |x| x * scalar }
        result_data.to_gpu! if @data.on_gpu?

        result = Variable.new(result_data, @requires_grad)

        if result.requires_grad?
          result.is_leaf = false
          # Gradient is just scalar * upstream
          grad_fn = CustomBackward.new("MulScalarBackward", ->(g : Tensor) {
            g_cpu = g.on_cpu? ? g : g.to_cpu
            r = Tensor.new(g.shape, g.dtype, Tensor::Device::CPU)
            g_data = g_cpu.cpu_data.not_nil!
            r_data = r.cpu_data.not_nil!
            g.numel.times { |i| r_data[i] = g_data[i] * scalar }
            [g.on_gpu? ? r.to_gpu : r] of Tensor?
          })
          grad_fn.inputs = [self]
          result.grad_fn = grad_fn
        end

        result
      end

      private def mul_tensors(a : Tensor, b : Tensor) : Tensor
        a_cpu = a.on_cpu? ? a : a.to_cpu
        b_cpu = b.on_cpu? ? b : b.to_cpu

        result = Tensor.new(a.shape, a.dtype, Tensor::Device::CPU)
        a_data = a_cpu.cpu_data.not_nil!
        b_data = b_cpu.cpu_data.not_nil!
        result_data = result.cpu_data.not_nil!

        a.numel.times { |i| result_data[i] = a_data[i] * b_data[i] }

        a.on_gpu? ? result.to_gpu : result
      end

      # Division
      def /(other : Variable) : Variable
        result_data = div_tensors(@data, other.data)
        result = Variable.new(result_data, @requires_grad || other.requires_grad?)

        if result.requires_grad?
          result.is_leaf = false
          grad_fn = DivBackward.new(@data.clone, other.data.clone)
          grad_fn.inputs = [self, other]
          result.grad_fn = grad_fn
        end

        result
      end

      private def div_tensors(a : Tensor, b : Tensor) : Tensor
        a_cpu = a.on_cpu? ? a : a.to_cpu
        b_cpu = b.on_cpu? ? b : b.to_cpu

        result = Tensor.new(a.shape, a.dtype, Tensor::Device::CPU)
        a_data = a_cpu.cpu_data.not_nil!
        b_data = b_cpu.cpu_data.not_nil!
        result_data = result.cpu_data.not_nil!

        a.numel.times { |i| result_data[i] = a_data[i] / b_data[i] }

        a.on_gpu? ? result.to_gpu : result
      end

      # Matrix multiplication
      def matmul(other : Variable) : Variable
        result_data = matmul_tensors(@data, other.data)
        result = Variable.new(result_data, @requires_grad || other.requires_grad?)

        if result.requires_grad?
          result.is_leaf = false
          grad_fn = MatmulBackward.new(@data.clone, other.data.clone)
          grad_fn.inputs = [self, other]
          result.grad_fn = grad_fn
        end

        result
      end

      private def matmul_tensors(a : Tensor, b : Tensor) : Tensor
        m = a.shape[0]
        k = a.shape[1]
        n = b.shape[1]

        a_cpu = a.on_cpu? ? a : a.to_cpu
        b_cpu = b.on_cpu? ? b : b.to_cpu

        result = Tensor.new(Shape.new(m, n), a.dtype, Tensor::Device::CPU)
        a_data = a_cpu.cpu_data.not_nil!
        b_data = b_cpu.cpu_data.not_nil!
        c_data = result.cpu_data.not_nil!

        m.times do |i|
          n.times do |j|
            sum = 0.0_f32
            k.times do |kk|
              sum += a_data[i * k + kk] * b_data[kk * n + j]
            end
            c_data[i * n + j] = sum
          end
        end

        a.on_gpu? ? result.to_gpu : result
      end

      # Sum reduction
      def sum : Variable
        total = 0.0_f32
        data_cpu = @data.on_cpu? ? @data : @data.to_cpu
        data_cpu.cpu_data.not_nil!.each { |x| total += x }

        result_data = Tensor.new(1, device: @data.device)
        result_data.to_cpu!
        result_data.cpu_data.not_nil![0] = total
        result_data.to_gpu! if @data.on_gpu?

        result = Variable.new(result_data, @requires_grad)

        if result.requires_grad?
          result.is_leaf = false
          grad_fn = SumBackward.new(@data.shape)
          grad_fn.inputs = [self]
          result.grad_fn = grad_fn
        end

        result
      end

      # Mean reduction
      def mean : Variable
        total = 0.0_f32
        data_cpu = @data.on_cpu? ? @data : @data.to_cpu
        data_cpu.cpu_data.not_nil!.each { |x| total += x }
        total /= @data.numel

        result_data = Tensor.new(1, device: @data.device)
        result_data.to_cpu!
        result_data.cpu_data.not_nil![0] = total
        result_data.to_gpu! if @data.on_gpu?

        result = Variable.new(result_data, @requires_grad)

        if result.requires_grad?
          result.is_leaf = false
          grad_fn = MeanBackward.new(@data.shape)
          grad_fn.inputs = [self]
          result.grad_fn = grad_fn
        end

        result
      end

      # ReLU activation
      def relu : Variable
        data_cpu = @data.on_cpu? ? @data : @data.to_cpu

        result_data = Tensor.new(@data.shape, @data.dtype, Tensor::Device::CPU)
        in_data = data_cpu.cpu_data.not_nil!
        out_data = result_data.cpu_data.not_nil!

        @data.numel.times { |i| out_data[i] = Math.max(0.0_f32, in_data[i]) }

        result_data = result_data.to_gpu if @data.on_gpu?
        result = Variable.new(result_data, @requires_grad)

        if result.requires_grad?
          result.is_leaf = false
          grad_fn = ReluBackward.new(@data.clone)
          grad_fn.inputs = [self]
          result.grad_fn = grad_fn
        end

        result
      end

      # Sigmoid activation
      def sigmoid : Variable
        data_cpu = @data.on_cpu? ? @data : @data.to_cpu

        result_data = Tensor.new(@data.shape, @data.dtype, Tensor::Device::CPU)
        in_data = data_cpu.cpu_data.not_nil!
        out_data = result_data.cpu_data.not_nil!

        @data.numel.times { |i| out_data[i] = 1.0_f32 / (1.0_f32 + Math.exp(-in_data[i])) }

        result_data = result_data.to_gpu if @data.on_gpu?
        result = Variable.new(result_data, @requires_grad)

        if result.requires_grad?
          result.is_leaf = false
          grad_fn = SigmoidBackward.new(result_data.clone)
          grad_fn.inputs = [self]
          result.grad_fn = grad_fn
        end

        result
      end

      # Transpose
      def t : Variable
        result = Variable.new(@data.t, @requires_grad)
        # Transpose backward is just transpose again
        if result.requires_grad?
          result.is_leaf = false
          grad_fn = CustomBackward.new("TransposeBackward", ->(g : Tensor) {
            [g.t] of Tensor?
          })
          grad_fn.inputs = [self]
          result.grad_fn = grad_fn
        end
        result
      end

      # Reshape
      def reshape(*dims : Int32) : Variable
        result = Variable.new(@data.reshape(*dims), @requires_grad)
        if result.requires_grad?
          result.is_leaf = false
          original_shape = @data.shape
          grad_fn = CustomBackward.new("ReshapeBackward", ->(g : Tensor) {
            [g.reshape(original_shape)] of Tensor?
          })
          grad_fn.inputs = [self]
          result.grad_fn = grad_fn
        end
        result
      end

      # String representation
      def to_s(io : IO) : Nil
        io << "Variable("
        io << @data.to_s
        io << ", requires_grad=#{@requires_grad}"
        io << ", grad_fn=#{@grad_fn.try(&.name)}" if @grad_fn
        io << ")"
      end

      def inspect(io : IO) : Nil
        to_s(io)
      end

      # Item (for scalar tensors)
      def item : Float32
        raise "item() only works for scalar tensors" unless @data.numel == 1
        @data.to_a.first
      end
    end

    # Context manager for disabling gradient computation
    class NoGrad
      @@enabled = false

      def self.enabled? : Bool
        @@enabled
      end

      def self.enable : Nil
        @@enabled = true
      end

      def self.disable : Nil
        @@enabled = false
      end

      def self.with(&block)
        prev = @@enabled
        @@enabled = true
        begin
          yield
        ensure
          @@enabled = prev
        end
      end
    end
  end
end
