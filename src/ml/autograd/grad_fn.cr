# Gradient function base class and registry
# Each operation registers its backward function

require "../core/tensor"

module ML
  module Autograd
    # Forward declaration
    class Variable
    end

    # Base class for gradient functions
    # Each operation creates a GradFn that knows how to compute gradients
    abstract class GradFn
      # Inputs that contributed to this operation
      # Stored as weak references to avoid cycles
      property inputs : Array(Variable)

      # For debugging
      getter name : String

      def initialize(@name : String)
        @inputs = Array(Variable).new
      end

      # Compute gradients w.r.t. inputs given gradient of output
      # Returns array of gradients, one per input (or nil if input doesn't require grad)
      abstract def backward(grad_output : Tensor) : Array(Tensor?)

      # Number of inputs this operation takes
      def num_inputs : Int32
        @inputs.size
      end
    end

    # Gradient function for addition: c = a + b
    # dc/da = 1, dc/db = 1
    class AddBackward < GradFn
      def initialize
        super("AddBackward")
      end

      def backward(grad_output : Tensor) : Array(Tensor?)
        # Both gradients are just the upstream gradient
        [grad_output.clone, grad_output.clone] of Tensor?
      end
    end

    # Gradient function for subtraction: c = a - b
    # dc/da = 1, dc/db = -1
    class SubBackward < GradFn
      def initialize
        super("SubBackward")
      end

      def backward(grad_output : Tensor) : Array(Tensor?)
        # grad_a = grad_output, grad_b = -grad_output
        grad_b = grad_output.clone
        # Negate grad_b (TODO: use kernel)
        grad_b.to_cpu!
        grad_b.cpu_data.not_nil!.map! { |x| -x }
        grad_b.to_gpu! if grad_output.on_gpu?
        [grad_output.clone, grad_b] of Tensor?
      end
    end

    # Gradient function for multiplication: c = a * b
    # dc/da = b, dc/db = a
    class MulBackward < GradFn
      @a_data : Tensor
      @b_data : Tensor

      def initialize(@a_data : Tensor, @b_data : Tensor)
        super("MulBackward")
      end

      def backward(grad_output : Tensor) : Array(Tensor?)
        # grad_a = grad_output * b
        # grad_b = grad_output * a
        # TODO: use GPU kernel
        grad_a = elementwise_mul(grad_output, @b_data)
        grad_b = elementwise_mul(grad_output, @a_data)
        [grad_a, grad_b] of Tensor?
      end

      private def elementwise_mul(a : Tensor, b : Tensor) : Tensor
        # CPU fallback for now
        a_cpu = a.on_cpu? ? a : a.to_cpu
        b_cpu = b.on_cpu? ? b : b.to_cpu

        result = Tensor.new(a.shape, a.dtype, Tensor::Device::CPU)
        a_data = a_cpu.cpu_data.not_nil!
        b_data = b_cpu.cpu_data.not_nil!
        result_data = result.cpu_data.not_nil!

        a.shape.numel.times do |i|
          result_data[i] = a_data[i] * b_data[i]
        end

        a.on_gpu? ? result.to_gpu : result
      end
    end

    # Gradient function for division: c = a / b
    # dc/da = 1/b, dc/db = -a/b^2
    class DivBackward < GradFn
      @a_data : Tensor
      @b_data : Tensor

      def initialize(@a_data : Tensor, @b_data : Tensor)
        super("DivBackward")
      end

      def backward(grad_output : Tensor) : Array(Tensor?)
        # grad_a = grad_output / b
        # grad_b = -grad_output * a / b^2
        # TODO: use GPU kernel
        a_cpu = @a_data.on_cpu? ? @a_data : @a_data.to_cpu
        b_cpu = @b_data.on_cpu? ? @b_data : @b_data.to_cpu
        grad_cpu = grad_output.on_cpu? ? grad_output : grad_output.to_cpu

        grad_a = Tensor.new(grad_output.shape, grad_output.dtype, Tensor::Device::CPU)
        grad_b = Tensor.new(grad_output.shape, grad_output.dtype, Tensor::Device::CPU)

        a_d = a_cpu.cpu_data.not_nil!
        b_d = b_cpu.cpu_data.not_nil!
        g_d = grad_cpu.cpu_data.not_nil!
        ga_d = grad_a.cpu_data.not_nil!
        gb_d = grad_b.cpu_data.not_nil!

        grad_output.shape.numel.times do |i|
          ga_d[i] = g_d[i] / b_d[i]
          gb_d[i] = -g_d[i] * a_d[i] / (b_d[i] * b_d[i])
        end

        if grad_output.on_gpu?
          [grad_a.to_gpu, grad_b.to_gpu] of Tensor?
        else
          [grad_a, grad_b] of Tensor?
        end
      end
    end

    # Gradient function for matrix multiplication: C = A @ B
    # dL/dA = dL/dC @ B^T
    # dL/dB = A^T @ dL/dC
    class MatmulBackward < GradFn
      @a_data : Tensor
      @b_data : Tensor

      def initialize(@a_data : Tensor, @b_data : Tensor)
        super("MatmulBackward")
      end

      def backward(grad_output : Tensor) : Array(Tensor?)
        # TODO: use GPU matmul kernel
        # For now, CPU implementation
        grad_a = matmul_cpu(grad_output, @b_data.t)
        grad_b = matmul_cpu(@a_data.t, grad_output)
        [grad_a, grad_b] of Tensor?
      end

      private def matmul_cpu(a : Tensor, b : Tensor) : Tensor
        a_cpu = a.on_cpu? ? a : a.to_cpu
        b_cpu = b.on_cpu? ? b : b.to_cpu

        m = a.shape[0]
        k = a.shape[1]
        n = b.shape[1]

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

        result
      end
    end

    # Gradient for ReLU: out = max(0, x)
    # d(relu)/dx = 1 if x > 0 else 0
    class ReluBackward < GradFn
      @input_data : Tensor

      def initialize(@input_data : Tensor)
        super("ReluBackward")
      end

      def backward(grad_output : Tensor) : Array(Tensor?)
        # grad_input = grad_output * (input > 0)
        in_cpu = @input_data.on_cpu? ? @input_data : @input_data.to_cpu
        grad_cpu = grad_output.on_cpu? ? grad_output : grad_output.to_cpu

        result = Tensor.new(grad_output.shape, grad_output.dtype, Tensor::Device::CPU)
        in_data = in_cpu.cpu_data.not_nil!
        grad_data = grad_cpu.cpu_data.not_nil!
        result_data = result.cpu_data.not_nil!

        grad_output.shape.numel.times do |i|
          result_data[i] = in_data[i] > 0 ? grad_data[i] : 0.0_f32
        end

        [grad_output.on_gpu? ? result.to_gpu : result] of Tensor?
      end
    end

    # Gradient for Sigmoid: out = 1 / (1 + exp(-x))
    # d(sigmoid)/dx = out * (1 - out)
    class SigmoidBackward < GradFn
      @output_data : Tensor  # Store sigmoid output, not input

      def initialize(@output_data : Tensor)
        super("SigmoidBackward")
      end

      def backward(grad_output : Tensor) : Array(Tensor?)
        out_cpu = @output_data.on_cpu? ? @output_data : @output_data.to_cpu
        grad_cpu = grad_output.on_cpu? ? grad_output : grad_output.to_cpu

        result = Tensor.new(grad_output.shape, grad_output.dtype, Tensor::Device::CPU)
        out_data = out_cpu.cpu_data.not_nil!
        grad_data = grad_cpu.cpu_data.not_nil!
        result_data = result.cpu_data.not_nil!

        grad_output.shape.numel.times do |i|
          s = out_data[i]
          result_data[i] = grad_data[i] * s * (1.0_f32 - s)
        end

        [grad_output.on_gpu? ? result.to_gpu : result] of Tensor?
      end
    end

    # Gradient for sum reduction
    # d(sum)/dx = 1 (broadcast)
    class SumBackward < GradFn
      @input_shape : Shape

      def initialize(@input_shape : Shape)
        super("SumBackward")
      end

      def backward(grad_output : Tensor) : Array(Tensor?)
        # grad_output is scalar, broadcast to input shape
        grad_val = grad_output.to_a.first

        result = Tensor.new(@input_shape, grad_output.dtype, Tensor::Device::CPU)
        result.fill!(grad_val)

        [grad_output.on_gpu? ? result.to_gpu : result] of Tensor?
      end
    end

    # Gradient for mean reduction
    # d(mean)/dx = 1/n
    class MeanBackward < GradFn
      @input_shape : Shape

      def initialize(@input_shape : Shape)
        super("MeanBackward")
      end

      def backward(grad_output : Tensor) : Array(Tensor?)
        grad_val = grad_output.to_a.first
        n = @input_shape.numel.to_f32

        result = Tensor.new(@input_shape, grad_output.dtype, Tensor::Device::CPU)
        result.fill!(grad_val / n)

        [grad_output.on_gpu? ? result.to_gpu : result] of Tensor?
      end
    end

    # Accumulate gradient (no-op backward, just stores gradient)
    class AccumulateGrad < GradFn
      def initialize
        super("AccumulateGrad")
      end

      def backward(grad_output : Tensor) : Array(Tensor?)
        # Leaf variable - accumulate gradient, don't propagate
        Array(Tensor?).new
      end
    end

    # Placeholder for custom backward functions (e.g., rasterizer)
    class CustomBackward < GradFn
      @backward_fn : Proc(Tensor, Array(Tensor?))

      def initialize(name : String, @backward_fn : Proc(Tensor, Array(Tensor?)))
        super(name)
      end

      def backward(grad_output : Tensor) : Array(Tensor?)
        @backward_fn.call(grad_output)
      end
    end
  end
end
