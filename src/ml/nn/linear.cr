# Linear (fully-connected) layer
# y = x @ W^T + b

require "../autograd/variable"
require "../core/tensor"
require "./gpu_ops"

module ML
  module NN
    # Linear layer: out_features = in_features @ weight^T + bias
    class Linear
      getter weight : Autograd::Variable
      getter bias : Autograd::Variable?
      getter in_features : Int32
      getter out_features : Int32

      def initialize(@in_features : Int32, @out_features : Int32, bias : Bool = true, device : Tensor::Device = Tensor::Device::GPU)
        # Initialize weight with Kaiming/He initialization
        # std = sqrt(2 / in_features) for ReLU
        std = Math.sqrt(2.0 / @in_features).to_f32

        weight_data = Tensor.randn(@out_features, @in_features, device: device)
        weight_data.to_cpu!
        weight_data.cpu_data.not_nil!.map! { |x| x * std }
        weight_data.to_gpu! if device.gpu?

        @weight = Autograd::Variable.new(weight_data, requires_grad: true)

        if bias
          # Initialize bias to zero
          bias_data = Tensor.zeros(@out_features, device: device)
          @bias = Autograd::Variable.new(bias_data, requires_grad: true)
        else
          @bias = nil
        end
      end

      # Forward pass: y = x @ W^T + b
      # x: [..., in_features] - arbitrary batch dimensions
      # output: [..., out_features]
      def forward(x : Autograd::Variable) : Autograd::Variable
        input_shape = x.data.shape.to_a
        in_features = input_shape[input_shape.size - 1]
        needs_grad = x.requires_grad? || @weight.requires_grad? || (@bias.try(&.requires_grad?) || false)

        if x.data.on_gpu? && @weight.data.on_gpu? && GPUOps.available? && !needs_grad
          batch_product = 1
          (input_shape.size - 1).times { |i| batch_product *= input_shape[i] }

          x_2d = if x.data.ndim == 2
                   x.data
                 else
                   x.data.reshape(batch_product, in_features)
                 end

          output_2d = Tensor.new(batch_product, @out_features, device: Tensor::Device::GPU)
          GPUOps.linear_forward(x_2d, @weight.data, @bias.try(&.data), output_2d)

          output_shape = input_shape[0...-1] + [@out_features]
          output = output_shape.size == 2 ? output_2d : output_2d.reshape(Shape.new(output_shape))

          return Autograd::Variable.new(output, false)
        end

        # Flatten all batch dims: [..., in_features] -> [batch_product, in_features]
        batch_product = 1
        (input_shape.size - 1).times { |i| batch_product *= input_shape[i] }

        # Reshape to 2D for matmul
        x_2d = reshape_for_linear(x, batch_product, in_features)

        # x @ W^T
        output = matmul(x_2d, @weight.transpose)

        # Add bias if present
        if b = @bias
          output = add_bias(output, b)
        end

        # Reshape back to original batch dims: [batch_product, out_features] -> [..., out_features]
        output_shape = input_shape[0...-1] + [@out_features]
        reshape_from_linear(output, output_shape)
      end

      private def reshape_for_linear(x : Autograd::Variable, batch : Int32, features : Int32) : Autograd::Variable
        return x if x.data.ndim == 2  # Already 2D

        if x.data.on_gpu?
          reshaped = x.data.reshape(batch, features)
          return Autograd::Variable.new(reshaped, x.requires_grad?)
        end

        x_data = x.data.on_cpu? ? x.data : x.data.to_cpu
        result = Tensor.new(batch, features, device: Tensor::Device::CPU)

        x_d = x_data.cpu_data.not_nil!
        r_d = result.cpu_data.not_nil!
        (batch * features).times { |i| r_d[i] = x_d[i] }

        result = result.to_gpu if x.data.on_gpu?
        Autograd::Variable.new(result, x.requires_grad?)
      end

      private def reshape_from_linear(x : Autograd::Variable, shape : Array(Int32)) : Autograd::Variable
        return x if shape.size == 2  # Already correct shape

        if x.data.on_gpu?
          reshaped = x.data.reshape(Shape.new(shape))
          return Autograd::Variable.new(reshaped, x.requires_grad?)
        end

        x_data = x.data.on_cpu? ? x.data : x.data.to_cpu
        result = Tensor.new(Shape.new(shape), x_data.dtype, Tensor::Device::CPU)

        x_d = x_data.cpu_data.not_nil!
        r_d = result.cpu_data.not_nil!
        x_data.numel.times { |i| r_d[i] = x_d[i] }

        result = result.to_gpu if x.data.on_gpu?
        Autograd::Variable.new(result, x.requires_grad?)
      end

      def call(x : Autograd::Variable) : Autograd::Variable
        forward(x)
      end

      # Get all trainable parameters
      def parameters : Array(Autograd::Variable)
        if b = @bias
          [@weight, b]
        else
          [@weight]
        end
      end

      # Matrix multiplication with autograd
      # a @ b where:
      #   a: [batch, m, k] or [m, k] or [k]
      #   b: [k, n] (already transposed weight)
      # Result: [batch, m, n] or [m, n] or [n]
      private def matmul(a : Autograd::Variable, b : Autograd::Variable) : Autograd::Variable
        # b is already transposed: [in_features, out_features]
        k = b.data.shape[0]  # in_features
        n = b.data.shape[1]  # out_features

        # Handle batched vs unbatched
        if a.data.ndim == 1
          # [in] @ [in, out] = [out]
          m = 1
          result_shape = Shape.new(n)
        else
          # [batch, in] @ [in, out] = [batch, out]
          m = a.data.shape[0]  # batch size
          result_shape = Shape.new(m, n)
        end

        # Try GPU path if both tensors on GPU
        if a.data.on_gpu? && b.data.on_gpu? && GPUOps.available?
          result = matmul_gpu(a.data, b.data, m, k, n)
        else
          result = matmul_cpu(a.data, b.data, m, k, n)
        end

        result_var = Autograd::Variable.new(result, a.requires_grad? || b.requires_grad?)

        if result_var.requires_grad?
          result_var.is_leaf = false
          grad_fn = Autograd::MatmulBackward.new(a.data.clone, b.data.clone)
          grad_fn.inputs = [a, b]
          result_var.grad_fn = grad_fn
        end

        result_var
      end

      # GPU matmul using Metal kernel
      # Note: For Linear layer, we compute input @ weight^T
      # input: [batch, in_features], weight_t: [in_features, out_features]
      private def matmul_gpu(a : Tensor, b : Tensor, m : Int32, k : Int32, n : Int32) : Tensor
        result = Tensor.new(m, n, device: Tensor::Device::GPU)

        # We need to transpose b back and call linear_forward which does input @ weight^T internally
        # Actually, b is weight^T = [in_features, out_features]
        # We need: a @ b = a @ weight^T
        # Our kernel expects: input @ weight^T where weight is [out, in]
        # So we need: input[batch, in] @ weight[out, in]^T = input[batch, in] @ [in, out]
        # b is already [in, out], so we need to transpose it to get weight[out, in]

        # Create weight in [out_features, in_features] format
        weight = b.t  # [out_features, in_features]

        GPUOps.linear_forward(a, weight, nil, result)
        result
      end

      # CPU matmul fallback
      private def matmul_cpu(a : Tensor, b : Tensor, m : Int32, k : Int32, n : Int32) : Tensor
        a_data = a.on_cpu? ? a : a.to_cpu
        b_data = b.on_cpu? ? b : b.to_cpu

        result_shape = m == 1 ? Shape.new(n) : Shape.new(m, n)
        result = Tensor.new(result_shape, a_data.dtype, Tensor::Device::CPU)

        a_d = a_data.cpu_data.not_nil!
        b_d = b_data.cpu_data.not_nil!
        r_d = result.cpu_data.not_nil!

        # Standard matmul: result[i,j] = sum_k a[i,k] * b[k,j]
        if m == 1
          # Unbatched: [in] @ [in, out] = [out]
          n.times do |j|
            sum = 0.0_f32
            k.times do |kk|
              sum += a_d[kk] * b_d[kk * n + j]
            end
            r_d[j] = sum
          end
        else
          # Batched: [batch, in] @ [in, out] = [batch, out]
          m.times do |i|
            n.times do |j|
              sum = 0.0_f32
              k.times do |kk|
                sum += a_d[i * k + kk] * b_d[kk * n + j]
              end
              r_d[i * n + j] = sum
            end
          end
        end

        result = result.to_gpu if a.on_gpu?
        result
      end

      # Add bias with broadcasting
      private def add_bias(x : Autograd::Variable, bias : Autograd::Variable) : Autograd::Variable
        x_data = x.data.on_cpu? ? x.data : x.data.to_cpu
        b_data = bias.data.on_cpu? ? bias.data : bias.data.to_cpu

        result = Tensor.new(x.data.shape, x.data.dtype, Tensor::Device::CPU)

        x_d = x_data.cpu_data.not_nil!
        b_d = b_data.cpu_data.not_nil!
        r_d = result.cpu_data.not_nil!

        if x_data.ndim == 1
          # [out] + [out]
          x_data.numel.times { |i| r_d[i] = x_d[i] + b_d[i] }
        else
          # [batch, out] + [out] (broadcast)
          batch = x_data.shape[0]
          out_dim = x_data.shape[1]
          batch.times do |i|
            out_dim.times do |j|
              r_d[i * out_dim + j] = x_d[i * out_dim + j] + b_d[j]
            end
          end
        end

        result = result.to_gpu if x.data.on_gpu?

        result_var = Autograd::Variable.new(result, x.requires_grad? || bias.requires_grad?)

        if result_var.requires_grad?
          result_var.is_leaf = false
          # Bias gradient needs sum over batch dimension
          grad_fn = Autograd::CustomBackward.new("LinearBiasBackward", ->(g : Tensor) {
            # grad_x = g (same shape)
            # grad_bias = sum(g, dim=0) if batched, else g
            grad_x = g.clone

            g_cpu = g.on_cpu? ? g : g.to_cpu
            g_data = g_cpu.cpu_data.not_nil!

            if g.ndim == 1
              grad_b = g.clone
            else
              batch = g.shape[0]
              out_dim = g.shape[1]
              grad_b = Tensor.zeros(out_dim, device: Tensor::Device::CPU)
              gb_data = grad_b.cpu_data.not_nil!

              batch.times do |i|
                out_dim.times do |j|
                  gb_data[j] += g_data[i * out_dim + j]
                end
              end

              grad_b = grad_b.to_gpu if g.on_gpu?
            end

            [grad_x, grad_b] of Tensor?
          })
          grad_fn.inputs = [x, bias]
          result_var.grad_fn = grad_fn
        end

        result_var
      end

      # Transpose weight for forward
      private def transpose_weight : Tensor
        @weight.data.t
      end
    end

    # Alias for Linear
    alias Dense = Linear
  end
end
