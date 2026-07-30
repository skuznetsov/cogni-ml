# Layer Normalization
# Normalizes over the last dimension(s) with learnable affine transform

require "../autograd/variable"
require "../core/tensor"
require "../ops/normalization"
require "./gpu_ops"

module ML
  module NN
    # Layer Normalization: y = (x - mean) / sqrt(var + eps) * gamma + beta
    class LayerNorm
      getter normalized_shape : Array(Int32)
      getter weight : Autograd::Variable # gamma
      getter bias : Autograd::Variable   # beta
      getter eps : Float32

      def initialize(normalized_shape : Array(Int32) | Int32, @eps : Float32 = 1e-5_f32, device : Tensor::Device = Tensor.default_device)
        @normalized_shape = normalized_shape.is_a?(Int32) ? [normalized_shape] : normalized_shape
        if @normalized_shape.empty?
          raise ArgumentError.new("normalized_shape must not be empty")
        end
        unless @normalized_shape.all? { |dimension| dimension > 0 }
          raise ArgumentError.new("normalized_shape dimensions must be positive")
        end
        unless @eps.finite? && @eps > 0.0_f32
          raise ArgumentError.new("eps must be finite and positive")
        end

        # Number of elements in normalized dimensions
        num_elements = Shape.new(@normalized_shape).numel

        # Initialize gamma (weight) to ones
        weight_data = Tensor.ones(num_elements, device: device)
        @weight = Autograd::Variable.new(weight_data, requires_grad: true)

        # Initialize beta (bias) to zeros
        bias_data = Tensor.zeros(num_elements, device: device)
        @bias = Autograd::Variable.new(bias_data, requires_grad: true)
      end

      # Convenience constructor for single dimension
      def self.new(dim : Int32, eps : Float32 = 1e-5_f32, device : Tensor::Device = Tensor.default_device)
        new([dim], eps, device)
      end

      # Forward pass
      # x: [..., *normalized_shape]
      # output: same shape as x
      def forward(x : Autograd::Variable) : Autograd::Variable
        norm_size = Ops::CPU.validate_layer_norm_inputs!(
          x.data,
          @weight.data,
          @bias.data,
          @normalized_shape,
          @eps
        )

        # Number of "batches" (everything except normalized dims)
        total = x.data.numel
        batch_size = total // norm_size
        needs_grad = !Autograd::NoGrad.enabled? &&
                     (x.requires_grad? || @weight.requires_grad? || @bias.requires_grad?)

        # Try GPU path if all tensors on GPU
        if x.data.on_gpu? && @weight.data.on_gpu? && @bias.data.on_gpu? &&
           x.data.contiguous? && @weight.data.contiguous? && @bias.data.contiguous? &&
           GPUOps.available?
          result = forward_gpu(x.data, batch_size, norm_size)

          means = [] of Float32
          inv_stds = [] of Float32

          if needs_grad
            means = Array(Float32).new(batch_size, 0.0_f32)
            inv_stds = Array(Float32).new(batch_size, 0.0_f32)

            # For backward, we need mean and inv_std from CPU computation
            # TODO: Compute these in GPU kernel and read back
            x_cpu = x.data.to_cpu
            x_d = x_cpu.cpu_data.not_nil!
            batch_size.times do |b|
              offset = b * norm_size
              mean = 0.0_f32
              norm_size.times { |i| mean += x_d[offset + i] }
              mean /= norm_size
              means[b] = mean

              var = 0.0_f32
              norm_size.times do |i|
                diff = x_d[offset + i] - mean
                var += diff * diff
              end
              var /= norm_size
              inv_stds[b] = 1.0_f32 / Math.sqrt(var + @eps)
            end
          end
        else
          result, means, inv_stds = forward_cpu(x.data, batch_size, norm_size)
        end

        result_var = Autograd::Variable.new(result, needs_grad)

        if result_var.requires_grad?
          result_var.is_leaf = false

          # Store for backward
          x_clone = x.data.clone
          eps = @eps
          norm_size_cap = norm_size
          batch_size_cap = batch_size
          means_cap = means
          inv_stds_cap = inv_stds
          weight_data_cap = @weight.data.clone
          x_on_gpu = x.data.on_gpu?
          weight_on_gpu = @weight.data.on_gpu?
          bias_on_gpu = @bias.data.on_gpu?

          grad_fn = Autograd::CustomBackward.new("LayerNormBackward", ->(grad_output : Tensor) {
            g_cpu = grad_output.to_contiguous_cpu
            x_cpu = x_clone.to_contiguous_cpu
            w_cpu = weight_data_cap.to_contiguous_cpu

            g_d = g_cpu.cpu_data.not_nil!
            x_d_bw = x_cpu.cpu_data.not_nil!
            w_d_bw = w_cpu.cpu_data.not_nil!

            grad_x = Tensor.new(x_clone.shape, x_clone.dtype, Tensor::Device::CPU)
            grad_w = Tensor.zeros(norm_size_cap, device: Tensor::Device::CPU)
            grad_b = Tensor.zeros(norm_size_cap, device: Tensor::Device::CPU)

            gx_d = grad_x.cpu_data.not_nil!
            gw_d = grad_w.cpu_data.not_nil!
            gb_d = grad_b.cpu_data.not_nil!

            n = norm_size_cap.to_f32

            batch_size_cap.times do |b|
              offset = b * norm_size_cap
              mean = means_cap[b]
              inv_std = inv_stds_cap[b]

              # Compute normalized values for this batch
              x_norm = Array(Float32).new(norm_size_cap) do |i|
                (x_d_bw[offset + i] - mean) * inv_std
              end

              # Accumulate grad_weight and grad_bias
              norm_size_cap.times do |i|
                gw_d[i] += g_d[offset + i] * x_norm[i]
                gb_d[i] += g_d[offset + i]
              end

              # Compute grad_x (complex due to mean/var dependencies)
              # d_xnorm = grad_out * weight
              d_xnorm = Array(Float32).new(norm_size_cap) { |i| g_d[offset + i] * w_d_bw[i] }

              # Sum terms for variance gradient
              sum_d_xnorm = d_xnorm.sum
              sum_d_xnorm_xnorm = 0.0_f32
              norm_size_cap.times { |i| sum_d_xnorm_xnorm += d_xnorm[i] * x_norm[i] }

              # grad_x = inv_std * (d_xnorm - mean(d_xnorm) - x_norm * mean(d_xnorm * x_norm))
              norm_size_cap.times do |i|
                gx_d[offset + i] = inv_std * (d_xnorm[i] - sum_d_xnorm / n - x_norm[i] * sum_d_xnorm_xnorm / n)
              end
            end

            [
              x_on_gpu ? grad_x.to_gpu : grad_x,
              weight_on_gpu ? grad_w.to_gpu : grad_w,
              bias_on_gpu ? grad_b.to_gpu : grad_b,
            ] of Tensor?
          })

          grad_fn.inputs = [x, @weight, @bias]
          result_var.grad_fn = grad_fn
        end

        result_var
      end

      def call(x : Autograd::Variable) : Autograd::Variable
        forward(x)
      end

      # Get all trainable parameters
      def parameters : Array(Autograd::Variable)
        [@weight, @bias]
      end

      # GPU forward pass
      private def forward_gpu(x : Tensor, batch_size : Int32, norm_size : Int32) : Tensor
        # Reshape for GPU kernel: [batch, features]
        x_2d = if x.ndim == 2
                 x
               else
                 Tensor.new(batch_size, norm_size, device: Tensor::Device::GPU).tap do |t|
                   # Copy data (kernel expects contiguous [batch, features])
                   t.buffer.not_nil!.write(x.buffer.not_nil!.read(x.numel))
                 end
               end

        result_2d = Tensor.new(batch_size, norm_size, device: Tensor::Device::GPU)
        GPUOps.layernorm_forward(x_2d, @weight.data, @bias.data, result_2d, @eps)

        # Reshape back if needed
        if x.ndim == 2
          result_2d
        else
          result = Tensor.new(x.shape, x.dtype, Tensor::Device::GPU)
          result.buffer.not_nil!.write(result_2d.buffer.not_nil!.read(result_2d.numel))
          result
        end
      end

      # CPU forward pass
      private def forward_cpu(x : Tensor, batch_size : Int32, norm_size : Int32) : {Tensor, Array(Float32), Array(Float32)}
        evaluation = Ops::CPU.layer_norm_with_stats(
          x,
          @weight.data,
          @bias.data,
          @normalized_shape,
          @eps
        )
        result = x.on_gpu? ? evaluation.output.to_gpu : evaluation.output
        {result, evaluation.means, evaluation.inv_stds}
      end
    end

    # RMSNorm - simplified LayerNorm without mean centering
    # Used in some modern architectures (LLaMA, etc.)
    class RMSNorm
      getter dim : Int32
      getter weight : Autograd::Variable
      getter eps : Float32

      def initialize(@dim : Int32, @eps : Float32 = 1e-5_f32, device : Tensor::Device = Tensor.default_device)
        if @dim <= 0
          raise ArgumentError.new("dim must be positive")
        end
        unless @eps.finite? && @eps > 0.0_f32
          raise ArgumentError.new("eps must be finite and positive")
        end

        weight_data = Tensor.ones(@dim, device: device)
        @weight = Autograd::Variable.new(weight_data, requires_grad: true)
      end

      def forward(x : Autograd::Variable) : Autograd::Variable
        Ops::CPU.validate_rms_norm_inputs!(x.data, @weight.data, @dim, @eps)

        # Calculate RMS over last dimension
        total = x.data.numel
        batch_size = total // @dim

        needs_grad = !Autograd::NoGrad.enabled? &&
                     (x.requires_grad? || @weight.requires_grad?)

        # Try GPU path
        if x.data.on_gpu? && @weight.data.on_gpu? &&
           x.data.contiguous? && @weight.data.contiguous? && GPUOps.available?
          result = forward_gpu(x.data, batch_size)
        else
          result = forward_cpu(x.data, batch_size)
        end

        result_var = Autograd::Variable.new(result, needs_grad)

        if needs_grad
          result_var.is_leaf = false
          x_clone = x.data.clone
          w_clone = @weight.data.clone
          dim_cap = @dim
          eps_cap = @eps
          x_on_gpu = x.data.on_gpu?
          w_on_gpu = @weight.data.on_gpu?

          grad_fn = Autograd::CustomBackward.new("RMSNormBackward", ->(g : Tensor) {
            g_cpu = g.to_contiguous_cpu
            x_cpu = x_clone.to_contiguous_cpu
            w_cpu = w_clone.to_contiguous_cpu

            g_d = g_cpu.cpu_data.not_nil!
            x_d = x_cpu.cpu_data.not_nil!
            w_d = w_cpu.cpu_data.not_nil!

            grad_x = Tensor.new(x_clone.shape, x_clone.dtype, Tensor::Device::CPU)
            grad_w = Tensor.zeros(dim_cap, device: Tensor::Device::CPU)
            gx_d = grad_x.cpu_data.not_nil!
            gw_d = grad_w.cpu_data.not_nil!

            batch = x_clone.numel // dim_cap
            dim_f = dim_cap.to_f32

            batch.times do |b|
              offset = b * dim_cap
              sum_sq = 0.0_f32
              dim_cap.times { |i| sum_sq += x_d[offset + i] * x_d[offset + i] }
              inv_r = 1.0_f32 / Math.sqrt(sum_sq / dim_f + eps_cap)
              inv_r3 = inv_r * inv_r * inv_r

              dot = 0.0_f32
              dim_cap.times do |i|
                dot += g_d[offset + i] * w_d[i] * x_d[offset + i]
              end

              scale = inv_r3 / dim_f

              dim_cap.times do |i|
                gx_d[offset + i] = inv_r * w_d[i] * g_d[offset + i] - x_d[offset + i] * dot * scale
                gw_d[i] += g_d[offset + i] * x_d[offset + i] * inv_r
              end
            end

            [x_on_gpu ? grad_x.to_gpu : grad_x, w_on_gpu ? grad_w.to_gpu : grad_w] of Tensor?
          })

          grad_fn.inputs = [x, @weight]
          result_var.grad_fn = grad_fn
        end

        result_var
      end

      private def forward_gpu(x : Tensor, batch_size : Int32) : Tensor
        # Reshape for GPU kernel: [batch, features]
        x_2d = if x.ndim == 2
                 x
               else
                 Tensor.new(batch_size, @dim, device: Tensor::Device::GPU).tap do |t|
                   t.buffer.not_nil!.write(x.buffer.not_nil!.read(x.numel))
                 end
               end

        result_2d = Tensor.new(batch_size, @dim, device: Tensor::Device::GPU)
        GPUOps.rmsnorm_forward(x_2d, @weight.data, result_2d, @eps)

        if x.ndim == 2
          result_2d
        else
          result = Tensor.new(x.shape, x.dtype, Tensor::Device::GPU)
          result.buffer.not_nil!.write(result_2d.buffer.not_nil!.read(result_2d.numel))
          result
        end
      end

      private def forward_cpu(x : Tensor, batch_size : Int32) : Tensor
        result = Ops::CPU.rms_norm(x, @weight.data, @dim, @eps)
        x.on_gpu? ? result.to_gpu : result
      end

      def call(x : Autograd::Variable) : Autograd::Variable
        forward(x)
      end

      def parameters : Array(Autograd::Variable)
        [@weight]
      end
    end
  end
end
