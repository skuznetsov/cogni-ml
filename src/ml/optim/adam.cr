# Adam and AdamW optimizers
# Adaptive learning rate with momentum and RMSprop-like scaling

require "../autograd/variable"
require "../core/tensor"

module ML
  module Optim
    # Parameter group for different learning rates
    struct ParamGroup
      property params : Array(Autograd::Variable)
      property lr : Float32
      property weight_decay : Float32
      property name : String

      def initialize(@params : Array(Autograd::Variable), @lr : Float32 = 0.001_f32, @weight_decay : Float32 = 0.0_f32, @name : String = "default")
      end
    end

    # Adam optimizer state for a single parameter
    class AdamState
      property m : Tensor  # First moment (mean of gradients)
      property v : Tensor  # Second moment (mean of squared gradients)
      property step : Int32

      def initialize(shape : Shape, device : Tensor::Device)
        @m = Tensor.new(shape, DType::F32, device)
        @m.fill!(0.0_f32)
        @v = Tensor.new(shape, DType::F32, device)
        @v.fill!(0.0_f32)
        @step = 0
      end
    end

    # Adam optimizer with optional weight decay (AdamW)
    class Adam
      property lr : Float32          # Learning rate
      property beta1 : Float32       # Exponential decay for first moment
      property beta2 : Float32       # Exponential decay for second moment
      property eps : Float32         # Small constant for numerical stability
      property weight_decay : Float32 # L2 regularization (0 for standard Adam)
      property amsgrad : Bool        # Use AMSGrad variant

      @param_groups : Array(ParamGroup)
      @state : Hash(UInt64, AdamState)

      def initialize(
        params : Array(Autograd::Variable),
        @lr : Float32 = 0.001_f32,
        @beta1 : Float32 = 0.9_f32,
        @beta2 : Float32 = 0.999_f32,
        @eps : Float32 = 1e-8_f32,
        @weight_decay : Float32 = 0.0_f32,
        @amsgrad : Bool = false
      )
        @param_groups = [ParamGroup.new(params, @lr, @weight_decay)]
        @state = Hash(UInt64, AdamState).new
      end

      # Create with parameter groups
      def self.new(param_groups : Array(ParamGroup), **kwargs) : Adam
        opt = Adam.allocate
        opt.initialize_with_groups(param_groups, **kwargs)
        opt
      end

      protected def initialize_with_groups(
        param_groups : Array(ParamGroup),
        @lr : Float32 = 0.001_f32,
        @beta1 : Float32 = 0.9_f32,
        @beta2 : Float32 = 0.999_f32,
        @eps : Float32 = 1e-8_f32,
        @weight_decay : Float32 = 0.0_f32,
        @amsgrad : Bool = false
      )
        @param_groups = param_groups
        @state = Hash(UInt64, AdamState).new
      end

      # Add parameter group
      def add_param_group(group : ParamGroup) : Nil
        @param_groups << group
      end

      # Single optimization step
      def step : Nil
        @param_groups.each do |group|
          group.params.each do |param|
            next unless param.requires_grad?
            next unless param.grad

            grad = param.grad.not_nil!
            param_data = param.data

            # Get or create state
            state = @state[param.object_id] ||= AdamState.new(param_data.shape, param_data.device)
            state.step += 1

            # Apply weight decay (AdamW style - decoupled)
            effective_wd = group.weight_decay > 0 ? group.weight_decay : @weight_decay
            if effective_wd > 0
              # param = param - lr * wd * param
              apply_weight_decay!(param_data, group.lr, effective_wd)
            end

            # Update moments
            update_adam!(param_data, grad, state, group.lr)
          end
        end
      end

      private def apply_weight_decay!(param : Tensor, lr : Float32, wd : Float32) : Nil
        # param = param * (1 - lr * wd)
        decay_factor = 1.0_f32 - lr * wd

        if param.on_cpu?
          param.cpu_data.not_nil!.map! { |x| x * decay_factor }
        else
          # Transfer to CPU, apply, transfer back
          # TODO: GPU kernel for this
          param.to_cpu!
          param.cpu_data.not_nil!.map! { |x| x * decay_factor }
          param.to_gpu!
        end
      end

      private def update_adam!(param : Tensor, grad : Tensor, state : AdamState, lr : Float32) : Nil
        # Bias correction
        bias_correction1 = 1.0_f32 - (@beta1 ** state.step)
        bias_correction2 = 1.0_f32 - (@beta2 ** state.step)

        # Update m and v, then update param
        # m = beta1 * m + (1 - beta1) * grad
        # v = beta2 * v + (1 - beta2) * grad^2
        # m_hat = m / (1 - beta1^t)
        # v_hat = v / (1 - beta2^t)
        # param = param - lr * m_hat / (sqrt(v_hat) + eps)

        # CPU implementation
        grad_cpu = grad.on_cpu? ? grad : grad.to_cpu
        param_was_gpu = param.on_gpu?

        if param.on_gpu?
          param.to_cpu!
        end
        if state.m.on_gpu?
          state.m.to_cpu!
          state.v.to_cpu!
        end

        param_data = param.cpu_data.not_nil!
        grad_data = grad_cpu.cpu_data.not_nil!
        m_data = state.m.cpu_data.not_nil!
        v_data = state.v.cpu_data.not_nil!

        param.numel.times do |i|
          g = grad_data[i]

          # Update moments
          m_data[i] = @beta1 * m_data[i] + (1.0_f32 - @beta1) * g
          v_data[i] = @beta2 * v_data[i] + (1.0_f32 - @beta2) * g * g

          # Bias correction
          m_hat = m_data[i] / bias_correction1
          v_hat = v_data[i] / bias_correction2

          # Update parameter
          param_data[i] -= lr * m_hat / (Math.sqrt(v_hat) + @eps)
        end

        if param_was_gpu
          param.to_gpu!
          state.m.to_gpu!
          state.v.to_gpu!
        end
      end

      # Zero all gradients
      def zero_grad : Nil
        @param_groups.each do |group|
          group.params.each(&.zero_grad!)
        end
      end

      # Get all parameters
      def parameters : Array(Autograd::Variable)
        @param_groups.flat_map(&.params)
      end

      # State dict for checkpointing
      def state_dict : Hash(String, {Tensor, Tensor, Int32})
        result = Hash(String, {Tensor, Tensor, Int32}).new
        @state.each do |id, s|
          result[id.to_s] = {s.m, s.v, s.step}
        end
        result
      end

      # Load state dict
      def load_state_dict(dict : Hash(String, {Tensor, Tensor, Int32})) : Nil
        dict.each do |id_str, (m, v, step)|
          id = id_str.to_u64
          if state = @state[id]?
            state.m = m
            state.v = v
            state.step = step
          end
        end
      end
    end

    # SGD optimizer (for reference/comparison)
    class SGD
      property lr : Float32
      property momentum : Float32
      property weight_decay : Float32
      property dampening : Float32
      property nesterov : Bool

      @params : Array(Autograd::Variable)
      @velocity : Hash(UInt64, Tensor)

      def initialize(
        @params : Array(Autograd::Variable),
        @lr : Float32 = 0.01_f32,
        @momentum : Float32 = 0.0_f32,
        @weight_decay : Float32 = 0.0_f32,
        @dampening : Float32 = 0.0_f32,
        @nesterov : Bool = false
      )
        @velocity = Hash(UInt64, Tensor).new
      end

      def step : Nil
        @params.each do |param|
          next unless param.requires_grad?
          next unless param.grad

          grad = param.grad.not_nil!
          param_data = param.data

          # Apply weight decay
          if @weight_decay > 0
            grad_cpu = grad.on_cpu? ? grad : grad.to_cpu
            param_cpu = param_data.on_cpu? ? param_data : param_data.to_cpu
            grad_data = grad_cpu.cpu_data.not_nil!
            param_d = param_cpu.cpu_data.not_nil!
            grad.numel.times { |i| grad_data[i] += @weight_decay * param_d[i] }
          end

          if @momentum > 0
            v = @velocity[param.object_id] ||= begin
              t = Tensor.new(param_data.shape, DType::F32, param_data.device)
              t.fill!(0.0_f32)
              t
            end

            # v = momentum * v + (1 - dampening) * grad
            # if nesterov: param = param - lr * (grad + momentum * v)
            # else: param = param - lr * v

            v_cpu = v.on_cpu? ? v : v.to_cpu
            grad_cpu = grad.on_cpu? ? grad : grad.to_cpu
            param_cpu = param_data.on_cpu? ? param_data : param_data.to_cpu

            v_data = v_cpu.cpu_data.not_nil!
            g_data = grad_cpu.cpu_data.not_nil!
            p_data = param_cpu.cpu_data.not_nil!

            param.numel.times do |i|
              v_data[i] = @momentum * v_data[i] + (1.0_f32 - @dampening) * g_data[i]
              if @nesterov
                p_data[i] -= @lr * (g_data[i] + @momentum * v_data[i])
              else
                p_data[i] -= @lr * v_data[i]
              end
            end

            # TODO: sync back to GPU if needed
          else
            # Simple SGD: param = param - lr * grad
            grad_cpu = grad.on_cpu? ? grad : grad.to_cpu
            param_was_gpu = param_data.on_gpu?
            param_data.to_cpu! if param_was_gpu

            g_data = grad_cpu.cpu_data.not_nil!
            p_data = param_data.cpu_data.not_nil!

            param.numel.times { |i| p_data[i] -= @lr * g_data[i] }

            param_data.to_gpu! if param_was_gpu
          end
        end
      end

      def zero_grad : Nil
        @params.each(&.zero_grad!)
      end
    end

    # Learning rate scheduler base
    abstract class LRScheduler
      abstract def step : Nil
      abstract def get_lr : Float32
    end

    # Step decay scheduler
    class StepLR < LRScheduler
      @optimizer : Adam
      @step_size : Int32
      @gamma : Float32
      @current_step : Int32
      @base_lr : Float32

      def initialize(@optimizer : Adam, @step_size : Int32, @gamma : Float32 = 0.1_f32)
        @current_step = 0
        @base_lr = @optimizer.lr
      end

      def step : Nil
        @current_step += 1
        if @current_step % @step_size == 0
          @optimizer.lr = @base_lr * (@gamma ** (@current_step // @step_size))
        end
      end

      def get_lr : Float32
        @optimizer.lr
      end
    end

    # Exponential decay scheduler
    class ExponentialLR < LRScheduler
      @optimizer : Adam
      @gamma : Float32

      def initialize(@optimizer : Adam, @gamma : Float32 = 0.99_f32)
      end

      def step : Nil
        @optimizer.lr *= @gamma
      end

      def get_lr : Float32
        @optimizer.lr
      end
    end
  end
end
