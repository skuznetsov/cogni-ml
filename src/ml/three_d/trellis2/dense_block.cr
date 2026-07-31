# CPU-only F32 reference implementation of the TRELLIS.2 shared-modulated
# transformer cross block.  This intentionally keeps the native boundary small:
# parameters use the existing Linear/MLP modules, while the head-shaped
# attention and rotary paths are evaluated by a deterministic CPU reference.

require "../../autograd/variable"
require "../../core/tensor"
require "../../nn/linear"
require "../../nn/layernorm"
require "../../nn/vit"

module ML::ThreeD::Trellis2
  module DenseBlockCPU
    extend self

    def reject_gpu!(tensor : Tensor, name : String) : Nil
      raise ArgumentError.new("#{name} must be a CPU tensor") if tensor.on_gpu?
      raise ArgumentError.new("#{name} must use F32") unless tensor.dtype.f32?
    end

    def tensor_from(data : Array(Float32), shape : Shape) : Tensor
      Tensor.from_array(data, shape)
    end

    def add(a : Tensor, b : Tensor) : Tensor
      raise ArgumentError.new("tensor add shape mismatch: #{a.shape} vs #{b.shape}") unless a.shape == b.shape
      reject_gpu!(a, "left tensor")
      reject_gpu!(b, "right tensor")
      ad = a.to_contiguous_cpu.cpu_data.not_nil!
      bd = b.to_contiguous_cpu.cpu_data.not_nil!
      result_values = Array(Float32).new(a.numel, 0.0_f32)
      a.numel.times { |i| result_values[i] = ad[i] + bd[i] }
      tensor_from(result_values, a.shape)
    end

    # Elementwise [batch, sequence, channels] * [batch, channels] broadcast.
    def mul_last_broadcast(a : Tensor, b : Tensor) : Tensor
      raise ArgumentError.new("broadcast multiply expects a rank-3 tensor") unless a.ndim == 3
      raise ArgumentError.new("broadcast multiply expects a rank-2 tensor") unless b.ndim == 2
      unless a.shape[0] == b.shape[0] && a.shape[2] == b.shape[1]
        raise ArgumentError.new("broadcast multiply shape mismatch: #{a.shape} vs #{b.shape}")
      end
      reject_gpu!(a, "left tensor")
      reject_gpu!(b, "broadcast tensor")
      ad = a.to_contiguous_cpu.cpu_data.not_nil!
      bd = b.to_contiguous_cpu.cpu_data.not_nil!
      batch = a.shape[0]
      length = a.shape[1]
      channels = a.shape[2]
      result_values = Array(Float32).new(a.numel, 0.0_f32)
      batch.times do |batch_index|
        length.times do |position|
          channels.times do |channel|
            offset = (batch_index * length + position) * channels + channel
            result_values[offset] = ad[offset] * bd[batch_index * channels + channel]
          end
        end
      end
      tensor_from(result_values, a.shape)
    end

    # Elementwise [batch, sequence, channels] * (1 + [batch, channels]).
    def affine_last(a : Tensor, scale : Tensor, shift : Tensor) : Tensor
      raise ArgumentError.new("affine expects rank-3 input") unless a.ndim == 3
      raise ArgumentError.new("affine scale and shift must be rank-2") unless scale.ndim == 2 && shift.ndim == 2
      unless a.shape[0] == scale.shape[0] && scale.shape == shift.shape && a.shape[2] == scale.shape[1]
        raise ArgumentError.new("affine shape mismatch: #{a.shape}, #{scale.shape}, #{shift.shape}")
      end
      reject_gpu!(a, "affine input")
      reject_gpu!(scale, "affine scale")
      reject_gpu!(shift, "affine shift")
      ad = a.to_contiguous_cpu.cpu_data.not_nil!
      sd = scale.to_contiguous_cpu.cpu_data.not_nil!
      hd = shift.to_contiguous_cpu.cpu_data.not_nil!
      batch = a.shape[0]
      length = a.shape[1]
      channels = a.shape[2]
      result_values = Array(Float32).new(a.numel, 0.0_f32)
      batch.times do |batch_index|
        length.times do |position|
          channels.times do |channel|
            offset = (batch_index * length + position) * channels + channel
            condition_offset = batch_index * channels + channel
            result_values[offset] = ad[offset] * (1.0_f32 + sd[condition_offset]) + hd[condition_offset]
          end
        end
      end
      tensor_from(result_values, a.shape)
    end

    def linear(linear : ML::NN::Linear, input : Tensor) : Tensor
      reject_gpu!(input, "linear input")
      output = linear.forward(Autograd::Variable.new(input, requires_grad: false)).data
      reject_gpu!(output, "linear output")
      output.to_contiguous_cpu
    end

    def ensure_cpu!(variable : Autograd::Variable, name : String) : Tensor
      reject_gpu!(variable.data, name)
      variable.data
    end
  end

  # LayerNorm32(channels, elementwise_affine=False, eps=1e-6) from TRELLIS.2.
  class NonAffineLayerNorm
    getter channels : Int32
    getter eps : Float32

    def initialize(@channels : Int32, @eps : Float32 = 1e-6_f32, device : Tensor::Device = Tensor::Device::CPU)
      raise ArgumentError.new("NonAffineLayerNorm channels must be positive") unless @channels > 0
      raise ArgumentError.new("NonAffineLayerNorm eps must be finite and positive") unless @eps.finite? && @eps > 0.0_f32
      raise ArgumentError.new("NonAffineLayerNorm is CPU-only") if device.gpu?
    end

    def forward(x : Autograd::Variable) : Autograd::Variable
      input = DenseBlockCPU.ensure_cpu!(x, "LayerNorm input")
      unless input.ndim >= 1 && input.shape[-1] == @channels
        raise ArgumentError.new("LayerNorm input trailing dimension #{input.shape[-1]} does not match #{@channels}")
      end

      input_cpu = input.to_contiguous_cpu
      values = input_cpu.cpu_data.not_nil!
      result_values = Array(Float32).new(input.numel, 0.0_f32)
      rows = input.numel // @channels
      rows.times do |row|
        offset = row * @channels
        sum = 0.0_f64
        @channels.times { |i| sum += values[offset + i].to_f64 }
        mean = sum / @channels.to_f64
        squared = 0.0_f64
        @channels.times do |i|
          delta = values[offset + i].to_f64 - mean
          squared += delta * delta
        end
        inv_std = 1.0_f64 / Math.sqrt(squared / @channels.to_f64 + @eps.to_f64)
        @channels.times do |i|
          result_values[offset + i] = ((values[offset + i].to_f64 - mean) * inv_std).to_f32
        end
      end
      Autograd::Variable.new(DenseBlockCPU.tensor_from(result_values, input.shape), requires_grad: false)
    end

    def call(x : Autograd::Variable) : Autograd::Variable
      forward(x)
    end
  end

  # TRELLIS.2's per-head normalization:
  # F.normalize(x.float(), dim=-1) * gamma * sqrt(head_dim).
  class MultiHeadRMSNorm
    getter dim : Int32
    getter heads : Int32
    getter gamma : Autograd::Variable
    getter eps : Float32

    def initialize(@dim : Int32, @heads : Int32, device : Tensor::Device = Tensor::Device::CPU, @eps : Float32 = 1e-12_f32)
      raise ArgumentError.new("MultiHeadRMSNorm dim must be positive") unless @dim > 0
      raise ArgumentError.new("MultiHeadRMSNorm heads must be positive") unless @heads > 0
      raise ArgumentError.new("MultiHeadRMSNorm eps must be finite and positive") unless @eps.finite? && @eps > 0.0_f32
      raise ArgumentError.new("MultiHeadRMSNorm is CPU-only") if device.gpu?
      @gamma = Autograd::Variable.new(Tensor.ones(@heads, @dim, device: Tensor::Device::CPU), requires_grad: true)
    end

    def forward(x : Autograd::Variable) : Autograd::Variable
      input = DenseBlockCPU.ensure_cpu!(x, "MultiHeadRMSNorm input")
      unless input.ndim == 4 && input.shape[2] == @heads && input.shape[3] == @dim
        raise ArgumentError.new(
          "MultiHeadRMSNorm input must have shape [batch, sequence, #{@heads}, #{@dim}], got #{input.shape}"
        )
      end

      values = input.to_contiguous_cpu.cpu_data.not_nil!
      gamma_values = @gamma.data.cpu_data.not_nil!
      result_values = Array(Float32).new(input.numel, 0.0_f32)
      rows = input.shape[0] * input.shape[1] * @heads
      scale = Math.sqrt(@dim.to_f64)
      rows.times do |row|
        offset = row * @dim
        sum_sq = 0.0_f64
        @dim.times do |i|
          value = values[offset + i].to_f64
          sum_sq += value * value
        end
        norm = Math.sqrt(sum_sq)
        norm = @eps.to_f64 if norm < @eps.to_f64
        @dim.times do |i|
          result_values[offset + i] = (values[offset + i].to_f64 / norm * gamma_values[(row % @heads) * @dim + i].to_f64 * scale).to_f32
        end
      end
      Autograd::Variable.new(DenseBlockCPU.tensor_from(result_values, input.shape), requires_grad: false)
    end

    def call(x : Autograd::Variable) : Autograd::Variable
      forward(x)
    end
  end

  record AttentionTrace,
    output : Tensor,
    q : Tensor,
    k : Tensor

  # Dense self-attention helper exposing the parameter layout used upstream.
  class SelfAttention
    getter channels : Int32
    getter num_heads : Int32
    getter head_dim : Int32
    getter to_qkv : ML::NN::Linear
    getter to_out : ML::NN::Linear
    getter q_rms_norm : MultiHeadRMSNorm
    getter k_rms_norm : MultiHeadRMSNorm
    getter qk_rms_norm : Bool
    getter use_rope : Bool

    def initialize(
      @channels : Int32,
      @num_heads : Int32,
      @qk_rms_norm : Bool = true,
      @use_rope : Bool = true,
      device : Tensor::Device = Tensor::Device::CPU,
    )
      raise ArgumentError.new("SelfAttention CPU slice requires qk_rms_norm=true") unless @qk_rms_norm
      raise ArgumentError.new("SelfAttention CPU slice requires use_rope=true") unless @use_rope
      raise ArgumentError.new("SelfAttention channels must be divisible by num_heads") if @channels <= 0 || @num_heads <= 0 || @channels % @num_heads != 0
      @head_dim = @channels // @num_heads
      raise ArgumentError.new("SelfAttention head dimension must be even for RoPE") if @use_rope && !@head_dim.even?
      raise ArgumentError.new("SelfAttention is CPU-only") if device.gpu?
      @to_qkv = ML::NN::Linear.new(@channels, 3 * @channels, device: Tensor::Device::CPU)
      @to_out = ML::NN::Linear.new(@channels, @channels, device: Tensor::Device::CPU)
      # The pinned block enables Q/K RMSNorm. Keep helper fields non-nil so
      # callers can load the exact state-dict names used by the fixture.
      @q_rms_norm = MultiHeadRMSNorm.new(@head_dim, @num_heads, device: Tensor::Device::CPU)
      @k_rms_norm = MultiHeadRMSNorm.new(@head_dim, @num_heads, device: Tensor::Device::CPU)
    end

    def forward(x : Autograd::Variable, phases : Tensor? = nil) : Autograd::Variable
      result = forward_with_trace(x, phases)
      Autograd::Variable.new(result.output, requires_grad: false)
    end

    def call(x : Autograd::Variable, phases : Tensor? = nil) : Autograd::Variable
      forward(x, phases)
    end

    def forward_with_trace(x : Autograd::Variable, phases : Tensor? = nil) : AttentionTrace
      input = DenseBlockCPU.ensure_cpu!(x, "self-attention input")
      unless input.ndim == 3 && input.shape[2] == @channels && input.shape[0] > 0 && input.shape[1] > 0
        raise ArgumentError.new("self-attention input must have shape [batch, sequence, #{@channels}], got #{input.shape}")
      end
      length = input.shape[1]
      if @use_rope
        raise ArgumentError.new("self-attention RoPE phases are required") unless phases
        validate_phases!(phases.not_nil!, length)
      elsif phases
        DenseBlockCPU.reject_gpu!(phases, "RoPE phases")
      end

      qkv = DenseBlockCPU.linear(@to_qkv, input)
      qkv_values = qkv.cpu_data.not_nil!
      batch = input.shape[0]
      q = Tensor.new(batch, length, @num_heads, @head_dim, device: Tensor::Device::CPU)
      k = Tensor.new(batch, length, @num_heads, @head_dim, device: Tensor::Device::CPU)
      v = Tensor.new(batch, length, @num_heads, @head_dim, device: Tensor::Device::CPU)
      q_values = q.cpu_data.not_nil!
      k_values = k.cpu_data.not_nil!
      v_values = v.cpu_data.not_nil!
      batch.times do |batch_index|
        length.times do |position|
          @num_heads.times do |head|
            @head_dim.times do |dimension|
              src = ((batch_index * length + position) * 3 * @channels) +
                    ((head * @head_dim) + dimension)
              q_offset = ((batch_index * length + position) * @num_heads + head) * @head_dim + dimension
              k_offset = q_offset
              v_offset = q_offset
              q_values[q_offset] = qkv_values[src]
              k_values[k_offset] = qkv_values[src + @channels]
              v_values[v_offset] = qkv_values[src + 2 * @channels]
            end
          end
        end
      end

      q_var = @q_rms_norm.forward(Autograd::Variable.new(q, requires_grad: false))
      k_var = @k_rms_norm.forward(Autograd::Variable.new(k, requires_grad: false))
      q = q_var.data
      k = k_var.data
      if @use_rope
        q = rotary(q, phases.not_nil!, length)
        k = rotary(k, phases.not_nil!, length)
      end

      attended = scaled_attention(q, k, v)
      attended_flat = attended.reshape(batch, length, @channels)
      output = DenseBlockCPU.linear(@to_out, attended_flat)
      AttentionTrace.new(output, q, k)
    end

    private def validate_phases!(phases : Tensor, length : Int32) : Nil
      DenseBlockCPU.reject_gpu!(phases, "RoPE phases")
      expected = [length, @head_dim // 2, 2]
      unless phases.shape.to_a == expected
        raise ArgumentError.new("RoPE phases shape #{phases.shape} does not match #{expected}")
      end
    end

    private def rotary(x : Tensor, phases : Tensor, length : Int32) : Tensor
      values = x.to_contiguous_cpu.cpu_data.not_nil!
      phase_values = phases.to_contiguous_cpu.cpu_data.not_nil!
      batch = x.shape[0]
      result_values = Array(Float32).new(x.numel, 0.0_f32)
      batch.times do |batch_index|
        length.times do |position|
          @num_heads.times do |head|
            pair_count = @head_dim // 2
            pair_count.times do |pair|
              phase_offset = (position * pair_count + pair) * 2
              cos = phase_values[phase_offset]
              sin = phase_values[phase_offset + 1]
              offset = ((batch_index * length + position) * @num_heads + head) * @head_dim + pair * 2
              real = values[offset]
              imag = values[offset + 1]
              result_values[offset] = real * cos - imag * sin
              result_values[offset + 1] = real * sin + imag * cos
            end
          end
        end
      end
      DenseBlockCPU.tensor_from(result_values, x.shape)
    end

    private def scaled_attention(q : Tensor, k : Tensor, v : Tensor) : Tensor
      batch = q.shape[0]
      length = q.shape[1]
      values_q = q.to_contiguous_cpu.cpu_data.not_nil!
      values_k = k.to_contiguous_cpu.cpu_data.not_nil!
      values_v = v.to_contiguous_cpu.cpu_data.not_nil!
      result_values = Array(Float32).new(q.numel, 0.0_f32)
      scale = Math.sqrt(@head_dim.to_f64)
      scores = Array(Float64).new(length, 0.0)
      weights = Array(Float64).new(length, 0.0)
      batch.times do |batch_index|
        @num_heads.times do |head|
          length.times do |query_position|
            max_score = -Float64::INFINITY
            length.times do |key_position|
              sum = 0.0_f64
              @head_dim.times do |dimension|
                q_offset = ((batch_index * length + query_position) * @num_heads + head) * @head_dim + dimension
                k_offset = ((batch_index * length + key_position) * @num_heads + head) * @head_dim + dimension
                sum += values_q[q_offset].to_f64 * values_k[k_offset].to_f64
              end
              score = sum / scale
              scores[key_position] = score
              max_score = score if score > max_score
            end
            total = 0.0_f64
            length.times do |key_position|
              weight = Math.exp(scores[key_position] - max_score)
              weights[key_position] = weight
              total += weight
            end
            length.times { |key_position| weights[key_position] /= total }
            @head_dim.times do |dimension|
              value = 0.0_f64
              length.times do |key_position|
                v_offset = ((batch_index * length + key_position) * @num_heads + head) * @head_dim + dimension
                value += weights[key_position] * values_v[v_offset].to_f64
              end
              result_values[((batch_index * length + query_position) * @num_heads + head) * @head_dim + dimension] = value.to_f32
            end
          end
        end
      end
      DenseBlockCPU.tensor_from(result_values, q.shape)
    end
  end

  # Dense cross-attention helper exposing the parameter layout used upstream.
  class CrossAttention
    getter channels : Int32
    getter context_channels : Int32
    getter num_heads : Int32
    getter head_dim : Int32
    getter to_q : ML::NN::Linear
    getter to_kv : ML::NN::Linear
    getter to_out : ML::NN::Linear
    getter q_rms_norm : MultiHeadRMSNorm
    getter k_rms_norm : MultiHeadRMSNorm
    getter qk_rms_norm : Bool

    def initialize(
      @channels : Int32,
      @context_channels : Int32,
      @num_heads : Int32,
      @qk_rms_norm : Bool = true,
      device : Tensor::Device = Tensor::Device::CPU,
    )
      raise ArgumentError.new("CrossAttention CPU slice requires qk_rms_norm=true") unless @qk_rms_norm
      raise ArgumentError.new("CrossAttention channels must be divisible by num_heads") if @channels <= 0 || @context_channels <= 0 || @num_heads <= 0 || @channels % @num_heads != 0
      @head_dim = @channels // @num_heads
      raise ArgumentError.new("CrossAttention is CPU-only") if device.gpu?
      @to_q = ML::NN::Linear.new(@channels, @channels, device: Tensor::Device::CPU)
      @to_kv = ML::NN::Linear.new(@context_channels, 2 * @channels, device: Tensor::Device::CPU)
      @to_out = ML::NN::Linear.new(@channels, @channels, device: Tensor::Device::CPU)
      @q_rms_norm = MultiHeadRMSNorm.new(@head_dim, @num_heads, device: Tensor::Device::CPU)
      @k_rms_norm = MultiHeadRMSNorm.new(@head_dim, @num_heads, device: Tensor::Device::CPU)
    end

    def forward(x : Autograd::Variable, context : Autograd::Variable) : Autograd::Variable
      result = forward_with_trace(x, context)
      Autograd::Variable.new(result.output, requires_grad: false)
    end

    def call(x : Autograd::Variable, context : Autograd::Variable) : Autograd::Variable
      forward(x, context)
    end

    def forward_with_trace(x : Autograd::Variable, context : Autograd::Variable) : AttentionTrace
      input = DenseBlockCPU.ensure_cpu!(x, "cross-attention input")
      memory = DenseBlockCPU.ensure_cpu!(context, "cross-attention context")
      unless input.ndim == 3 && input.shape[2] == @channels && input.shape[0] > 0 && input.shape[1] > 0
        raise ArgumentError.new("cross-attention input must have shape [batch, sequence, #{@channels}], got #{input.shape}")
      end
      unless memory.ndim == 3 && memory.shape[2] == @context_channels && memory.shape[0] == input.shape[0] && memory.shape[1] > 0
        raise ArgumentError.new("cross-attention context must have shape [batch, sequence, #{@context_channels}], got #{memory.shape}")
      end
      batch = input.shape[0]
      length = input.shape[1]
      context_length = memory.shape[1]

      q_projected = DenseBlockCPU.linear(@to_q, input)
      kv_projected = DenseBlockCPU.linear(@to_kv, memory)
      q = Tensor.new(batch, length, @num_heads, @head_dim, device: Tensor::Device::CPU)
      k = Tensor.new(batch, context_length, @num_heads, @head_dim, device: Tensor::Device::CPU)
      v = Tensor.new(batch, context_length, @num_heads, @head_dim, device: Tensor::Device::CPU)
      q_values = q.cpu_data.not_nil!
      k_values = k.cpu_data.not_nil!
      v_values = v.cpu_data.not_nil!
      q_projected_values = q_projected.cpu_data.not_nil!
      kv_values = kv_projected.cpu_data.not_nil!
      batch.times do |batch_index|
        length.times do |position|
          @num_heads.times do |head|
            @head_dim.times do |dimension|
              q_offset = ((batch_index * length + position) * @num_heads + head) * @head_dim + dimension
              q_values[q_offset] = q_projected_values[(batch_index * length + position) * @channels + head * @head_dim + dimension]
            end
          end
        end
        context_length.times do |position|
          @num_heads.times do |head|
            @head_dim.times do |dimension|
              offset = ((batch_index * context_length + position) * @num_heads + head) * @head_dim + dimension
              source = (batch_index * context_length + position) * 2 * @channels + head * @head_dim + dimension
              k_values[offset] = kv_values[source]
              v_values[offset] = kv_values[source + @channels]
            end
          end
        end
      end

      q = @q_rms_norm.forward(Autograd::Variable.new(q, requires_grad: false)).data
      k = @k_rms_norm.forward(Autograd::Variable.new(k, requires_grad: false)).data
      attended = scaled_attention(q, k, v, length, context_length)
      output = DenseBlockCPU.linear(@to_out, attended.reshape(batch, length, @channels))
      AttentionTrace.new(output, q, k)
    end

    private def scaled_attention(q : Tensor, k : Tensor, v : Tensor, query_length : Int32, key_length : Int32) : Tensor
      batch = q.shape[0]
      values_q = q.to_contiguous_cpu.cpu_data.not_nil!
      values_k = k.to_contiguous_cpu.cpu_data.not_nil!
      values_v = v.to_contiguous_cpu.cpu_data.not_nil!
      result_values = Array(Float32).new(q.numel, 0.0_f32)
      scale = Math.sqrt(@head_dim.to_f64)
      scores = Array(Float64).new(key_length, 0.0)
      weights = Array(Float64).new(key_length, 0.0)
      batch.times do |batch_index|
        @num_heads.times do |head|
          query_length.times do |query_position|
            max_score = -Float64::INFINITY
            key_length.times do |key_position|
              sum = 0.0_f64
              @head_dim.times do |dimension|
                q_offset = ((batch_index * query_length + query_position) * @num_heads + head) * @head_dim + dimension
                k_offset = ((batch_index * key_length + key_position) * @num_heads + head) * @head_dim + dimension
                sum += values_q[q_offset].to_f64 * values_k[k_offset].to_f64
              end
              score = sum / scale
              scores[key_position] = score
              max_score = score if score > max_score
            end
            total = 0.0_f64
            key_length.times do |key_position|
              weight = Math.exp(scores[key_position] - max_score)
              weights[key_position] = weight
              total += weight
            end
            key_length.times { |key_position| weights[key_position] /= total }
            @head_dim.times do |dimension|
              value = 0.0_f64
              key_length.times do |key_position|
                v_offset = ((batch_index * key_length + key_position) * @num_heads + head) * @head_dim + dimension
                value += weights[key_position] * values_v[v_offset].to_f64
              end
              result_values[((batch_index * query_length + query_position) * @num_heads + head) * @head_dim + dimension] = value.to_f32
            end
          end
        end
      end
      DenseBlockCPU.tensor_from(result_values, Shape.new(batch, query_length, @num_heads, @head_dim))
    end
  end

  class SharedModulatedTransformerCrossBlock
    MAX_REFERENCE_PARAMETER_BYTES    = 64_i64 * 1024_i64 * 1024_i64
    MAX_REFERENCE_PARAMETER_ELEMENTS = MAX_REFERENCE_PARAMETER_BYTES // 4_i64

    getter channels : Int32
    getter context_channels : Int32
    getter num_heads : Int32
    getter mlp_ratio : Float32
    getter parameter_elements : Int64
    getter modulation : Autograd::Variable
    getter norm1 : NonAffineLayerNorm
    getter norm2 : ML::NN::LayerNorm
    getter norm3 : NonAffineLayerNorm
    getter self_attn : SelfAttention
    getter cross_attn : CrossAttention
    getter mlp : ML::NN::MLP
    getter eps : Float32

    def initialize(
      @channels : Int32,
      @context_channels : Int32,
      @num_heads : Int32,
      @mlp_ratio : Float32 = 4.0_f32,
      device : Tensor::Device = Tensor::Device::CPU,
      qk_rms_norm : Bool = true,
      qk_rms_norm_cross : Bool = true,
      use_rope : Bool = true,
      share_mod : Bool = true,
      @eps : Float32 = 1e-6_f32,
    )
      raise ArgumentError.new("SharedModulatedTransformerCrossBlock is CPU-only") if device.gpu?
      raise ArgumentError.new("only shared modulation is admitted by this CPU slice") unless share_mod
      raise ArgumentError.new("SharedModulatedTransformerCrossBlock CPU slice requires qk_rms_norm=true") unless qk_rms_norm
      raise ArgumentError.new("SharedModulatedTransformerCrossBlock CPU slice requires qk_rms_norm_cross=true") unless qk_rms_norm_cross
      raise ArgumentError.new("SharedModulatedTransformerCrossBlock CPU slice requires use_rope=true") unless use_rope
      raise ArgumentError.new("channels must be positive") unless @channels > 0
      raise ArgumentError.new("context_channels must be positive") unless @context_channels > 0
      raise ArgumentError.new("num_heads must be positive and divide channels") unless @num_heads > 0 && @channels % @num_heads == 0
      raise ArgumentError.new("mlp_ratio must be finite and positive") unless @mlp_ratio.finite? && @mlp_ratio > 0.0_f32
      raise ArgumentError.new("eps must be finite and positive") unless @eps.finite? && @eps > 0.0_f32
      raise ArgumentError.new("channels overflow shared modulation width") if @channels.to_i64 * 6_i64 > Int32::MAX
      hidden_value = @channels.to_f64 * @mlp_ratio.to_f64
      unless 1.0 <= hidden_value <= Int32::MAX
        raise ArgumentError.new("mlp hidden dimension must be positive and representable")
      end
      hidden = hidden_value.to_i32
      c = @channels.to_i64
      k = @context_channels.to_i64
      m = hidden.to_i64
      @parameter_elements =
        6_i64 * c +   # shared block modulation
          2_i64 * c + # affine norm2
          3_i64 * c * c + 3_i64 * c +
          c * c + c +
          2_i64 * c + # self Q/K RMS gamma
          c * c + c +
          2_i64 * c * k + 2_i64 * c +
          2_i64 * c + # cross Q/K RMS gamma
          c * c + c +
          m * c + m +
          c * m + c
      if @parameter_elements > MAX_REFERENCE_PARAMETER_ELEMENTS
        raise ArgumentError.new(
          "dense CPU oracle parameter budget #{@parameter_elements} elements exceeds " \
          "#{MAX_REFERENCE_PARAMETER_ELEMENTS}"
        )
      end

      @modulation = Autograd::Variable.new(Tensor.zeros(6 * @channels, device: Tensor::Device::CPU), requires_grad: false)
      @norm1 = NonAffineLayerNorm.new(@channels, @eps, device: Tensor::Device::CPU)
      @norm2 = ML::NN::LayerNorm.new(@channels, eps: @eps, device: Tensor::Device::CPU)
      @norm3 = NonAffineLayerNorm.new(@channels, @eps, device: Tensor::Device::CPU)
      @self_attn = SelfAttention.new(@channels, @num_heads, qk_rms_norm: qk_rms_norm, use_rope: use_rope, device: Tensor::Device::CPU)
      @cross_attn = CrossAttention.new(@channels, @context_channels, @num_heads, qk_rms_norm: qk_rms_norm_cross, device: Tensor::Device::CPU)
      @mlp = ML::NN::MLP.new(@channels, hidden_features: hidden, out_features: @channels, device: Tensor::Device::CPU)
      freeze_parameters!
    end

    def forward(
      x : Autograd::Variable,
      mod : Autograd::Variable,
      context : Autograd::Variable,
      phases : Tensor? = nil,
    ) : Autograd::Variable
      trace = forward_with_trace(x, mod, context, phases)
      Autograd::Variable.new(trace["output"], requires_grad: false)
    end

    def call(
      x : Autograd::Variable,
      mod : Autograd::Variable,
      context : Autograd::Variable,
      phases : Tensor? = nil,
    ) : Autograd::Variable
      forward(x, mod, context, phases)
    end

    def forward_with_trace(
      x : Autograd::Variable,
      mod : Autograd::Variable,
      context : Autograd::Variable,
      phases : Tensor? = nil,
    ) : Hash(String, Tensor)
      # Upstream complex RoPE phases cross this boundary as real-pair F32
      # `[sequence, head_dim / 2, 2]` values. Coordinate-to-phase generation
      # and timestep embedding remain separate, unadmitted slices.
      input = DenseBlockCPU.ensure_cpu!(x, "block input")
      modulation_input = DenseBlockCPU.ensure_cpu!(mod, "block modulation input")
      memory = DenseBlockCPU.ensure_cpu!(context, "block context")
      validate_block_inputs!(input, modulation_input, memory)

      combined = add_modulation(modulation_input)
      # Tensor has no slicing primitive; chunk the contiguous CPU payload below.
      combined_chunks = split_modulation(combined, input.shape[0])
      shift_msa = combined_chunks[0]
      scale_msa = combined_chunks[1]
      gate_msa = combined_chunks[2]
      shift_mlp = combined_chunks[3]
      scale_mlp = combined_chunks[4]
      gate_mlp = combined_chunks[5]

      norm1 = @norm1.forward(x).data
      modulated_self = DenseBlockCPU.affine_last(norm1, scale_msa, shift_msa)
      self_trace = @self_attn.forward_with_trace(Autograd::Variable.new(modulated_self, requires_grad: false), phases)
      gated_self = DenseBlockCPU.mul_last_broadcast(self_trace.output, gate_msa)
      after_self = DenseBlockCPU.add(input, gated_self)

      norm2 = @norm2.forward(Autograd::Variable.new(after_self, requires_grad: false)).data
      cross_trace = @cross_attn.forward_with_trace(
        Autograd::Variable.new(norm2, requires_grad: false),
        context
      )
      after_cross = DenseBlockCPU.add(after_self, cross_trace.output)

      norm3 = @norm3.forward(Autograd::Variable.new(after_cross, requires_grad: false)).data
      modulated_mlp = DenseBlockCPU.affine_last(norm3, scale_mlp, shift_mlp)
      mlp_output = @mlp.forward(Autograd::Variable.new(modulated_mlp, requires_grad: false)).data.to_contiguous_cpu
      gated_mlp = DenseBlockCPU.mul_last_broadcast(mlp_output, gate_mlp)
      output = DenseBlockCPU.add(after_cross, gated_mlp)

      {
        "combined_mod"    => combined,
        "norm1"           => norm1,
        "modulated_self"  => modulated_self,
        "self_q"          => self_trace.q,
        "self_k"          => self_trace.k,
        "self_attention"  => self_trace.output,
        "after_self"      => after_self,
        "norm2"           => norm2,
        "cross_q"         => cross_trace.q,
        "cross_k"         => cross_trace.k,
        "cross_attention" => cross_trace.output,
        "after_cross"     => after_cross,
        "norm3"           => norm3,
        "modulated_mlp"   => modulated_mlp,
        "mlp_output"      => mlp_output,
        "output"          => output,
      }
    end

    private def validate_block_inputs!(x : Tensor, mod : Tensor, context : Tensor) : Nil
      unless x.ndim == 3 && x.shape[0] > 0 && x.shape[1] > 0 && x.shape[2] == @channels
        raise ArgumentError.new("block input must have shape [batch, sequence, #{@channels}], got #{x.shape}")
      end
      unless mod.ndim == 2 && mod.shape[0] == x.shape[0] && mod.shape[1] == 6 * @channels
        raise ArgumentError.new("block modulation must have shape [#{x.shape[0]}, #{6 * @channels}], got #{mod.shape}")
      end
      unless context.ndim == 3 && context.shape[0] == x.shape[0] && context.shape[1] > 0 && context.shape[2] == @context_channels
        raise ArgumentError.new("block context must have shape [#{x.shape[0]}, sequence, #{@context_channels}], got #{context.shape}")
      end
    end

    private def freeze_parameters! : Nil
      freeze_linear(@self_attn.to_qkv)
      freeze_linear(@self_attn.to_out)
      freeze_linear(@cross_attn.to_q)
      freeze_linear(@cross_attn.to_kv)
      freeze_linear(@cross_attn.to_out)
      freeze_linear(@mlp.fc1)
      freeze_linear(@mlp.fc2)
      @norm2.weight.requires_grad = false
      @norm2.bias.requires_grad = false
      @self_attn.q_rms_norm.gamma.requires_grad = false
      @self_attn.k_rms_norm.gamma.requires_grad = false
      @cross_attn.q_rms_norm.gamma.requires_grad = false
      @cross_attn.k_rms_norm.gamma.requires_grad = false
      @modulation.requires_grad = false
    end

    private def freeze_linear(linear : ML::NN::Linear) : Nil
      linear.weight.requires_grad = false
      if bias = linear.bias
        bias.requires_grad = false
      end
    end

    private def add_modulation(mod : Tensor) : Tensor
      base = @modulation.data.to_contiguous_cpu
      base_values = base.cpu_data.not_nil!
      mod_values = mod.to_contiguous_cpu.cpu_data.not_nil!
      batch = mod.shape[0]
      result_values = Array(Float32).new(mod.numel, 0.0_f32)
      batch.times do |batch_index|
        @modulation.data.numel.times do |i|
          result_values[batch_index * @modulation.data.numel + i] = mod_values[batch_index * @modulation.data.numel + i] + base_values[i]
        end
      end
      DenseBlockCPU.tensor_from(result_values, mod.shape)
    end

    private def split_modulation(combined : Tensor, batch : Int32) : Array(Tensor)
      values = combined.to_contiguous_cpu.cpu_data.not_nil!
      chunks = Array(Tensor).new(6)
      6.times do |chunk_index|
        data = Array(Float32).new(batch * @channels, 0.0_f32)
        batch.times do |batch_index|
          @channels.times do |channel|
            data[batch_index * @channels + channel] = values[batch_index * 6 * @channels + chunk_index * @channels + channel]
          end
        end
        chunks << DenseBlockCPU.tensor_from(data, Shape.new(batch, @channels))
      end
      chunks
    end
  end
end
