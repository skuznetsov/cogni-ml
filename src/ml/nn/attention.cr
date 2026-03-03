# Multi-Head Attention
# Core mechanism for Transformer architectures

require "../autograd/variable"
require "../core/tensor"
require "./linear"
require "./gpu_ops"

module ML
  module NN
    # Multi-Head Attention
    # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    class MultiHeadAttention
      getter embed_dim : Int32
      getter num_heads : Int32
      getter head_dim : Int32
      getter dropout : Float32

      # Projections
      getter q_proj : Linear
      getter k_proj : Linear
      getter v_proj : Linear
      getter out_proj : Linear

      def initialize(
        @embed_dim : Int32,
        @num_heads : Int32,
        @dropout : Float32 = 0.0_f32,
        bias : Bool = true,
        device : Tensor::Device = Tensor.default_device
      )
        raise ArgumentError.new("embed_dim must be divisible by num_heads") unless @embed_dim % @num_heads == 0

        @head_dim = @embed_dim // @num_heads

        # QKV projections
        @q_proj = Linear.new(@embed_dim, @embed_dim, bias: bias, device: device)
        @k_proj = Linear.new(@embed_dim, @embed_dim, bias: bias, device: device)
        @v_proj = Linear.new(@embed_dim, @embed_dim, bias: bias, device: device)

        # Output projection
        @out_proj = Linear.new(@embed_dim, @embed_dim, bias: bias, device: device)
      end

      # Forward pass
      # query, key, value: [batch, seq_len, embed_dim]
      # attn_mask: optional [batch, seq_len, seq_len] or [seq_len, seq_len]
      # Returns: [batch, seq_len, embed_dim]
      def forward(
        query : Autograd::Variable,
        key : Autograd::Variable,
        value : Autograd::Variable,
        attn_mask : Tensor? = nil,
        need_weights : Bool = false
      ) : Autograd::Variable
        batch_size = query.data.shape[0]
        tgt_len = query.data.shape[1]
        src_len = key.data.shape[1]

        # Project Q, K, V
        q = @q_proj.forward(query)  # [batch, tgt_len, embed_dim]
        k = @k_proj.forward(key)    # [batch, src_len, embed_dim]
        v = @v_proj.forward(value)  # [batch, src_len, embed_dim]

        # Reshape for multi-head: [batch, seq_len, num_heads, head_dim]
        # Then transpose to: [batch, num_heads, seq_len, head_dim]
        q = reshape_for_heads(q, batch_size, tgt_len)
        k = reshape_for_heads(k, batch_size, src_len)
        v = reshape_for_heads(v, batch_size, src_len)

        # Scaled dot-product attention
        # scores = Q @ K^T / sqrt(d_k)
        scale = 1.0_f32 / Math.sqrt(@head_dim.to_f32)
        attn_output = scaled_dot_product_attention(q, k, v, scale, attn_mask)

        # Reshape back: [batch, num_heads, tgt_len, head_dim] -> [batch, tgt_len, embed_dim]
        attn_output = reshape_from_heads(attn_output, batch_size, tgt_len)

        # Output projection
        @out_proj.forward(attn_output)
      end

      def call(
        query : Autograd::Variable,
        key : Autograd::Variable,
        value : Autograd::Variable,
        attn_mask : Tensor? = nil
      ) : Autograd::Variable
        forward(query, key, value, attn_mask)
      end

      # Self-attention convenience method
      def self_attention(x : Autograd::Variable, attn_mask : Tensor? = nil) : Autograd::Variable
        forward(x, x, x, attn_mask)
      end

      # Get all trainable parameters
      def parameters : Array(Autograd::Variable)
        @q_proj.parameters + @k_proj.parameters + @v_proj.parameters + @out_proj.parameters
      end

      # Reshape from [batch, seq_len, embed_dim] to [batch, num_heads, seq_len, head_dim]
      private def reshape_for_heads(x : Autograd::Variable, batch_size : Int32, seq_len : Int32) : Autograd::Variable
        if x.data.on_gpu? && GPUOps.available? && ENV["GS_ATTN_RESHAPE_CPU"]?.nil?
          flat = Tensor.new(batch_size * @num_heads, seq_len, @head_dim, device: Tensor::Device::GPU)
          GPUOps.reshape_for_heads(x.data, flat, batch_size, seq_len, @num_heads, @head_dim)
          reshaped = flat.reshape(batch_size, @num_heads, seq_len, @head_dim)
          result_var = Autograd::Variable.new(reshaped, x.requires_grad?)

          if result_var.requires_grad?
            result_var.is_leaf = false
            batch_cap = batch_size
            seq_len_cap = seq_len
            num_heads_cap = @num_heads
            head_dim_cap = @head_dim
            embed_dim_cap = @embed_dim
            x_on_gpu = x.data.on_gpu?

            grad_fn = Autograd::CustomBackward.new("ReshapeForHeadsBackward", ->(g : Tensor) {
              g_cpu = g.on_cpu? ? g : g.to_cpu
              g_d = g_cpu.cpu_data.not_nil!

              grad_x = Tensor.zeros(batch_cap, seq_len_cap, embed_dim_cap, device: Tensor::Device::CPU)
              gx_d = grad_x.cpu_data.not_nil!

              batch_cap.times do |b|
                seq_len_cap.times do |s|
                  num_heads_cap.times do |h|
                    head_dim_cap.times do |d|
                      src_idx = b * num_heads_cap * seq_len_cap * head_dim_cap + h * seq_len_cap * head_dim_cap + s * head_dim_cap + d
                      dst_idx = b * seq_len_cap * embed_dim_cap + s * embed_dim_cap + h * head_dim_cap + d
                      gx_d[dst_idx] = g_d[src_idx]
                    end
                  end
                end
              end

              [x_on_gpu ? grad_x.to_gpu : grad_x] of Tensor?
            })
            grad_fn.inputs = [x]
            result_var.grad_fn = grad_fn
          end

          return result_var
        end

        x_data = x.data.on_cpu? ? x.data : x.data.to_cpu
        x_d = x_data.cpu_data.not_nil!

        # Target shape: [batch, num_heads, seq_len, head_dim]
        result = Tensor.new(batch_size, @num_heads, seq_len, @head_dim, device: Tensor::Device::CPU)
        r_d = result.cpu_data.not_nil!

        batch_size.times do |b|
          seq_len.times do |s|
            @num_heads.times do |h|
              @head_dim.times do |d|
                src_idx = b * seq_len * @embed_dim + s * @embed_dim + h * @head_dim + d
                dst_idx = b * @num_heads * seq_len * @head_dim + h * seq_len * @head_dim + s * @head_dim + d
                r_d[dst_idx] = x_d[src_idx]
              end
            end
          end
        end

        result = result.to_gpu if x.data.on_gpu?
        result_var = Autograd::Variable.new(result, x.requires_grad?)

        if result_var.requires_grad?
          result_var.is_leaf = false
          batch_cap = batch_size
          seq_len_cap = seq_len
          num_heads_cap = @num_heads
          head_dim_cap = @head_dim
          embed_dim_cap = @embed_dim
          x_on_gpu = x.data.on_gpu?

          grad_fn = Autograd::CustomBackward.new("ReshapeForHeadsBackward", ->(g : Tensor) {
            g_cpu = g.on_cpu? ? g : g.to_cpu
            g_d = g_cpu.cpu_data.not_nil!

            grad_x = Tensor.zeros(batch_cap, seq_len_cap, embed_dim_cap, device: Tensor::Device::CPU)
            gx_d = grad_x.cpu_data.not_nil!

            batch_cap.times do |b|
              seq_len_cap.times do |s|
                num_heads_cap.times do |h|
                  head_dim_cap.times do |d|
                    src_idx = b * num_heads_cap * seq_len_cap * head_dim_cap + h * seq_len_cap * head_dim_cap + s * head_dim_cap + d
                    dst_idx = b * seq_len_cap * embed_dim_cap + s * embed_dim_cap + h * head_dim_cap + d
                    gx_d[dst_idx] = g_d[src_idx]
                  end
                end
              end
            end

            [x_on_gpu ? grad_x.to_gpu : grad_x] of Tensor?
          })
          grad_fn.inputs = [x]
          result_var.grad_fn = grad_fn
        end

        result_var
      end

      # Reshape from [batch, num_heads, seq_len, head_dim] to [batch, seq_len, embed_dim]
      private def reshape_from_heads(x : Autograd::Variable, batch_size : Int32, seq_len : Int32) : Autograd::Variable
        if x.data.on_gpu? && GPUOps.available? && ENV["GS_ATTN_RESHAPE_CPU"]?.nil?
          flat = x.data.reshape(batch_size * @num_heads, seq_len, @head_dim)
          result = Tensor.new(batch_size, seq_len, @embed_dim, device: Tensor::Device::GPU)
          GPUOps.reshape_from_heads(flat, result, batch_size, seq_len, @num_heads, @head_dim)
          result_var = Autograd::Variable.new(result, x.requires_grad?)

          if result_var.requires_grad?
            result_var.is_leaf = false
            batch_cap = batch_size
            seq_len_cap = seq_len
            num_heads_cap = @num_heads
            head_dim_cap = @head_dim
            embed_dim_cap = @embed_dim
            x_on_gpu = x.data.on_gpu?

            grad_fn = Autograd::CustomBackward.new("ReshapeFromHeadsBackward", ->(g : Tensor) {
              g_cpu = g.on_cpu? ? g : g.to_cpu
              g_d = g_cpu.cpu_data.not_nil!

              grad_x = Tensor.zeros(batch_cap, num_heads_cap, seq_len_cap, head_dim_cap, device: Tensor::Device::CPU)
              gx_d = grad_x.cpu_data.not_nil!

              batch_cap.times do |b|
                seq_len_cap.times do |s|
                  num_heads_cap.times do |h|
                    head_dim_cap.times do |d|
                      dst_idx = b * seq_len_cap * embed_dim_cap + s * embed_dim_cap + h * head_dim_cap + d
                      src_idx = b * num_heads_cap * seq_len_cap * head_dim_cap + h * seq_len_cap * head_dim_cap + s * head_dim_cap + d
                      gx_d[src_idx] = g_d[dst_idx]
                    end
                  end
                end
              end

              [x_on_gpu ? grad_x.to_gpu : grad_x] of Tensor?
            })
            grad_fn.inputs = [x]
            result_var.grad_fn = grad_fn
          end

          return result_var
        end

        x_data = x.data.on_cpu? ? x.data : x.data.to_cpu
        x_d = x_data.cpu_data.not_nil!

        # Target shape: [batch, seq_len, embed_dim]
        result = Tensor.new(batch_size, seq_len, @embed_dim, device: Tensor::Device::CPU)
        r_d = result.cpu_data.not_nil!

        batch_size.times do |b|
          seq_len.times do |s|
            @num_heads.times do |h|
              @head_dim.times do |d|
                src_idx = b * @num_heads * seq_len * @head_dim + h * seq_len * @head_dim + s * @head_dim + d
                dst_idx = b * seq_len * @embed_dim + s * @embed_dim + h * @head_dim + d
                r_d[dst_idx] = x_d[src_idx]
              end
            end
          end
        end

        result = result.to_gpu if x.data.on_gpu?
        result_var = Autograd::Variable.new(result, x.requires_grad?)

        if result_var.requires_grad?
          result_var.is_leaf = false
          batch_cap = batch_size
          seq_len_cap = seq_len
          num_heads_cap = @num_heads
          head_dim_cap = @head_dim
          embed_dim_cap = @embed_dim
          x_on_gpu = x.data.on_gpu?

          grad_fn = Autograd::CustomBackward.new("ReshapeFromHeadsBackward", ->(g : Tensor) {
            g_cpu = g.on_cpu? ? g : g.to_cpu
            g_d = g_cpu.cpu_data.not_nil!

            grad_x = Tensor.zeros(batch_cap, num_heads_cap, seq_len_cap, head_dim_cap, device: Tensor::Device::CPU)
            gx_d = grad_x.cpu_data.not_nil!

            batch_cap.times do |b|
              seq_len_cap.times do |s|
                num_heads_cap.times do |h|
                  head_dim_cap.times do |d|
                    dst_idx = b * seq_len_cap * embed_dim_cap + s * embed_dim_cap + h * head_dim_cap + d
                    src_idx = b * num_heads_cap * seq_len_cap * head_dim_cap + h * seq_len_cap * head_dim_cap + s * head_dim_cap + d
                    gx_d[src_idx] = g_d[dst_idx]
                  end
                end
              end
            end

            [x_on_gpu ? grad_x.to_gpu : grad_x] of Tensor?
          })
          grad_fn.inputs = [x]
          result_var.grad_fn = grad_fn
        end

        result_var
      end

      # Scaled dot-product attention
      # Q, K, V: [batch, num_heads, seq_len, head_dim]
      private def scaled_dot_product_attention(
        q : Autograd::Variable,
        k : Autograd::Variable,
        v : Autograd::Variable,
        scale : Float32,
        mask : Tensor?
      ) : Autograd::Variable
        needs_grad = q.requires_grad? || k.requires_grad? || v.requires_grad?

        # Try GPU path if tensors are on GPU and mask is not provided
        # (GPU kernel doesn't support masking yet)
        if !needs_grad && q.data.on_gpu? && k.data.on_gpu? && v.data.on_gpu? && mask.nil? && GPUOps.available?
          return scaled_dot_product_attention_gpu(q, k, v, scale)
        end

        # CPU path (with autograd if needed)
        scaled_dot_product_attention_cpu(q, k, v, scale, mask, needs_grad)
      end

      # GPU implementation using fused attention kernel
      private def scaled_dot_product_attention_gpu(
        q : Autograd::Variable,
        k : Autograd::Variable,
        v : Autograd::Variable,
        scale : Float32
      ) : Autograd::Variable
        batch = q.data.shape[0]
        heads = q.data.shape[1]
        tgt_len = q.data.shape[2]
        head_dim = q.data.shape[3]

        # Reshape from [batch, heads, seq, head_dim] to [batch*heads, seq, head_dim]
        # for the fused attention kernel
        batch_heads = batch * heads

        # Create reshaped views (contiguous in memory)
        q_reshaped = Tensor.new(batch_heads, tgt_len, head_dim, device: Tensor::Device::GPU)
        k_reshaped = Tensor.new(batch_heads, tgt_len, head_dim, device: Tensor::Device::GPU)
        v_reshaped = Tensor.new(batch_heads, tgt_len, head_dim, device: Tensor::Device::GPU)
        output_reshaped = Tensor.new(batch_heads, tgt_len, head_dim, device: Tensor::Device::GPU)

        # Copy data (reshape is just a view change for contiguous 4D -> 3D)
        # Since [batch, heads, seq, head_dim] is contiguous and we're flattening first two dims,
        # we can copy directly
        q_buf = q.data.buffer.not_nil!
        k_buf = k.data.buffer.not_nil!
        v_buf = v.data.buffer.not_nil!

        q_reshaped.buffer.not_nil!.copy_from(q_buf, q.data.numel.to_i64 * 4_i64)
        k_reshaped.buffer.not_nil!.copy_from(k_buf, k.data.numel.to_i64 * 4_i64)
        v_reshaped.buffer.not_nil!.copy_from(v_buf, v.data.numel.to_i64 * 4_i64)

        # Run fused attention kernel
        GPUOps.fused_attention(q_reshaped, k_reshaped, v_reshaped, output_reshaped, scale)

        # Reshape output back to [batch, heads, seq, head_dim]
        output = Tensor.new(batch, heads, tgt_len, head_dim, device: Tensor::Device::GPU)
        output.buffer.not_nil!.copy_from(output_reshaped.buffer.not_nil!, output.numel.to_i64 * 4_i64)

        Autograd::Variable.new(output, q.requires_grad? || k.requires_grad? || v.requires_grad?)
      end

      # CPU implementation (original code)
      private def scaled_dot_product_attention_cpu(
        q : Autograd::Variable,
        k : Autograd::Variable,
        v : Autograd::Variable,
        scale : Float32,
        mask : Tensor?,
        needs_grad : Bool
      ) : Autograd::Variable
        q_data = q.data.on_cpu? ? q.data : q.data.to_cpu
        k_data = k.data.on_cpu? ? k.data : k.data.to_cpu
        v_data = v.data.on_cpu? ? v.data : v.data.to_cpu

        batch = q_data.shape[0]
        heads = q_data.shape[1]
        tgt_len = q_data.shape[2]
        src_len = k_data.shape[2]
        head_dim = q_data.shape[3]

        q_d = q_data.cpu_data.not_nil!
        k_d = k_data.cpu_data.not_nil!
        v_d = v_data.cpu_data.not_nil!

        # Compute attention scores: [batch, heads, tgt_len, src_len]
        scores = Tensor.new(batch, heads, tgt_len, src_len, device: Tensor::Device::CPU)
        s_d = scores.cpu_data.not_nil!

        batch.times do |b|
          heads.times do |h|
            tgt_len.times do |i|
              src_len.times do |j|
                # dot product of q[b,h,i,:] and k[b,h,j,:]
                dot = 0.0_f32
                head_dim.times do |d|
                  q_idx = b * heads * tgt_len * head_dim + h * tgt_len * head_dim + i * head_dim + d
                  k_idx = b * heads * src_len * head_dim + h * src_len * head_dim + j * head_dim + d
                  dot += q_d[q_idx] * k_d[k_idx]
                end
                s_idx = b * heads * tgt_len * src_len + h * tgt_len * src_len + i * src_len + j
                s_d[s_idx] = dot * scale
              end
            end
          end
        end

        # Apply mask if provided (additive mask, -inf for masked positions)
        if m = mask
          m_cpu = m.on_cpu? ? m : m.to_cpu
          m_d = m_cpu.cpu_data.not_nil!

          if m.ndim == 2
            # [tgt_len, src_len] - broadcast to all batch/heads
            batch.times do |b|
              heads.times do |h|
                tgt_len.times do |i|
                  src_len.times do |j|
                    s_idx = b * heads * tgt_len * src_len + h * tgt_len * src_len + i * src_len + j
                    m_idx = i * src_len + j
                    s_d[s_idx] += m_d[m_idx]
                  end
                end
              end
            end
          end
        end

        # Softmax over last dimension (src_len)
        batch.times do |b|
          heads.times do |h|
            tgt_len.times do |i|
              offset = b * heads * tgt_len * src_len + h * tgt_len * src_len + i * src_len

              # Find max for numerical stability
              max_val = s_d[offset]
              (1...src_len).each { |j| max_val = Math.max(max_val, s_d[offset + j]) }

              # Exp and sum
              sum = 0.0_f32
              src_len.times do |j|
                s_d[offset + j] = Math.exp(s_d[offset + j] - max_val)
                sum += s_d[offset + j]
              end

              # Normalize
              src_len.times { |j| s_d[offset + j] /= sum }
            end
          end
        end

        # Output: scores @ V -> [batch, heads, tgt_len, head_dim]
        output = Tensor.new(batch, heads, tgt_len, head_dim, device: Tensor::Device::CPU)
        o_d = output.cpu_data.not_nil!

        batch.times do |b|
          heads.times do |h|
            tgt_len.times do |i|
              head_dim.times do |d|
                sum = 0.0_f32
                src_len.times do |j|
                  s_idx = b * heads * tgt_len * src_len + h * tgt_len * src_len + i * src_len + j
                  v_idx = b * heads * src_len * head_dim + h * src_len * head_dim + j * head_dim + d
                  sum += s_d[s_idx] * v_d[v_idx]
                end
                o_idx = b * heads * tgt_len * head_dim + h * tgt_len * head_dim + i * head_dim + d
                o_d[o_idx] = sum
              end
            end
          end
        end

        output = output.to_gpu if q.data.on_gpu?
        result_var = Autograd::Variable.new(output, needs_grad)

        if needs_grad
          result_var.is_leaf = false

          q_cpu = q_data
          k_cpu = k_data
          v_cpu = v_data
          s_cpu = scores

          q_on_gpu = q.data.on_gpu?
          k_on_gpu = k.data.on_gpu?
          v_on_gpu = v.data.on_gpu?

          scale_cap = scale

          grad_fn = Autograd::CustomBackward.new("AttentionBackward", ->(g : Tensor) {
            g_cpu = g.on_cpu? ? g : g.to_cpu

            g_d = g_cpu.cpu_data.not_nil!
            q_d = q_cpu.cpu_data.not_nil!
            k_d = k_cpu.cpu_data.not_nil!
            v_d = v_cpu.cpu_data.not_nil!
            s_d_bw = s_cpu.cpu_data.not_nil!

            grad_q = Tensor.new(q_cpu.shape, q_cpu.dtype, Tensor::Device::CPU)
            grad_k = Tensor.new(k_cpu.shape, k_cpu.dtype, Tensor::Device::CPU)
            grad_v = Tensor.new(v_cpu.shape, v_cpu.dtype, Tensor::Device::CPU)

            gq_d = grad_q.cpu_data.not_nil!
            gk_d = grad_k.cpu_data.not_nil!
            gv_d = grad_v.cpu_data.not_nil!

            batch.times do |b|
              heads.times do |h|
                q_base = (b * heads + h) * tgt_len * head_dim
                k_base = (b * heads + h) * src_len * head_dim
                v_base = (b * heads + h) * src_len * head_dim
                s_base = (b * heads + h) * tgt_len * src_len

                tgt_len.times do |i|
                  g_offset = q_base + i * head_dim
                  s_row_base = s_base + i * src_len

                  dS = Array(Float32).new(src_len, 0.0_f32)

                  # dS = dO @ V^T
                  src_len.times do |j|
                    sum = 0.0_f32
                    v_offset = v_base + j * head_dim
                    head_dim.times do |d|
                      sum += g_d[g_offset + d] * v_d[v_offset + d]
                    end
                    dS[j] = sum
                  end

                  # softmax backward: dScores = (dS - sum(dS * S)) * S
                  dot = 0.0_f32
                  src_len.times do |j|
                    dot += dS[j] * s_d_bw[s_row_base + j]
                  end

                  src_len.times do |j|
                    ds = (dS[j] - dot) * s_d_bw[s_row_base + j]
                    k_offset = k_base + j * head_dim

                    head_dim.times do |d|
                      gq_d[g_offset + d] += ds * k_d[k_offset + d] * scale_cap
                      gk_d[k_offset + d] += ds * q_d[g_offset + d] * scale_cap
                    end
                  end

                  # dV = S^T @ dO
                  src_len.times do |j|
                    s_val = s_d_bw[s_row_base + j]
                    v_offset = v_base + j * head_dim
                    head_dim.times do |d|
                      gv_d[v_offset + d] += s_val * g_d[g_offset + d]
                    end
                  end
                end
              end
            end

            [
              q_on_gpu ? grad_q.to_gpu : grad_q,
              k_on_gpu ? grad_k.to_gpu : grad_k,
              v_on_gpu ? grad_v.to_gpu : grad_v,
            ] of Tensor?
          })

          grad_fn.inputs = [q, k, v]
          result_var.grad_fn = grad_fn
        end

        result_var
      end
    end

    # Cross-attention (for decoder attending to encoder)
    class CrossAttention < MultiHeadAttention
      # Same as MultiHeadAttention, just a semantic alias
      def forward_cross(
        query : Autograd::Variable,   # From decoder
        memory : Autograd::Variable,  # From encoder
        attn_mask : Tensor? = nil
      ) : Autograd::Variable
        forward(query, memory, memory, attn_mask)
      end
    end
  end
end
