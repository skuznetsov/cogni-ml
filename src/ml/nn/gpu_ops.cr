# GPU operations for neural network layers
# Uses Metal kernels from nn.metal for acceleration

require "../metal/device"
require "../metal/dispatch"
require "../core/tensor"

module ML
  module NN
    # Load nn.metal kernel source at compile time
    NN_KERNEL_SOURCE = {{ read_file("#{__DIR__}/../metal/kernels/nn.metal") }}

    # GPU operations module
    module GPUOps
      extend self

      TILE_SIZE = 16

      @@pipelines = Hash(String, Metal::ComputePipeline).new
      @@initialized = false

      # Initialize GPU pipelines (lazy)
      def ensure_initialized
        return if @@initialized
        return unless Metal::Device.available?
        @@initialized = true
      end

      # Get or create pipeline for a kernel
      def get_pipeline(name : String) : Metal::ComputePipeline
        @@pipelines[name] ||= Metal::ComputePipeline.new(name, NN_KERNEL_SOURCE, name)
      end

      # Linear forward: output = input @ weight^T + bias
      # input: [batch, in_features]
      # weight: [out_features, in_features]
      # bias: [out_features] or nil
      # output: [batch, out_features]
      def linear_forward(
        input : Tensor,
        weight : Tensor,
        bias : Tensor?,
        output : Tensor
      ) : Nil
        ensure_initialized

        batch = input.shape[0]
        in_features = input.shape[1]
        out_features = weight.shape[0]
        use_bias = bias ? 1_u32 : 0_u32

        # Use simple kernel for small matrices, tiled for large
        kernel_name = (in_features >= 64 && out_features >= 64) ? "linear_forward_tiled" : "linear_forward_simple"
        pipeline = get_pipeline(kernel_name)

        # Create dummy bias buffer if not provided
        bias_buf = bias.try(&.buffer) || input.buffer.not_nil!

        Metal::Dispatch.execute(pipeline) do |encoder|
          encoder.set_buffer(input.buffer.not_nil!, 0)
          encoder.set_buffer(weight.buffer.not_nil!, 1)
          encoder.set_buffer(bias_buf, 2)
          encoder.set_buffer(output.buffer.not_nil!, 3)
          encoder.set_value(batch.to_u32, 4)
          encoder.set_value(in_features.to_u32, 5)
          encoder.set_value(out_features.to_u32, 6)
          encoder.set_value(use_bias, 7)

          if kernel_name == "linear_forward_tiled"
            # Set threadgroup memory for tiles
            tile_mem_size = TILE_SIZE * TILE_SIZE * sizeof(Float32)
            encoder.set_threadgroup_memory(tile_mem_size, 0)
            encoder.set_threadgroup_memory(tile_mem_size, 1)
          end

          encoder.dispatch_2d(out_features, batch, {TILE_SIZE, TILE_SIZE})
        end
      end

      # Linear forward with fused GELU activation
      def linear_gelu_forward(
        input : Tensor,
        weight : Tensor,
        bias : Tensor?,
        output : Tensor
      ) : Nil
        ensure_initialized

        batch = input.shape[0]
        in_features = input.shape[1]
        out_features = weight.shape[0]
        use_bias = bias ? 1_u32 : 0_u32

        pipeline = get_pipeline("linear_gelu_forward")
        bias_buf = bias.try(&.buffer) || input.buffer.not_nil!

        Metal::Dispatch.execute(pipeline) do |encoder|
          encoder.set_buffer(input.buffer.not_nil!, 0)
          encoder.set_buffer(weight.buffer.not_nil!, 1)
          encoder.set_buffer(bias_buf, 2)
          encoder.set_buffer(output.buffer.not_nil!, 3)
          encoder.set_value(batch.to_u32, 4)
          encoder.set_value(in_features.to_u32, 5)
          encoder.set_value(out_features.to_u32, 6)
          encoder.set_value(use_bias, 7)
          encoder.dispatch_2d(out_features, batch, {TILE_SIZE, TILE_SIZE})
        end
      end

      # LayerNorm forward
      # input: [batch, features]
      # gamma, beta: [features]
      # output: [batch, features]
      def layernorm_forward(
        input : Tensor,
        gamma : Tensor,
        beta : Tensor,
        output : Tensor,
        eps : Float32 = 1e-5_f32
      ) : Nil
        ensure_initialized

        batch = input.shape[0]
        features = input.shape[input.shape.ndim - 1]

        pipeline = get_pipeline("layernorm_forward")

        Metal::Dispatch.execute(pipeline) do |encoder|
          encoder.set_buffer(input.buffer.not_nil!, 0)
          encoder.set_buffer(gamma.buffer.not_nil!, 1)
          encoder.set_buffer(beta.buffer.not_nil!, 2)
          encoder.set_buffer(output.buffer.not_nil!, 3)
          encoder.set_value(batch.to_u32, 4)
          encoder.set_value(features.to_u32, 5)
          encoder.set_value(eps, 6)
          encoder.dispatch_1d(batch)
        end
      end

      # Elementwise add: out = a + b
      def add(a : Tensor, b : Tensor, output : Tensor) : Nil
        ensure_initialized

        count = a.numel
        raise "Add size mismatch" if b.numel != count || output.numel != count

        pipeline = get_pipeline("add_tensors")

        Metal::Dispatch.execute(pipeline) do |encoder|
          encoder.set_buffer(a.buffer.not_nil!, 0)
          encoder.set_buffer(b.buffer.not_nil!, 1)
          encoder.set_buffer(output.buffer.not_nil!, 2)
          encoder.set_value(count.to_u32, 3)
          encoder.dispatch_1d(count, 256)
        end
      end

      # RMSNorm forward
      def rmsnorm_forward(
        input : Tensor,
        gamma : Tensor,
        output : Tensor,
        eps : Float32 = 1e-5_f32
      ) : Nil
        ensure_initialized

        batch = input.shape[0]
        features = input.shape[input.shape.ndim - 1]

        pipeline = get_pipeline("rmsnorm_forward")

        Metal::Dispatch.execute(pipeline) do |encoder|
          encoder.set_buffer(input.buffer.not_nil!, 0)
          encoder.set_buffer(gamma.buffer.not_nil!, 1)
          encoder.set_buffer(output.buffer.not_nil!, 2)
          encoder.set_value(batch.to_u32, 3)
          encoder.set_value(features.to_u32, 4)
          encoder.set_value(eps, 5)
          encoder.dispatch_1d(batch)
        end
      end

      # Add bias: output = input + bias (broadcast over batch)
      def add_bias(
        input : Tensor,
        bias : Tensor,
        output : Tensor
      ) : Nil
        ensure_initialized

        batch = input.shape[0]
        features = input.shape[input.shape.ndim - 1]

        pipeline = get_pipeline("add_bias")

        Metal::Dispatch.execute(pipeline) do |encoder|
          encoder.set_buffer(input.buffer.not_nil!, 0)
          encoder.set_buffer(bias.buffer.not_nil!, 1)
          encoder.set_buffer(output.buffer.not_nil!, 2)
          encoder.set_value(batch.to_u32, 3)
          encoder.set_value(features.to_u32, 4)
          encoder.dispatch_2d(features, batch)
        end
      end

      # Check if GPU is available for NN ops
      def available? : Bool
        Metal::Device.available?
      end

      # ============================================================================
      # Fused Attention Operations
      # ============================================================================

      # Fused scaled dot-product attention: output = softmax(Q @ K^T / sqrt(d)) @ V
      # Q, K, V: [batch_heads, seq_len, head_dim]
      # Output: [batch_heads, seq_len, head_dim]
      def fused_attention(
        q : Tensor,
        k : Tensor,
        v : Tensor,
        output : Tensor,
        scale : Float32
      ) : Nil
        ensure_initialized

        batch_heads = q.shape[0]
        seq_len = q.shape[1]
        head_dim = q.shape[2]

        # Use flash_attention for memory efficiency
        pipeline = get_pipeline("flash_attention")

        Metal::Dispatch.execute(pipeline) do |encoder|
          encoder.set_buffer(q.buffer.not_nil!, 0)
          encoder.set_buffer(k.buffer.not_nil!, 1)
          encoder.set_buffer(v.buffer.not_nil!, 2)
          encoder.set_buffer(output.buffer.not_nil!, 3)
          encoder.set_value(batch_heads.to_u32, 4)
          encoder.set_value(seq_len.to_u32, 5)
          encoder.set_value(head_dim.to_u32, 6)
          encoder.set_value(scale, 7)
          encoder.dispatch_2d(head_dim, batch_heads * seq_len, {TILE_SIZE, TILE_SIZE})
        end
      end

      # Fused attention with tiled optimization (for larger sequences)
      def fused_attention_tiled(
        q : Tensor,
        k : Tensor,
        v : Tensor,
        output : Tensor,
        scale : Float32
      ) : Nil
        ensure_initialized

        batch_heads = q.shape[0]
        seq_len = q.shape[1]
        head_dim = q.shape[2]

        pipeline = get_pipeline("fused_attention_tiled")

        # Threadgroup memory for scores and partial V accumulation
        scores_mem_size = seq_len * sizeof(Float32)
        v_mem_size = head_dim * sizeof(Float32)

        Metal::Dispatch.execute(pipeline) do |encoder|
          encoder.set_buffer(q.buffer.not_nil!, 0)
          encoder.set_buffer(k.buffer.not_nil!, 1)
          encoder.set_buffer(v.buffer.not_nil!, 2)
          encoder.set_buffer(output.buffer.not_nil!, 3)
          encoder.set_value(batch_heads.to_u32, 4)
          encoder.set_value(seq_len.to_u32, 5)
          encoder.set_value(head_dim.to_u32, 6)
          encoder.set_value(scale, 7)
          encoder.set_threadgroup_memory(scores_mem_size, 0)
          encoder.set_threadgroup_memory(v_mem_size, 1)
          # One threadgroup per query position
          encoder.dispatch_1d(batch_heads * seq_len)
        end
      end

      # Softmax over last dimension
      # input/output: [rows, cols]
      def softmax(
        input : Tensor,
        output : Tensor
      ) : Nil
        ensure_initialized

        rows = input.shape[0]
        cols = input.shape[input.shape.ndim - 1]

        pipeline = get_pipeline("softmax_forward")

        Metal::Dispatch.execute(pipeline) do |encoder|
          encoder.set_buffer(input.buffer.not_nil!, 0)
          encoder.set_buffer(output.buffer.not_nil!, 1)
          encoder.set_value(rows.to_u32, 2)
          encoder.set_value(cols.to_u32, 3)
          encoder.dispatch_1d(rows)
        end
      end

      # Batched matrix multiply: C[b] = A[b] @ B[b]
      # A: [batch, M, K], B: [batch, K, N], C: [batch, M, N]
      def batched_matmul(
        a : Tensor,
        b : Tensor,
        c : Tensor
      ) : Nil
        ensure_initialized

        batch = a.shape[0]
        m = a.shape[1]
        k = a.shape[2]
        n = b.shape[2]

        pipeline = get_pipeline("batched_matmul")
        tile_mem_size = TILE_SIZE * TILE_SIZE * sizeof(Float32)

        Metal::Dispatch.execute(pipeline) do |encoder|
          encoder.set_buffer(a.buffer.not_nil!, 0)
          encoder.set_buffer(b.buffer.not_nil!, 1)
          encoder.set_buffer(c.buffer.not_nil!, 2)
          encoder.set_value(batch.to_u32, 3)
          encoder.set_value(m.to_u32, 4)
          encoder.set_value(n.to_u32, 5)
          encoder.set_value(k.to_u32, 6)
          encoder.set_threadgroup_memory(tile_mem_size, 0)
          encoder.set_threadgroup_memory(tile_mem_size, 1)
          encoder.dispatch_3d(n, m, batch, {TILE_SIZE, TILE_SIZE, 1})
        end
      end

      # Batched matmul with transpose: C[b] = A[b] @ B[b]^T
      # A: [batch, M, K], B: [batch, N, K], C: [batch, M, N]
      def batched_matmul_tn(
        a : Tensor,
        b : Tensor,
        c : Tensor,
        scale : Float32 = 1.0_f32
      ) : Nil
        ensure_initialized

        batch = a.shape[0]
        m = a.shape[1]
        k = a.shape[2]
        n = b.shape[1]  # B is [batch, N, K], so N is shape[1]

        pipeline = get_pipeline("batched_matmul_tn")

        Metal::Dispatch.execute(pipeline) do |encoder|
          encoder.set_buffer(a.buffer.not_nil!, 0)
          encoder.set_buffer(b.buffer.not_nil!, 1)
          encoder.set_buffer(c.buffer.not_nil!, 2)
          encoder.set_value(batch.to_u32, 3)
          encoder.set_value(m.to_u32, 4)
          encoder.set_value(n.to_u32, 5)
          encoder.set_value(k.to_u32, 6)
          encoder.set_value(scale, 7)
          encoder.dispatch_3d(n, m, batch, {TILE_SIZE, TILE_SIZE, 1})
        end
      end

      # Reshape [batch, seq, embed] -> [batch * heads, seq, head_dim]
      def reshape_for_heads(
        input : Tensor,
        output : Tensor,
        batch : Int32,
        seq_len : Int32,
        num_heads : Int32,
        head_dim : Int32
      ) : Nil
        ensure_initialized

        pipeline = get_pipeline("reshape_for_heads")

        Metal::Dispatch.execute(pipeline) do |encoder|
          encoder.set_buffer(input.buffer.not_nil!, 0)
          encoder.set_buffer(output.buffer.not_nil!, 1)
          encoder.set_value(batch.to_u32, 2)
          encoder.set_value(seq_len.to_u32, 3)
          encoder.set_value(num_heads.to_u32, 4)
          encoder.set_value(head_dim.to_u32, 5)
          encoder.dispatch_3d(head_dim, seq_len, batch * num_heads, {TILE_SIZE, TILE_SIZE, 1})
        end
      end

      # Reshape [batch * heads, seq, head_dim] -> [batch, seq, embed]
      def reshape_from_heads(
        input : Tensor,
        output : Tensor,
        batch : Int32,
        seq_len : Int32,
        num_heads : Int32,
        head_dim : Int32
      ) : Nil
        ensure_initialized

        embed_dim = num_heads * head_dim
        pipeline = get_pipeline("reshape_from_heads")

        Metal::Dispatch.execute(pipeline) do |encoder|
          encoder.set_buffer(input.buffer.not_nil!, 0)
          encoder.set_buffer(output.buffer.not_nil!, 1)
          encoder.set_value(batch.to_u32, 2)
          encoder.set_value(seq_len.to_u32, 3)
          encoder.set_value(num_heads.to_u32, 4)
          encoder.set_value(head_dim.to_u32, 5)
          encoder.dispatch_3d(embed_dim, seq_len, batch, {TILE_SIZE, TILE_SIZE, 1})
        end
      end

      # Convert NCHW -> NHWC
      def nchw_to_nhwc(input : Tensor, output : Tensor) : Nil
        ensure_initialized

        batch = input.shape[0]
        channels = input.shape[1]
        height = input.shape[2]
        width = input.shape[3]

        pipeline = get_pipeline("nchw_to_nhwc")

        Metal::Dispatch.execute(pipeline) do |encoder|
          encoder.set_buffer(input.buffer.not_nil!, 0)
          encoder.set_buffer(output.buffer.not_nil!, 1)
          encoder.set_value(batch.to_u32, 2)
          encoder.set_value(channels.to_u32, 3)
          encoder.set_value(height.to_u32, 4)
          encoder.set_value(width.to_u32, 5)
          encoder.dispatch_3d(width, height, batch * channels, {8, 8, 4})
        end
      end

      # Apply RoPE to [batch, seq, embed]
      def rope_apply(
        input : Tensor,
        freqs : Tensor,
        output : Tensor,
        batch : Int32,
        seq_len : Int32,
        embed_dim : Int32,
        num_heads : Int32,
        head_dim : Int32,
        width : Int32
      ) : Nil
        ensure_initialized

        pipeline = get_pipeline("rope_apply")
        freqs_len = freqs.numel

        Metal::Dispatch.execute(pipeline) do |encoder|
          encoder.set_buffer(input.buffer.not_nil!, 0)
          encoder.set_buffer(freqs.buffer.not_nil!, 1)
          encoder.set_buffer(output.buffer.not_nil!, 2)
          encoder.set_value(batch.to_u32, 3)
          encoder.set_value(seq_len.to_u32, 4)
          encoder.set_value(embed_dim.to_u32, 5)
          encoder.set_value(num_heads.to_u32, 6)
          encoder.set_value(head_dim.to_u32, 7)
          encoder.set_value(width.to_u32, 8)
          encoder.set_value(freqs_len.to_u32, 9)
          encoder.dispatch_3d(embed_dim, seq_len, batch, {TILE_SIZE, TILE_SIZE, 1})
        end
      end

      # ============================================================================
      # Fused LayerNorm + Linear
      # ============================================================================

      # Fused LayerNorm + Linear: output = Linear(LayerNorm(input))
      def fused_layernorm_linear(
        input : Tensor,
        ln_gamma : Tensor,
        ln_beta : Tensor,
        weight : Tensor,
        bias : Tensor?,
        output : Tensor,
        eps : Float32 = 1e-5_f32
      ) : Nil
        ensure_initialized

        batch = input.shape[0]
        in_features = input.shape[input.shape.ndim - 1]
        out_features = weight.shape[0]
        use_bias = bias ? 1_u32 : 0_u32

        pipeline = get_pipeline("fused_layernorm_linear")
        bias_buf = bias.try(&.buffer) || input.buffer.not_nil!

        Metal::Dispatch.execute(pipeline) do |encoder|
          encoder.set_buffer(input.buffer.not_nil!, 0)
          encoder.set_buffer(ln_gamma.buffer.not_nil!, 1)
          encoder.set_buffer(ln_beta.buffer.not_nil!, 2)
          encoder.set_buffer(weight.buffer.not_nil!, 3)
          encoder.set_buffer(bias_buf, 4)
          encoder.set_buffer(output.buffer.not_nil!, 5)
          encoder.set_value(batch.to_u32, 6)
          encoder.set_value(in_features.to_u32, 7)
          encoder.set_value(out_features.to_u32, 8)
          encoder.set_value(eps, 9)
          encoder.set_value(use_bias, 10)
          encoder.dispatch_2d(out_features, batch, {TILE_SIZE, TILE_SIZE})
        end
      end

      # Fused LayerNorm + Linear + GELU
      def fused_layernorm_linear_gelu(
        input : Tensor,
        ln_gamma : Tensor,
        ln_beta : Tensor,
        weight : Tensor,
        bias : Tensor?,
        output : Tensor,
        eps : Float32 = 1e-5_f32
      ) : Nil
        ensure_initialized

        batch = input.shape[0]
        in_features = input.shape[input.shape.ndim - 1]
        out_features = weight.shape[0]
        use_bias = bias ? 1_u32 : 0_u32

        pipeline = get_pipeline("fused_layernorm_linear_gelu")
        bias_buf = bias.try(&.buffer) || input.buffer.not_nil!

        Metal::Dispatch.execute(pipeline) do |encoder|
          encoder.set_buffer(input.buffer.not_nil!, 0)
          encoder.set_buffer(ln_gamma.buffer.not_nil!, 1)
          encoder.set_buffer(ln_beta.buffer.not_nil!, 2)
          encoder.set_buffer(weight.buffer.not_nil!, 3)
          encoder.set_buffer(bias_buf, 4)
          encoder.set_buffer(output.buffer.not_nil!, 5)
          encoder.set_value(batch.to_u32, 6)
          encoder.set_value(in_features.to_u32, 7)
          encoder.set_value(out_features.to_u32, 8)
          encoder.set_value(eps, 9)
          encoder.set_value(use_bias, 10)
          encoder.dispatch_2d(out_features, batch, {TILE_SIZE, TILE_SIZE})
        end
      end

      # Conv2D forward: output = conv2d(input, weight) + bias
      # input: [batch, H, W, in_channels] (NHWC format)
      # weight: [out_channels, in_channels, kH, kW] (OIHW format)
      # bias: [out_channels] or nil
      # output: [batch, H_out, W_out, out_channels]
      def conv2d_forward(
        input : Tensor,
        weight : Tensor,
        bias : Tensor?,
        output : Tensor,
        stride : Int32 = 1,
        padding : Int32 = 1,
        fuse_relu : Bool = false
      ) : Nil
        ensure_initialized

        batch = input.shape[0]
        in_h = input.shape[1]
        in_w = input.shape[2]
        in_channels = input.shape[3]

        out_channels = weight.shape[0]
        kernel_h = weight.shape[2]
        kernel_w = weight.shape[3]

        out_h = (in_h + 2 * padding - kernel_h) // stride + 1
        out_w = (in_w + 2 * padding - kernel_w) // stride + 1

        use_bias = bias ? 1_u32 : 0_u32
        kernel_name = fuse_relu ? "conv2d_forward_relu" : "conv2d_forward"
        pipeline = get_pipeline(kernel_name)

        # Create dummy bias buffer if not provided
        bias_buf = bias.try(&.buffer) || input.buffer.not_nil!

        Metal::Dispatch.execute(pipeline) do |encoder|
          encoder.set_buffer(input.buffer.not_nil!, 0)
          encoder.set_buffer(weight.buffer.not_nil!, 1)
          encoder.set_buffer(bias_buf, 2)
          encoder.set_buffer(output.buffer.not_nil!, 3)
          encoder.set_value(batch.to_u32, 4)
          encoder.set_value(in_h.to_u32, 5)
          encoder.set_value(in_w.to_u32, 6)
          encoder.set_value(in_channels.to_u32, 7)
          encoder.set_value(out_channels.to_u32, 8)
          encoder.set_value(kernel_h.to_u32, 9)
          encoder.set_value(kernel_w.to_u32, 10)
          encoder.set_value(stride.to_u32, 11)
          encoder.set_value(padding.to_u32, 12)
          encoder.set_value(use_bias, 13)

          # Dispatch: (out_w, out_h, batch * out_channels)
          encoder.dispatch_3d(out_w, out_h, batch * out_channels, {8, 8, 4})
        end
      end

      # ============================================================
      # Upsampling Operations
      # ============================================================

      # Nearest neighbor 2x upsampling (GPU)
      # input: [batch, h, w, channels] NHWC
      # output: [batch, h*2, w*2, channels]
      def upsample_nearest_2x(input : Tensor, output : Tensor)
        ensure_initialized

        batch = input.shape[0]
        in_h = input.shape[1]
        in_w = input.shape[2]
        channels = input.shape[3]
        out_h = in_h * 2
        out_w = in_w * 2

        pipeline = get_pipeline("upsample_nearest_2x")

        Metal::Dispatch.execute(pipeline) do |encoder|
          encoder.set_buffer(input.buffer.not_nil!, 0)
          encoder.set_buffer(output.buffer.not_nil!, 1)
          encoder.set_value(batch.to_u32, 2)
          encoder.set_value(in_h.to_u32, 3)
          encoder.set_value(in_w.to_u32, 4)
          encoder.set_value(channels.to_u32, 5)

          encoder.dispatch_3d(out_w, out_h, batch * channels, {16, 16, 1})
        end
      end

      # Bilinear 2x upsampling (GPU, higher quality)
      def upsample_bilinear_2x(input : Tensor, output : Tensor)
        ensure_initialized

        batch = input.shape[0]
        in_h = input.shape[1]
        in_w = input.shape[2]
        channels = input.shape[3]
        out_h = in_h * 2
        out_w = in_w * 2

        pipeline = get_pipeline("upsample_bilinear_2x")

        Metal::Dispatch.execute(pipeline) do |encoder|
          encoder.set_buffer(input.buffer.not_nil!, 0)
          encoder.set_buffer(output.buffer.not_nil!, 1)
          encoder.set_value(batch.to_u32, 2)
          encoder.set_value(in_h.to_u32, 3)
          encoder.set_value(in_w.to_u32, 4)
          encoder.set_value(channels.to_u32, 5)

          encoder.dispatch_3d(out_w, out_h, batch * channels, {16, 16, 1})
        end
      end

      # Fused resize + normalize for image preprocessing
      # input: [h, w, 3] HWC in [0,1]
      # output: [3, out_h, out_w] CHW normalized
      def resize_normalize(input : Tensor, output : Tensor, out_h : Int32, out_w : Int32)
        ensure_initialized

        in_h = input.shape[0]
        in_w = input.shape[1]

        pipeline = get_pipeline("resize_normalize_nchw")

        Metal::Dispatch.execute(pipeline) do |encoder|
          encoder.set_buffer(input.buffer.not_nil!, 0)
          encoder.set_buffer(output.buffer.not_nil!, 1)
          encoder.set_value(in_h.to_u32, 2)
          encoder.set_value(in_w.to_u32, 3)
          encoder.set_value(out_h.to_u32, 4)
          encoder.set_value(out_w.to_u32, 5)

          encoder.dispatch_3d(out_w, out_h, 3, {16, 16, 1})
        end
      end

      # ReLU forward (standalone)
      def relu_forward(input : Tensor, output : Tensor)
        ensure_initialized

        numel = input.numel

        pipeline = get_pipeline("relu_forward")

        Metal::Dispatch.execute(pipeline) do |encoder|
          encoder.set_buffer(input.buffer.not_nil!, 0)
          encoder.set_buffer(output.buffer.not_nil!, 1)

          encoder.dispatch_1d(numel, 256)
        end
      end
    end
  end
end
