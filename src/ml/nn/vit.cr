# Vision Transformer (ViT) components
# For MASt3R encoder

require "../autograd/variable"
require "../core/tensor"
require "./linear"
require "./layernorm"
require "./attention"
require "./gpu_ops"

module ML
  module NN
    # Patch Embedding: Convert image to sequence of patch embeddings
    # Image [B, C, H, W] -> Patches [B, num_patches, embed_dim]
    class PatchEmbedding
      getter patch_size : Int32
      getter embed_dim : Int32
      getter num_patches : Int32
      getter in_channels : Int32

      # Linear projection of flattened patches
      getter proj : Linear

      def initialize(
        img_size : Int32 = 224,
        @patch_size : Int32 = 16,
        @in_channels : Int32 = 3,
        @embed_dim : Int32 = 768,
        device : Tensor::Device = Tensor::Device::GPU
      )
        raise ArgumentError.new("Image size must be divisible by patch size") unless img_size % @patch_size == 0

        @num_patches = (img_size // @patch_size) ** 2
        patch_dim = in_channels * @patch_size * @patch_size

        @proj = Linear.new(patch_dim, @embed_dim, bias: true, device: device)
      end

      # Forward: [B, C, H, W] -> [B, num_patches, embed_dim]
      def forward(x : Autograd::Variable) : Autograd::Variable
        needs_grad = x.requires_grad? || @proj.weight.requires_grad? || (@proj.bias.try(&.requires_grad?) || false)

        if x.data.on_gpu? && GPUOps.available? && !needs_grad && ENV["GS_PATCH_EMBED_CPU"]?.nil?
          batch = x.data.shape[0]
          channels = x.data.shape[1]
          height = x.data.shape[2]
          width = x.data.shape[3]

          patches_h = height // @patch_size
          patches_w = width // @patch_size
          num_patches = patches_h * patches_w

          if channels == @in_channels && height % @patch_size == 0 && width % @patch_size == 0
            x_nhwc = Tensor.new(batch, height, width, channels, device: Tensor::Device::GPU)
            GPUOps.nchw_to_nhwc(x.data, x_nhwc)

            output = Tensor.new(batch, patches_h, patches_w, @embed_dim, device: Tensor::Device::GPU)
            weight = @proj.weight.data.reshape(@embed_dim, @in_channels, @patch_size, @patch_size)
            bias = @proj.bias.try(&.data)

            GPUOps.conv2d_forward(
              x_nhwc,
              weight,
              bias,
              output,
              stride: @patch_size,
              padding: 0
            )

            flat = output.reshape(batch, num_patches, @embed_dim)
            return Autograd::Variable.new(flat, false)
          end
        end

        x_data = x.data.on_cpu? ? x.data : x.data.to_cpu
        x_d = x_data.cpu_data.not_nil!

        batch = x_data.shape[0]
        channels = x_data.shape[1]
        height = x_data.shape[2]
        width = x_data.shape[3]

        patches_h = height // @patch_size
        patches_w = width // @patch_size
        num_patches = patches_h * patches_w
        patch_dim = channels * @patch_size * @patch_size

        # Extract and flatten patches
        patches = Tensor.new(batch, num_patches, patch_dim, device: Tensor::Device::CPU)
        p_d = patches.cpu_data.not_nil!

        batch.times do |b|
          patches_h.times do |ph|
            patches_w.times do |pw|
              patch_idx = ph * patches_w + pw

              channels.times do |c|
                @patch_size.times do |i|
                  @patch_size.times do |j|
                    src_h = ph * @patch_size + i
                    src_w = pw * @patch_size + j

                    src_idx = b * channels * height * width + c * height * width + src_h * width + src_w
                    dst_idx = b * num_patches * patch_dim + patch_idx * patch_dim + c * @patch_size * @patch_size + i * @patch_size + j

                    p_d[dst_idx] = x_d[src_idx]
                  end
                end
              end
            end
          end
        end

        patches = patches.to_gpu if x.data.on_gpu?
        patches_var = Autograd::Variable.new(patches, x.requires_grad?)

        # Project patches to embedding dimension
        @proj.forward(patches_var)
      end

      def call(x : Autograd::Variable) : Autograd::Variable
        forward(x)
      end

      def parameters : Array(Autograd::Variable)
        @proj.parameters
      end
    end

    # MLP block (Feed-Forward Network)
    class MLP
      getter fc1 : Linear
      getter fc2 : Linear
      getter dropout : Float32

      def initialize(
        in_features : Int32,
        hidden_features : Int32? = nil,
        out_features : Int32? = nil,
        @dropout : Float32 = 0.0_f32,
        device : Tensor::Device = Tensor::Device::GPU
      )
        hidden_dim = hidden_features || in_features * 4
        out_dim = out_features || in_features

        @fc1 = Linear.new(in_features, hidden_dim, device: device)
        @fc2 = Linear.new(hidden_dim, out_dim, device: device)
      end

      # Forward with GELU activation
      def forward(x : Autograd::Variable) : Autograd::Variable
        needs_grad = x.requires_grad? ||
          @fc1.weight.requires_grad? || (@fc1.bias.try(&.requires_grad?) || false) ||
          @fc2.weight.requires_grad? || (@fc2.bias.try(&.requires_grad?) || false)

        if x.data.on_gpu? && GPUOps.available? && !needs_grad
          input_shape = x.data.shape.to_a
          in_features = input_shape[input_shape.size - 1]

          batch_product = 1
          (input_shape.size - 1).times { |i| batch_product *= input_shape[i] }

          x_2d = if x.data.ndim == 2
                   x.data
                 else
                   x.data.reshape(batch_product, in_features)
                 end

          hidden = Tensor.new(batch_product, @fc1.out_features, device: Tensor::Device::GPU)
          GPUOps.linear_gelu_forward(x_2d, @fc1.weight.data, @fc1.bias.try(&.data), hidden)

          output = Tensor.new(batch_product, @fc2.out_features, device: Tensor::Device::GPU)
          GPUOps.linear_forward(hidden, @fc2.weight.data, @fc2.bias.try(&.data), output)

          output_shape = input_shape[0...-1] + [@fc2.out_features]
          output = output_shape.size == 2 ? output : output.reshape(Shape.new(output_shape))

          return Autograd::Variable.new(output, false)
        end

        h = @fc1.forward(x)
        h = gelu(h)
        @fc2.forward(h)
      end

      def call(x : Autograd::Variable) : Autograd::Variable
        forward(x)
      end

      def parameters : Array(Autograd::Variable)
        @fc1.parameters + @fc2.parameters
      end

      # GELU activation
      private def gelu(x : Autograd::Variable) : Autograd::Variable
        x_data = x.data.on_cpu? ? x.data : x.data.to_cpu
        x_d = x_data.cpu_data.not_nil!

        result = Tensor.new(x.data.shape, x.data.dtype, Tensor::Device::CPU)
        r_d = result.cpu_data.not_nil!

        # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt_2_pi = Math.sqrt(2.0 / Math::PI).to_f32

        x.data.numel.times do |i|
          v = x_d[i]
          r_d[i] = 0.5_f32 * v * (1.0_f32 + Math.tanh(sqrt_2_pi * (v + 0.044715_f32 * v * v * v)))
        end

        result = result.to_gpu if x.data.on_gpu?
        Autograd::Variable.new(result, x.requires_grad?)
      end
    end

    # Transformer Encoder Block
    # LayerNorm -> Self-Attention -> Residual -> LayerNorm -> MLP -> Residual
    class TransformerEncoderBlock
      getter attention : MultiHeadAttention
      getter mlp : MLP
      getter norm1 : LayerNorm
      getter norm2 : LayerNorm
      getter dropout : Float32

      def initialize(
        embed_dim : Int32,
        num_heads : Int32,
        mlp_ratio : Float32 = 4.0_f32,
        @dropout : Float32 = 0.0_f32,
        device : Tensor::Device = Tensor::Device::GPU
      )
        @attention = MultiHeadAttention.new(embed_dim, num_heads, dropout: @dropout, device: device)
        @mlp = MLP.new(embed_dim, (embed_dim * mlp_ratio).to_i32, embed_dim, @dropout, device)
        @norm1 = LayerNorm.new(embed_dim, device: device)
        @norm2 = LayerNorm.new(embed_dim, device: device)
      end

      # Forward: Pre-norm architecture (like ViT)
      # x: [batch, seq_len, embed_dim]
      def forward(x : Autograd::Variable, attn_mask : Tensor? = nil) : Autograd::Variable
        # Self-attention with residual
        normed = @norm1.forward(x)
        attn_out = @attention.self_attention(normed, attn_mask)
        x = add_residual(x, attn_out)

        # MLP with residual
        normed = @norm2.forward(x)
        mlp_out = @mlp.forward(normed)
        add_residual(x, mlp_out)
      end

      def call(x : Autograd::Variable, attn_mask : Tensor? = nil) : Autograd::Variable
        forward(x, attn_mask)
      end

      def parameters : Array(Autograd::Variable)
        @attention.parameters + @mlp.parameters + @norm1.parameters + @norm2.parameters
      end

      private def add_residual(x : Autograd::Variable, y : Autograd::Variable) : Autograd::Variable
        if x.data.on_gpu? && y.data.on_gpu? && GPUOps.available?
          result = Tensor.new(x.data.shape, x.data.dtype, Tensor::Device::GPU)
          GPUOps.add(x.data, y.data, result)
          result_var = Autograd::Variable.new(result, x.requires_grad? || y.requires_grad?)

          if result_var.requires_grad?
            result_var.is_leaf = false
            grad_fn = Autograd::AddBackward.new
            grad_fn.inputs = [x, y]
            result_var.grad_fn = grad_fn
          end

          return result_var
        end

        x_data = x.data.on_cpu? ? x.data : x.data.to_cpu
        y_data = y.data.on_cpu? ? y.data : y.data.to_cpu

        result = Tensor.new(x.data.shape, x.data.dtype, Tensor::Device::CPU)
        x_d = x_data.cpu_data.not_nil!
        y_d = y_data.cpu_data.not_nil!
        r_d = result.cpu_data.not_nil!

        x.data.numel.times { |i| r_d[i] = x_d[i] + y_d[i] }

        result = result.to_gpu if x.data.on_gpu?

        result_var = Autograd::Variable.new(result, x.requires_grad? || y.requires_grad?)

        if result_var.requires_grad?
          result_var.is_leaf = false
          grad_fn = Autograd::AddBackward.new
          grad_fn.inputs = [x, y]
          result_var.grad_fn = grad_fn
        end

        result_var
      end
    end

    # Full Vision Transformer Encoder
    class ViTEncoder
      getter patch_embed : PatchEmbedding
      getter blocks : Array(TransformerEncoderBlock)
      getter norm : LayerNorm

      # Learnable parameters
      getter cls_token : Autograd::Variable
      getter pos_embed : Autograd::Variable

      def initialize(
        img_size : Int32 = 224,
        patch_size : Int32 = 16,
        in_channels : Int32 = 3,
        embed_dim : Int32 = 768,
        depth : Int32 = 12,
        num_heads : Int32 = 12,
        mlp_ratio : Float32 = 4.0_f32,
        dropout : Float32 = 0.0_f32,
        device : Tensor::Device = Tensor::Device::GPU
      )
        @patch_embed = PatchEmbedding.new(img_size, patch_size, in_channels, embed_dim, device)

        num_patches = @patch_embed.num_patches

        # CLS token: [1, 1, embed_dim]
        cls_data = Tensor.randn(1, 1, embed_dim, device: device)
        cls_data.to_cpu!
        cls_data.cpu_data.not_nil!.map! { |x| x * 0.02_f32 }
        cls_data.to_gpu! if device.gpu?
        @cls_token = Autograd::Variable.new(cls_data, requires_grad: true)

        # Position embeddings: [1, num_patches + 1, embed_dim]
        pos_data = Tensor.randn(1, num_patches + 1, embed_dim, device: device)
        pos_data.to_cpu!
        pos_data.cpu_data.not_nil!.map! { |x| x * 0.02_f32 }
        pos_data.to_gpu! if device.gpu?
        @pos_embed = Autograd::Variable.new(pos_data, requires_grad: true)

        # Transformer blocks
        @blocks = Array(TransformerEncoderBlock).new(depth) do
          TransformerEncoderBlock.new(embed_dim, num_heads, mlp_ratio, dropout, device)
        end

        @norm = LayerNorm.new(embed_dim, device: device)
      end

      # Forward: Image -> Sequence of embeddings
      # x: [batch, channels, height, width]
      # Returns: [batch, num_patches + 1, embed_dim] (includes CLS token)
      def forward(x : Autograd::Variable) : Autograd::Variable
        batch = x.data.shape[0]

        # Patch embedding
        patches = @patch_embed.forward(x)  # [batch, num_patches, embed_dim]

        # Prepend CLS token (expand to batch)
        patches = prepend_cls_token(patches, batch)

        # Add position embeddings
        patches = add_position_embedding(patches)

        # Transformer blocks
        @blocks.each do |block|
          patches = block.forward(patches)
        end

        # Final layer norm
        @norm.forward(patches)
      end

      def call(x : Autograd::Variable) : Autograd::Variable
        forward(x)
      end

      # Get CLS token output (for classification)
      def forward_cls(x : Autograd::Variable) : Autograd::Variable
        output = forward(x)
        extract_cls_token(output)
      end

      def parameters : Array(Autograd::Variable)
        params = [@cls_token, @pos_embed]
        params += @patch_embed.parameters
        @blocks.each { |b| params += b.parameters }
        params += @norm.parameters
        params
      end

      private def prepend_cls_token(x : Autograd::Variable, batch : Int32) : Autograd::Variable
        x_data = x.data.on_cpu? ? x.data : x.data.to_cpu
        cls_data = @cls_token.data.on_cpu? ? @cls_token.data : @cls_token.data.to_cpu

        num_patches = x_data.shape[1]
        embed_dim = x_data.shape[2]

        # Result: [batch, num_patches + 1, embed_dim]
        result = Tensor.new(batch, num_patches + 1, embed_dim, device: Tensor::Device::CPU)

        x_d = x_data.cpu_data.not_nil!
        cls_d = cls_data.cpu_data.not_nil!
        r_d = result.cpu_data.not_nil!

        batch.times do |b|
          # Copy CLS token (first position)
          embed_dim.times do |e|
            r_d[b * (num_patches + 1) * embed_dim + e] = cls_d[e]
          end

          # Copy patches
          num_patches.times do |p|
            embed_dim.times do |e|
              src_idx = b * num_patches * embed_dim + p * embed_dim + e
              dst_idx = b * (num_patches + 1) * embed_dim + (p + 1) * embed_dim + e
              r_d[dst_idx] = x_d[src_idx]
            end
          end
        end

        result = result.to_gpu if x.data.on_gpu?
        Autograd::Variable.new(result, x.requires_grad?)
      end

      private def add_position_embedding(x : Autograd::Variable) : Autograd::Variable
        x_data = x.data.on_cpu? ? x.data : x.data.to_cpu
        pos_data = @pos_embed.data.on_cpu? ? @pos_embed.data : @pos_embed.data.to_cpu

        result = Tensor.new(x.data.shape, x.data.dtype, Tensor::Device::CPU)

        x_d = x_data.cpu_data.not_nil!
        p_d = pos_data.cpu_data.not_nil!
        r_d = result.cpu_data.not_nil!

        batch = x_data.shape[0]
        seq_len = x_data.shape[1]
        embed_dim = x_data.shape[2]

        batch.times do |b|
          seq_len.times do |s|
            embed_dim.times do |e|
              x_idx = b * seq_len * embed_dim + s * embed_dim + e
              p_idx = s * embed_dim + e  # pos_embed is [1, seq_len, embed_dim]
              r_d[x_idx] = x_d[x_idx] + p_d[p_idx]
            end
          end
        end

        result = result.to_gpu if x.data.on_gpu?
        Autograd::Variable.new(result, x.requires_grad? || @pos_embed.requires_grad?)
      end

      private def extract_cls_token(x : Autograd::Variable) : Autograd::Variable
        x_data = x.data.on_cpu? ? x.data : x.data.to_cpu
        x_d = x_data.cpu_data.not_nil!

        batch = x_data.shape[0]
        embed_dim = x_data.shape[2]

        result = Tensor.new(batch, embed_dim, device: Tensor::Device::CPU)
        r_d = result.cpu_data.not_nil!

        batch.times do |b|
          embed_dim.times do |e|
            r_d[b * embed_dim + e] = x_d[b * x_data.shape[1] * embed_dim + e]
          end
        end

        result = result.to_gpu if x.data.on_gpu?
        Autograd::Variable.new(result, x.requires_grad?)
      end
    end
  end
end
