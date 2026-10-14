require "./native"

# Native decoder ported from mlx-audio fish_s1_dac.py at
# e7488935ec2ebfafd38ef2b58e290d79f3d05dc7 (MIT).
module ML
  module Audio
    module Drama3
      # Shape controls used by the decoder. The defaults match the pinned
      # Fish S1 DAC checkpoint; smaller values are intended for numeric tests.
      class CodecConfig
        getter latent_channels : Int32
        getter codebook_dim : Int32
        getter semantic_codebook_size : Int32
        getter residual_codebooks : Int32
        getter residual_codebook_size : Int32
        getter post_layers : Int32
        getter transformer_heads : Int32
        getter transformer_head_dim : Int32
        getter transformer_ff_dim : Int32
        getter transformer_window : Int32
        getter quantizer_upsample_rates : Array(Int32)
        getter decoder_channels : Int32
        getter decoder_rates : Array(Int32)
        getter norm_eps : Float32
        getter rope_base : Float32

        def initialize(
          @latent_channels : Int32 = 1024,
          @codebook_dim : Int32 = 8,
          @semantic_codebook_size : Int32 = 4096,
          @residual_codebooks : Int32 = 9,
          @residual_codebook_size : Int32 = 1024,
          @post_layers : Int32 = 8,
          @transformer_heads : Int32 = 16,
          @transformer_head_dim : Int32 = 64,
          @transformer_ff_dim : Int32 = 3072,
          @transformer_window : Int32 = 128,
          @quantizer_upsample_rates : Array(Int32) = [2, 2],
          @decoder_channels : Int32 = 1536,
          @decoder_rates : Array(Int32) = [8, 8, 4, 2],
          @norm_eps : Float32 = 1.0e-5_f32,
          @rope_base : Float32 = 10_000.0_f32,
        )
          raise ArgumentError.new("latent_channels must equal heads * head_dim") unless @latent_channels == @transformer_heads * @transformer_head_dim
          raise ArgumentError.new("codebook dimensions and sizes must be positive") if @codebook_dim <= 0 || @semantic_codebook_size <= 0 || @residual_codebook_size <= 0
          raise ArgumentError.new("post_layers must be non-negative and residual_codebooks positive") if @post_layers < 0 || @residual_codebooks <= 0
          raise ArgumentError.new("transformer dimensions must be positive") if @transformer_heads <= 0 || @transformer_head_dim <= 0 || @transformer_ff_dim <= 0 || @transformer_window <= 0
          raise ArgumentError.new("upsample rates must be positive") if @quantizer_upsample_rates.any? { |rate| rate <= 0 }
          raise ArgumentError.new("decoder rates must be positive") if @decoder_rates.any? { |rate| rate <= 0 }
          divisor = 1 << @decoder_rates.size
          raise ArgumentError.new("decoder_channels must be divisible by 2^decoder_rates.size") if @decoder_channels % divisor != 0
          raise ArgumentError.new("decoder_channels must remain positive at every stage") if @decoder_channels >> @decoder_rates.size == 0
        end

        def self.default : CodecConfig
          new
        end
      end

      # Native, single-stream decoder for codebook-major DRAMA-3 DAC indices.
      # The input layout is [semantic, residual_0, ..., residual_N][frame].
      # Inference stays in Float32 and returns mono 44.1 kHz PCM samples.
      class Codec
        SAMPLE_RATE = 44_100

        # Optional stage callback used by parity tooling. Values are flattened
        # time-major matrices or mono waveform samples; the callback owns any
        # copy it needs to retain.
        property trace : Proc(String, Vector, Nil)?

        def initialize(
          @store : WeightStore,
          @config : CodecConfig = CodecConfig.default,
          @prefix : String = "codec.",
        )
          @trace = nil
        end

        def decode(codebooks : Array(Array(Int32))) : Vector
          latent = decode_codebooks(codebooks)
          trace_matrix("quantizer.codebook_sum", latent)
          latent = post_transformer(latent)
          trace_matrix("quantizer.post_module", latent)

          @config.quantizer_upsample_rates.reverse.each_with_index do |rate, stage|
            prefix = "quantizer.upsample.#{stage}"
            latent = conv_transpose1d(latent, "#{prefix}.0.conv", rate, rate)
            trace_matrix("#{prefix}.deconv", latent)
            latent = convnext(latent, "#{prefix}.1")
            trace_matrix("#{prefix}.convnext", latent)
          end

          waveform = decode_waveform(latent)
          expected_samples = codebooks[0].size
          @config.quantizer_upsample_rates.each { |rate| expected_samples *= rate }
          @config.decoder_rates.each { |rate| expected_samples *= rate }
          raise ArgumentError.new("DAC output length mismatch") unless waveform.size == expected_samples
          Ops.finite!(waveform, "DAC waveform")
          trace_vector("waveform", waveform)
        end

        private def decode_codebooks(codebooks : Array(Array(Int32))) : Matrix
          raise ArgumentError.new("at least the semantic codebook is required") if codebooks.empty?
          raise ArgumentError.new("too many codebooks for configured decoder") if codebooks.size > @config.residual_codebooks + 1

          frames = codebooks[0].size
          raise ArgumentError.new("codec input must contain at least one frame") if frames <= 0
          codebooks.each do |book|
            raise ArgumentError.new("all codebooks must have the same frame count") unless book.size == frames
          end

          latent = Array(Float32).new(frames * @config.latent_channels, 0.0_f32)
          codebooks.each_with_index do |indices, book_index|
            if book_index == 0
              prefix = "quantizer.semantic_quantizer.quantizers.0"
              vocab = @config.semantic_codebook_size
            else
              prefix = "quantizer.quantizer.quantizers.#{book_index - 1}"
              vocab = @config.residual_codebook_size
            end

            codebook = tensor("#{prefix}.codebook.weight")
            validate_shape(codebook, [vocab, @config.codebook_dim], "#{prefix}.codebook.weight")
            embedded = Array(Float32).new(frames * @config.codebook_dim, 0.0_f32)
            frames.times do |frame|
              # The reference clips token ids before embedding lookup.
              index = indices[frame].clamp(0, vocab - 1)
              src = index * @config.codebook_dim
              dst = frame * @config.codebook_dim
              @config.codebook_dim.times do |d|
                embedded[dst + d] = codebook.values[src + d]
              end
            end

            projected = conv1d(
              Matrix.new(frames, @config.codebook_dim, embedded),
              "#{prefix}.out_proj",
              1,
              1,
              1,
              1,
              true
            )
            latent.size.times { |i| latent[i] += projected.values[i] }
          end
          Matrix.new(frames, @config.latent_channels, latent)
        end

        private def post_transformer(input : Matrix) : Matrix
          hidden = input
          @config.post_layers.times do |layer_index|
            prefix = "quantizer.post_module.layers.#{layer_index}"
            hidden = transformer_layer(hidden, prefix, layer_index)
          end
          norm(hidden, "quantizer.post_module.norm.weight", @config.norm_eps)
        end

        private def transformer_layer(input : Matrix, prefix : String, _layer_index : Int32) : Matrix
          time = input.rows
          channels = input.cols
          heads = @config.transformer_heads
          head_dim = @config.transformer_head_dim
          expected = heads * head_dim
          raise ArgumentError.new("transformer input channel mismatch") unless channels == expected

          q = Array(Float32).new(time * channels, 0.0_f32)
          k = Array(Float32).new(time * channels, 0.0_f32)
          v = Array(Float32).new(time * channels, 0.0_f32)
          time.times do |t|
            normalized = rms_norm(input.row(t), "#{prefix}.attention_norm.weight", @config.norm_eps)
            qkv = linear(normalized, "#{prefix}.attention.wqkv.weight")
            raise ArgumentError.new("qkv projection has an unexpected width") unless qkv.size == channels * 3
            channels.times do |c|
              q[t * channels + c] = qkv[c]
              k[t * channels + c] = qkv[channels + c]
              v[t * channels + c] = qkv[2 * channels + c]
            end
            apply_rope!(q, t, channels, heads, head_dim)
            apply_rope!(k, t, channels, heads, head_dim)
          end

          attended = Array(Float32).new(time * channels, 0.0_f32)
          scale = (1.0 / Math.sqrt(head_dim.to_f32)).to_f32
          heads.times do |head|
            head_offset = head * head_dim
            time.times do |t|
              first_key = t - @config.transformer_window + 1
              first_key = 0 if first_key < 0
              scores = Array(Float32).new(t - first_key + 1, 0.0_f32)
              max_score = -Float32::INFINITY
              first_key.upto(t) do |key_t|
                score = 0.0_f32
                head_dim.times do |d|
                  score += q[t * channels + head_offset + d] * k[key_t * channels + head_offset + d]
                end
                score *= scale
                scores[key_t - first_key] = score
                max_score = score if score > max_score
              end

              denom = 0.0_f32
              scores.map! do |score|
                value = Math.exp((score - max_score).to_f64).to_f32
                denom += value
                value
              end
              head_dim.times do |d|
                value = 0.0_f32
                scores.each_with_index do |weight, i|
                  value += (weight / denom) * v[(first_key + i) * channels + head_offset + d]
                end
                attended[t * channels + head_offset + d] = value
              end
            end
          end

          attention_outputs = Array(Float32).new(time * channels, 0.0_f32)
          time.times do |t|
            projected = linear(attended[t * channels, channels], "#{prefix}.attention.wo.weight")
            scale_vector(projected, "#{prefix}.attention_layer_scale.gamma")
            channels.times do |c|
              attention_outputs[t * channels + c] = input.values[t * channels + c] + projected[c]
            end
          end
          after_attention = Matrix.new(time, channels, attention_outputs)

          ff_outputs = Array(Float32).new(time * channels, 0.0_f32)
          time.times do |t|
            row = rms_norm(after_attention.row(t), "#{prefix}.ffn_norm.weight", @config.norm_eps)
            w1 = linear(row, "#{prefix}.feed_forward.w1.weight")
            w3 = linear(row, "#{prefix}.feed_forward.w3.weight")
            raise ArgumentError.new("feed-forward projection width mismatch") unless w1.size == @config.transformer_ff_dim && w3.size == @config.transformer_ff_dim
            @config.transformer_ff_dim.times do |i|
              w1[i] = silu(w1[i]) * w3[i]
            end
            projected = linear(w1, "#{prefix}.feed_forward.w2.weight")
            scale_vector(projected, "#{prefix}.ffn_layer_scale.gamma")
            channels.times do |c|
              ff_outputs[t * channels + c] = after_attention.values[t * channels + c] + projected[c]
            end
          end
          Matrix.new(time, channels, ff_outputs)
        end

        private def apply_rope!(values : Array(Float32), position : Int32, channels : Int32, heads : Int32, head_dim : Int32)
          half_pairs = head_dim // 2
          heads.times do |head|
            base = position * channels + head * head_dim
            half_pairs.times do |pair|
              frequency = 1.0_f32 / @config.rope_base ** ((2 * pair).to_f32 / head_dim)
              angle = position.to_f32 * frequency
              cosine = Math.cos(angle).to_f32
              sine = Math.sin(angle).to_f32
              left = values[base + pair * 2]
              right = values[base + pair * 2 + 1]
              values[base + pair * 2] = left * cosine - right * sine
              values[base + pair * 2 + 1] = right * cosine + left * sine
            end
          end
        end

        private def decode_waveform(input : Matrix) : Array(Float32)
          x = conv1d(input, "decoder.model.0", 7, 1, 1, 1, true)
          trace_matrix("decoder.model.0", x)
          channels = @config.decoder_channels
          @config.decoder_rates.each_with_index do |rate, stage|
            block_index = stage + 1
            input_dim = channels >> stage
            output_dim = channels >> (stage + 1)
            prefix = "decoder.model.#{block_index}.block"
            x = snake(x, "#{prefix}.0.alpha")
            trace_matrix("decoder.model.#{block_index}.block.0", x)
            x = conv_transpose1d(x, "#{prefix}.1", rate * 2, rate, true)
            trace_matrix("decoder.model.#{block_index}.block.1", x)
            3.times do |unit_index|
              unit_prefix = "#{prefix}.#{unit_index + 2}.block"
              x = residual_unit(x, unit_prefix, output_dim, [1, 3, 9][unit_index])
              trace_matrix("decoder.model.#{block_index}.block.#{unit_index + 2}", x)
            end
            raise ArgumentError.new("decoder stage output width mismatch") unless x.cols == output_dim
          end

          final_index = @config.decoder_rates.size + 1
          x = snake(x, "decoder.model.#{final_index}.alpha")
          trace_matrix("decoder.model.#{final_index}", x)
          x = conv1d(x, "decoder.model.#{final_index + 1}", 7, 1, 1, 1, true)
          trace_matrix("decoder.model.#{final_index + 1}", x)
          samples = Array(Float32).new(x.rows, 0.0_f32)
          x.rows.times do |t|
            samples[t] = Math.tanh(x.values[t * x.cols]).to_f32
          end
          samples
        end

        private def residual_unit(input : Matrix, prefix : String, channels : Int32, dilation : Int32) : Matrix
          raise ArgumentError.new("residual unit input width mismatch") unless input.cols == channels
          y = snake(input, "#{prefix}.0.alpha")
          y = conv1d(y, "#{prefix}.1", 7, 1, 1, dilation, true)
          y = snake(y, "#{prefix}.2.alpha")
          y = conv1d(y, "#{prefix}.3", 1, 1, 1, 1, true)
          values = Array(Float32).new(input.values.size, 0.0_f32)
          values.size.times { |i| values[i] = input.values[i] + y.values[i] }
          Matrix.new(input.rows, input.cols, values)
        end

        private def convnext(input : Matrix, prefix : String) : Matrix
          channels = input.cols
          x = conv1d(input, "#{prefix}.dwconv.conv", 7, 1, channels, 1, false)
          normalized = Array(Float32).new(x.values.size, 0.0_f32)
          gamma = vector("#{prefix}.norm.weight")
          beta = vector("#{prefix}.norm.bias")
          raise ArgumentError.new("ConvNeXt norm width mismatch") unless gamma.size == channels && beta.size == channels
          x.rows.times do |t|
            mean = 0.0_f32
            channels.times { |c| mean += x.values[t * channels + c] }
            mean /= channels
            variance = 0.0_f32
            channels.times do |c|
              diff = x.values[t * channels + c] - mean
              variance += diff * diff
            end
            variance /= channels
            inv = 1.0_f32 / Math.sqrt(variance + 1.0e-6_f32)
            channels.times do |c|
              normalized[t * channels + c] = (x.values[t * channels + c] - mean) * inv * gamma[c] + beta[c]
            end
          end

          hidden_dim = channels * 4
          projected = Array(Float32).new(x.rows * hidden_dim, 0.0_f32)
          x.rows.times do |t|
            row = linear(normalized[t * channels, channels], "#{prefix}.pwconv1.weight", "#{prefix}.pwconv1.bias")
            raise ArgumentError.new("ConvNeXt hidden width mismatch") unless row.size == hidden_dim
            hidden_dim.times do |i|
              row[i] = gelu(row[i])
              projected[t * hidden_dim + i] = row[i]
            end
          end

          output = Array(Float32).new(x.values.size, 0.0_f32)
          gamma_scale = vector("#{prefix}.gamma")
          raise ArgumentError.new("ConvNeXt gamma width mismatch") unless gamma_scale.size == channels
          x.rows.times do |t|
            row = linear(projected[t * hidden_dim, hidden_dim], "#{prefix}.pwconv2.weight", "#{prefix}.pwconv2.bias")
            channels.times do |c|
              output[t * channels + c] = input.values[t * channels + c] + row[c] * gamma_scale[c]
            end
          end
          Matrix.new(x.rows, channels, output)
        end

        private def conv1d(
          input : Matrix,
          prefix : String,
          kernel : Int32,
          stride : Int32,
          groups : Int32,
          dilation : Int32,
          weight_normalized : Bool,
        ) : Matrix
          in_channels = input.cols
          weight_name = weight_normalized ? "#{prefix}.weight_v" : "#{prefix}.weight"
          weight = tensor(weight_name)
          raise ArgumentError.new("invalid Conv1d groups") if groups <= 0 || in_channels % groups != 0
          out_channels = weight.shape[0]
          in_per_group = in_channels // groups
          expected = [out_channels, in_per_group, kernel]
          validate_shape(weight, expected, weight_name)
          raise ArgumentError.new("Conv1d output channels not divisible by groups") if out_channels % groups != 0
          # WeightStore caches decoded tensor values; do not mutate its shared copy.
          effective = weight.values.dup
          if weight_normalized
            g = tensor("#{prefix}.weight_g")
            raise ArgumentError.new("Conv1d weight_g width mismatch") unless g.values.size == out_channels
            out_channels.times do |out_channel|
              norm_sq = 0.0_f32
              in_per_group.times do |i|
                kernel.times do |tap|
                  value = weight.values[(out_channel * in_per_group + i) * kernel + tap]
                  norm_sq += value * value
                end
              end
              scale = g.values[out_channel] / (Math.sqrt(norm_sq).to_f32 + 1.0e-12_f32)
              in_per_group.times do |i|
                kernel.times do |tap|
                  index = (out_channel * in_per_group + i) * kernel + tap
                  effective[index] *= scale
                end
              end
            end
          end

          bias = vector("#{prefix}.bias")
          raise ArgumentError.new("Conv1d bias width mismatch") unless bias.size == out_channels
          output_values = Array(Float32).new(input.rows * out_channels, 0.0_f32)
          left_pad = (kernel - 1) * dilation
          out_per_group = out_channels // groups
          input.rows.times do |t|
            out_channels.times do |out_channel|
              group = out_channel // out_per_group
              sum = bias[out_channel]
              in_per_group.times do |in_local|
                in_channel = group * in_per_group + in_local
                kernel.times do |tap|
                  input_t = t * stride - left_pad + tap * dilation
                  next if input_t < 0 || input_t >= input.rows
                  weight_index = (out_channel * in_per_group + in_local) * kernel + tap
                  sum += input.values[input_t * in_channels + in_channel] * effective[weight_index]
                end
              end
              output_values[t * out_channels + out_channel] = sum
            end
          end
          Matrix.new(input.rows, out_channels, output_values)
        end

        private def conv_transpose1d(input : Matrix, prefix : String, kernel : Int32, stride : Int32, weight_normalized : Bool = false) : Matrix
          weight_name = weight_normalized ? "#{prefix}.weight_v" : "#{prefix}.weight"
          weight = tensor(weight_name)
          in_channels = input.cols
          raise ArgumentError.new("invalid ConvTranspose1d weight shape") unless weight.shape.size == 3 && weight.shape[0] == in_channels && weight.shape[2] == kernel
          out_channels = weight.shape[1]
          # WeightStore caches decoded tensor values; do not mutate its shared copy.
          effective = weight.values.dup
          if weight_normalized
            g = tensor("#{prefix}.weight_g")
            raise ArgumentError.new("ConvTranspose1d weight_g width mismatch") unless g.values.size == in_channels
            in_channels.times do |in_channel|
              norm_sq = 0.0_f32
              (out_channels * kernel).times do |j|
                value = weight.values[in_channel * out_channels * kernel + j]
                norm_sq += value * value
              end
              scale = g.values[in_channel] / (Math.sqrt(norm_sq).to_f32 + 1.0e-12_f32)
              (out_channels * kernel).times do |j|
                effective[in_channel * out_channels * kernel + j] *= scale
              end
            end
          end
          bias = vector("#{prefix}.bias")
          raise ArgumentError.new("ConvTranspose1d bias width mismatch") unless bias.size == out_channels
          frames = input.rows * stride
          output_values = Array(Float32).new(frames * out_channels, 0.0_f32)
          input.rows.times do |t|
            in_channels.times do |in_channel|
              kernel.times do |tap|
                output_t = t * stride + tap
                next if output_t >= frames
                out_channels.times do |out_channel|
                  wi = (in_channel * out_channels + out_channel) * kernel + tap
                  output_values[output_t * out_channels + out_channel] += input.values[t * in_channels + in_channel] * effective[wi]
                end
              end
            end
          end
          frames.times do |t|
            out_channels.times { |c| output_values[t * out_channels + c] += bias[c] }
          end
          Matrix.new(frames, out_channels, output_values)
        end

        private def snake(input : Matrix, alpha_name : String) : Matrix
          alpha = tensor(alpha_name)
          raise ArgumentError.new("Snake alpha width mismatch") unless alpha.values.size == input.cols
          output_values = Array(Float32).new(input.values.size, 0.0_f32)
          input.rows.times do |t|
            input.cols.times do |c|
              value = input.values[t * input.cols + c]
              a = alpha[c]
              angle = a * value
              output_values[t * input.cols + c] = value + Math.sin(angle).to_f32 ** 2 / (a + 1.0e-9_f32)
            end
          end
          Matrix.new(input.rows, input.cols, output_values)
        end

        private def norm(input : Matrix, weight_name : String, eps : Float32) : Matrix
          rows = Array(Float32).new(input.values.size, 0.0_f32)
          input.rows.times do |t|
            row = rms_norm(input.row(t), weight_name, eps)
            input.cols.times { |c| rows[t * input.cols + c] = row[c] }
          end
          Matrix.new(input.rows, input.cols, rows)
        end

        private def rms_norm(input : Array(Float32), weight_name : String, eps : Float32) : Array(Float32)
          weight = vector(weight_name)
          raise ArgumentError.new("RMS norm width mismatch for #{weight_name}") unless weight.size == input.size
          mean_square = 0.0_f32
          input.each { |value| mean_square += value * value }
          mean_square /= input.size
          scale = 1.0_f32 / Math.sqrt(mean_square + eps).to_f32
          Array(Float32).new(input.size) { |i| input[i] * scale * weight[i] }
        end

        private def linear(input : Array(Float32), weight_name : String, bias_name : String? = nil) : Array(Float32)
          result = Ops.linear(input, matrix(weight_name))
          if name = bias_name
            bias = vector(name)
            raise ArgumentError.new("linear bias width mismatch for #{name}") unless bias.size == result.size
            result.size.times { |i| result[i] += bias[i] }
          end
          result
        end

        private def scale_vector(values : Array(Float32), scale_name : String)
          scale = vector(scale_name)
          raise ArgumentError.new("layer scale width mismatch for #{scale_name}") unless scale.size == values.size
          values.size.times { |i| values[i] *= scale[i] }
        end

        private def silu(value : Float32) : Float32
          value / (1.0_f32 + Math.exp(-value.to_f64).to_f32)
        end

        private def gelu(value : Float32) : Float32
          0.5_f32 * value * (1.0_f32 + Math.erf(value / Math.sqrt(2.0)).to_f32)
        end

        private def validate_shape(tensor : Tensor, expected : Array(Int32), name : String)
          raise ArgumentError.new("#{name} shape #{tensor.shape} does not match #{expected}") unless tensor.shape == expected
        end

        private def trace_matrix(name : String, matrix : Matrix) : Matrix
          @trace.try &.call(name, matrix.values.dup)
          matrix
        end

        private def trace_vector(name : String, values : Vector) : Vector
          @trace.try &.call(name, values.dup)
          values
        end

        # Checkpoints retain PyTorch parametrization names while the MLX port
        # asks for the names produced by its sanitize() method.
        private def tensor(name : String) : Tensor
          candidates = ["#{@prefix}#{name}"]
          if name.ends_with?(".weight_v") || name.ends_with?(".weight_g")
            stem = name[0...(name.size - 9)]
            original = name.ends_with?(".weight_v") ? "original1" : "original0"
            candidates << "#{@prefix}#{stem}.conv.parametrizations.weight.#{original}"
            candidates << "#{@prefix}#{stem}.parametrizations.weight.#{original}"
          elsif name.ends_with?(".bias")
            stem = name[0...(name.size - 5)]
            candidates << "#{@prefix}#{stem}.conv.bias"
          end

          candidates.each do |candidate|
            begin
              return @store.tensor(candidate)
            rescue ex : ArgumentError
              # Try the next spelling; WeightStore only raises this for a missing key.
            end
          end
          raise ArgumentError.new("missing codec tensor #{@prefix}#{name}; tried #{candidates.join(", ")}")
        end

        private def vector(name : String) : Vector
          tensor(name).values
        end

        private def matrix(name : String) : Matrix
          @store.matrix("#{@prefix}#{name}")
        end
      end
    end
  end
end
