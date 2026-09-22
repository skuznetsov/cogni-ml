# Exact CPU reference for one Qwen-Image 2.1 single-stream transformer block.
#
# This is intentionally parameterized so tiny fixtures can be checked against
# the upstream PyTorch equations before the fixed 4096-wide Metal path is
# admitted. Inputs and outputs use row-major [token, hidden] arrays.

require "./compute"

module ML::GGUF
  struct QwenImage21BlockConfig
    getter hidden_dim : Int32
    getter heads : Int32
    getter head_dim : Int32
    getter intermediate_dim : Int32
    getter axes_dims : StaticArray(Int32, 3)
    getter eps : Float32
    getter rope_theta : Float32

    def initialize(@hidden_dim, @heads, @head_dim, @intermediate_dim,
                   @axes_dims, @eps = 1e-6_f32, @rope_theta = 10_000.0_f32)
      raise ArgumentError.new("heads * head_dim must equal hidden_dim") unless @heads * @head_dim == @hidden_dim
      raise ArgumentError.new("RoPE axes must sum to head_dim") unless @axes_dims.sum == @head_dim
      raise ArgumentError.new("RoPE axis dimensions must be positive and even") unless @axes_dims.all? { |dim| dim > 0 && dim.even? }
    end
  end

  class QwenImage21BlockWeights
    getter to_q : QuantWeight
    getter to_k : QuantWeight
    getter to_v : QuantWeight
    getter to_out : QuantWeight
    getter norm_q : Array(Float32)
    getter norm_k : Array(Float32)
    getter gate_up : QuantWeight
    getter mlp_out : QuantWeight

    def initialize(@to_q, @to_k, @to_v, @to_out, @norm_q, @norm_k, @gate_up, @mlp_out)
    end
  end

  # Executes a complete transformer-block sequence behind one residency
  # boundary. Implementations may keep hidden states and scratch storage on an
  # accelerator while the outer transformer retains the exact CPU reference
  # for sequence construction and top-level projections.
  module QwenImage21LayerStackBackend
    abstract def forward_layers(
      hidden : Array(Float32), token_count : Int32,
      modulation : Array(Float32),
      positions : Array(StaticArray(Int32, 3)),
      image_ids : Array(Int32),
      layers : Array(QwenImage21BlockWeights),
      config : QwenImage21BlockConfig,
      key_valid : Array(Bool)?,
    ) : Array(Float32)
  end

  module QwenImage21BlockCPU
    # `modulation` is already selected per token and laid out as
    # [mod1.scale, mod1.gate, mod2.scale, mod2.gate]. `positions` holds
    # frame/height/width indices. Text tokens use image_id=-1; image tokens
    # sharing a non-negative id attend bidirectionally within that block.
    def self.forward(
      hidden : Array(Float32),
      token_count : Int32,
      modulation : Array(Float32),
      positions : Array(StaticArray(Int32, 3)),
      image_ids : Array(Int32),
      weights : QwenImage21BlockWeights,
      config : QwenImage21BlockConfig,
      key_valid : Array(Bool)? = nil,
      backend : ComputeBackend = F32Backend.new,
    ) : Array(Float32)
      dim = config.hidden_dim
      expected_hidden = token_count * dim
      raise ArgumentError.new("hidden size mismatch") unless hidden.size == expected_hidden
      raise ArgumentError.new("modulation size mismatch") unless modulation.size == token_count * 4 * dim
      raise ArgumentError.new("positions size mismatch") unless positions.size == token_count
      raise ArgumentError.new("image_ids size mismatch") unless image_ids.size == token_count
      raise ArgumentError.new("key_valid size mismatch") if key_valid && key_valid.size != token_count
      validate_weights(weights, config)

      zeros = Array(Float32).new(dim, 0.0_f32)

      norm1 = layer_norm(hidden, token_count, dim, config.eps)
      gate1 = Array(Float32).new(expected_hidden, 0.0_f32)
      token_count.times do |token|
        hidden_off = token * dim
        mod_off = token * 4 * dim
        dim.times do |j|
          norm1[hidden_off + j] *= 1.0_f32 + modulation[mod_off + j]
          gate1[hidden_off + j] = Math.tanh(modulation[mod_off + dim + j])
        end
      end

      q = backend.matmul(norm1, token_count, weights.to_q, zeros)
      k = backend.matmul(norm1, token_count, weights.to_k, zeros)
      v = backend.matmul(norm1, token_count, weights.to_v, zeros)
      rmsnorm_heads!(q, token_count, config, weights.norm_q)
      rmsnorm_heads!(k, token_count, config, weights.norm_k)
      apply_rope!(q, token_count, config, positions)
      apply_rope!(k, token_count, config, positions)

      attended = block_causal_attention(q, k, v, token_count, config, image_ids, key_valid)
      projected = backend.matmul(attended, token_count, weights.to_out, zeros)
      state = Array(Float32).new(expected_hidden) { |i| hidden[i] + gate1[i] * projected[i] }

      norm2 = layer_norm(state, token_count, dim, config.eps)
      gate2 = Array(Float32).new(expected_hidden, 0.0_f32)
      token_count.times do |token|
        hidden_off = token * dim
        mod_off = token * 4 * dim + 2 * dim
        dim.times do |j|
          norm2[hidden_off + j] *= 1.0_f32 + modulation[mod_off + j]
          gate2[hidden_off + j] = Math.tanh(modulation[mod_off + dim + j])
        end
      end

      fused = backend.matmul(
        norm2,
        token_count,
        weights.gate_up,
        Array(Float32).new(2 * config.intermediate_dim, 0.0_f32),
      )
      activated = Array(Float32).new(token_count * config.intermediate_dim, 0.0_f32)
      token_count.times do |token|
        fused_off = token * 2 * config.intermediate_dim
        out_off = token * config.intermediate_dim
        config.intermediate_dim.times do |j|
          gate = fused[fused_off + j]
          projection = fused[fused_off + config.intermediate_dim + j]
          activated[out_off + j] = silu(gate) * projection
        end
      end
      mlp = backend.matmul(
        activated,
        token_count,
        weights.mlp_out,
        zeros,
      )
      Array(Float32).new(expected_hidden) { |i| state[i] + gate2[i] * mlp[i] }
    end

    private def self.validate_weights(weights : QwenImage21BlockWeights, config : QwenImage21BlockConfig) : Nil
      dim = config.hidden_dim
      {weights.to_q, weights.to_k, weights.to_v, weights.to_out}.each do |weight|
        raise ArgumentError.new("attention weight shape mismatch") unless weight.in_dim == dim && weight.out_dim == dim
      end
      raise ArgumentError.new("norm_q size mismatch") unless weights.norm_q.size == config.head_dim
      raise ArgumentError.new("norm_k size mismatch") unless weights.norm_k.size == config.head_dim
      unless weights.gate_up.in_dim == dim && weights.gate_up.out_dim == 2 * config.intermediate_dim
        raise ArgumentError.new("gate_up weight shape mismatch")
      end
      unless weights.mlp_out.in_dim == config.intermediate_dim && weights.mlp_out.out_dim == dim
        raise ArgumentError.new("MLP output weight shape mismatch")
      end
    end

    private def self.layer_norm(input : Array(Float32), rows : Int32, dim : Int32, eps : Float32) : Array(Float32)
      output = Array(Float32).new(input.size, 0.0_f32)
      rows.times do |row|
        off = row * dim
        mean = 0.0_f64
        dim.times { |j| mean += input[off + j] }
        mean /= dim
        variance = 0.0_f64
        dim.times do |j|
          delta = input[off + j] - mean
          variance += delta * delta
        end
        inv_std = 1.0_f64 / Math.sqrt(variance / dim + eps)
        dim.times { |j| output[off + j] = ((input[off + j] - mean) * inv_std).to_f32 }
      end
      output
    end

    private def self.rmsnorm_heads!(values : Array(Float32), tokens : Int32,
                                    config : QwenImage21BlockConfig,
                                    weight : Array(Float32)) : Nil
      tokens.times do |token|
        config.heads.times do |head|
          off = token * config.hidden_dim + head * config.head_dim
          mean_square = 0.0_f64
          config.head_dim.times { |j| mean_square += values[off + j].to_f64 ** 2 }
          inv_rms = 1.0_f64 / Math.sqrt(mean_square / config.head_dim + config.eps)
          config.head_dim.times do |j|
            values[off + j] = (values[off + j] * inv_rms * weight[j]).to_f32
          end
        end
      end
    end

    private def self.apply_rope!(values : Array(Float32), tokens : Int32,
                                 config : QwenImage21BlockConfig,
                                 positions : Array(StaticArray(Int32, 3))) : Nil
      tokens.times do |token|
        config.heads.times do |head|
          base = token * config.hidden_dim + head * config.head_dim
          axis_offset = 0
          3.times do |axis|
            axis_dim = config.axes_dims[axis]
            (axis_dim // 2).times do |pair|
              index = base + axis_offset + pair * 2
              frequency = 1.0_f64 / (config.rope_theta.to_f64 ** ((pair * 2).to_f64 / axis_dim))
              angle = positions[token][axis] * frequency
              cos = Math.cos(angle)
              sin = Math.sin(angle)
              real = values[index]
              imag = values[index + 1]
              values[index] = (real * cos - imag * sin).to_f32
              values[index + 1] = (real * sin + imag * cos).to_f32
            end
            axis_offset += axis_dim
          end
        end
      end
    end

    private def self.block_causal_attention(
      q : Array(Float32), k : Array(Float32), v : Array(Float32),
      tokens : Int32, config : QwenImage21BlockConfig,
      image_ids : Array(Int32), key_valid : Array(Bool)?,
    ) : Array(Float32)
      output = Array(Float32).new(tokens * config.hidden_dim, 0.0_f32)
      scale = 1.0_f64 / Math.sqrt(config.head_dim)

      tokens.times do |query_token|
        config.heads.times do |head|
          q_off = query_token * config.hidden_dim + head * config.head_dim
          allowed = [] of Int32
          scores = [] of Float64
          tokens.times do |key_token|
            next if key_valid && !key_valid[key_token]
            same_image = image_ids[query_token] >= 0 && image_ids[query_token] == image_ids[key_token]
            next unless query_token >= key_token || same_image
            k_off = key_token * config.hidden_dim + head * config.head_dim
            score = 0.0_f64
            config.head_dim.times { |j| score += q[q_off + j] * k[k_off + j] }
            allowed << key_token
            scores << score * scale
          end
          raise ArgumentError.new("attention row has no valid keys") if allowed.empty?
          max_score = scores.max
          denominator = 0.0_f64
          scores.map! do |score|
            value = Math.exp(score - max_score)
            denominator += value
            value
          end
          allowed.each_with_index do |key_token, index|
            probability = scores[index] / denominator
            v_off = key_token * config.hidden_dim + head * config.head_dim
            config.head_dim.times { |j| output[q_off + j] += (probability * v[v_off + j]).to_f32 }
          end
        end
      end
      output
    end

    @[AlwaysInline]
    private def self.silu(value : Float32) : Float32
      (value / (1.0_f32 + Math.exp(-value))).to_f32
    end
  end
end
