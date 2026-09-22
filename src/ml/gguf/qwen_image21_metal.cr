require "./qwen_image21_block"
require "./qwen35_metal"

# Hybrid reference backend for Qwen-Image 2.1 block admission.
#
# Projection matmuls use the established Qwen 3.5 Metal K-quant kernels while
# normalization, RoPE, attention, and elementwise math remain on the CPU. This
# deliberately narrow boundary supports exact one-block CPU/Metal parity before
# the full block is fused or made resident on-device.
module ML::GGUF
  class QwenImage21MetalProjectionBackend
    include ComputeBackend

    getter metal_projection_count = 0

    def initialize(@strict : Bool = true)
      @cpu = F32Backend.new
    end

    def self.available? : Bool
      {% if flag?(:cpu_only) %}
        false
      {% else %}
        Qwen35Metal.available?
      {% end %}
    end

    def matmul(x : Array(Float32), rows : Int32, qw : QuantWeight,
               bias : Array(Float32)) : Array(Float32)
      unless bias.size == qw.out_dim
        raise ArgumentError.new("bias size #{bias.size} does not match projection output #{qw.out_dim}")
      end

      {% if flag?(:cpu_only) %}
        if @strict
          raise ArgumentError.new("Metal disabled (cpu_only)")
        else
          @cpu.matmul(x, rows, qw, bias)
        end
      {% else %}
        if result = Qwen35Metal.matmul(qw, x, rows)
          @metal_projection_count += 1
          add_bias!(result, rows, qw.out_dim, bias)
          result
        elsif @strict
          raise ArgumentError.new(
            "no Qwen-Image Metal projection route for #{qw.type.name} #{qw.in_dim}x#{qw.out_dim} batch=#{rows}"
          )
        else
          @cpu.matmul(x, rows, qw, bias)
        end
      {% end %}
    end

    def layer_norm!(x : Array(Float32), n_pos : Int32, dim : Int32,
                    w : Array(Float32), b : Array(Float32)) : Nil
      @cpu.layer_norm!(x, n_pos, dim, w, b)
    end

    def softmax_row!(scores : Array(Float32), offset : Int32, len : Int32) : Nil
      @cpu.softmax_row!(scores, offset, len)
    end

    def gelu(x : Float32) : Float32
      @cpu.gelu(x)
    end

    def dot(a : Array(Float32), a_off : Int32, b : Array(Float32),
            b_off : Int32, len : Int32) : Float32
      @cpu.dot(a, a_off, b, b_off, len)
    end

    private def add_bias!(output : Array(Float32), rows : Int32, out_dim : Int32,
                          bias : Array(Float32)) : Nil
      return if bias.all?(&.zero?)
      rows.times do |row|
        offset = row * out_dim
        out_dim.times { |column| output[offset + column] += bias[column] }
      end
    end
  end

  struct QwenImage21MetalBlockStats
    getter command_buffers : Int32
    getter projection_dispatches : Int32
    getter intermediate_readbacks : Int32
    getter final_readbacks : Int32

    def initialize(@command_buffers, @projection_dispatches,
                   @intermediate_readbacks, @final_readbacks)
    end
  end

  class QwenImage21MetalBlockResult
    getter hidden : Array(Float32)
    getter stats : QwenImage21MetalBlockStats

    def initialize(@hidden, @stats)
    end
  end

  # Exact single-command-buffer block-stack path. All normalization, Q/K RoPE,
  # segmented block-causal attention, projections, SwiGLU, and residual
  # updates remain in Metal buffers; only the final hidden state is read back.
  module QwenImage21MetalBlock
    def self.available? : Bool
      {% if flag?(:cpu_only) %}
        false
      {% else %}
        Qwen35Metal.available?
      {% end %}
    end

    {% if flag?(:cpu_only) %}
      def self.forward(
        hidden : Array(Float32), token_count : Int32,
        modulation : Array(Float32),
        positions : Array(StaticArray(Int32, 3)),
        image_ids : Array(Int32),
        weights : QwenImage21BlockWeights,
        config : QwenImage21BlockConfig,
        key_valid : Array(Bool)? = nil,
      ) : QwenImage21MetalBlockResult
        raise "Metal disabled (cpu_only)"
      end

      def self.forward_layers(
        hidden : Array(Float32), token_count : Int32,
        modulation : Array(Float32),
        positions : Array(StaticArray(Int32, 3)),
        image_ids : Array(Int32),
        layers : Array(QwenImage21BlockWeights),
        config : QwenImage21BlockConfig,
        key_valid : Array(Bool)? = nil,
      ) : QwenImage21MetalBlockResult
        raise "Metal disabled (cpu_only)"
      end
    {% else %}
      SOURCE = {{ read_file("#{__DIR__}/kernels/qwen_image21.metal") }}

      def self.forward(
        hidden : Array(Float32), token_count : Int32,
        modulation : Array(Float32),
        positions : Array(StaticArray(Int32, 3)),
        image_ids : Array(Int32),
        weights : QwenImage21BlockWeights,
        config : QwenImage21BlockConfig,
        key_valid : Array(Bool)? = nil,
      ) : QwenImage21MetalBlockResult
        forward_layers(
          hidden, token_count, modulation, positions, image_ids,
          [weights], config, key_valid,
        )
      end

      def self.forward_layers(
        hidden : Array(Float32), token_count : Int32,
        modulation : Array(Float32),
        positions : Array(StaticArray(Int32, 3)),
        image_ids : Array(Int32),
        layers : Array(QwenImage21BlockWeights),
        config : QwenImage21BlockConfig,
        key_valid : Array(Bool)? = nil,
      ) : QwenImage21MetalBlockResult
        validate_inputs(hidden, token_count, modulation, positions, image_ids,
          layers, config, key_valid)
        valid = key_valid || Array(Bool).new(token_count, true)
        validate_attention_rows(image_ids, valid)
        ML::Metal::Device.init!

        dim = config.hidden_dim
        intermediate = config.intermediate_dim
        hidden_count = token_count * dim
        hidden_bytes = hidden_count.to_i64 * sizeof(Float32)
        buffers = [] of ML::MetalBuffer

        begin
          current_hidden_buf = upload_f32(hidden, buffers)
          next_hidden_buf = allocate(hidden_bytes, buffers)
          modulation_buf = upload_f32(modulation, buffers)
          positions_buf = upload_i32(positions.flat_map(&.to_a), buffers)
          image_ids_buf = upload_i32(image_ids, buffers)
          key_valid_buf = upload_u8(valid.map { |value| value ? 1_u8 : 0_u8 }, buffers)
          q_norm_weight_bufs = layers.map { |weights| upload_f32(weights.norm_q, buffers) }
          k_norm_weight_bufs = layers.map { |weights| upload_f32(weights.norm_k, buffers) }

          norm1_buf = allocate(hidden_bytes, buffers)
          gate1_buf = allocate(hidden_bytes, buffers)
          q_buf = allocate(hidden_bytes, buffers)
          k_buf = allocate(hidden_bytes, buffers)
          v_buf = allocate(hidden_bytes, buffers)
          attended_buf = allocate(hidden_bytes, buffers)
          projected_buf = allocate(hidden_bytes, buffers)
          state_buf = allocate(hidden_bytes, buffers)
          norm2_buf = allocate(hidden_bytes, buffers)
          gate2_buf = allocate(hidden_bytes, buffers)
          fused_buf = allocate(token_count.to_i64 * 2_i64 * intermediate * sizeof(Float32), buffers)
          activated_buf = allocate(token_count.to_i64 * intermediate * sizeof(Float32), buffers)
          mlp_buf = allocate(hidden_bytes, buffers)

          command = ML::Metal::CommandBuffer.new
          encoder = ML::Metal::ComputeEncoder.new(command)
          layers.each_with_index do |weights, layer_index|
            encode_layernorm_modulate_gate(
              encoder, current_hidden_buf, modulation_buf, norm1_buf, gate1_buf,
              token_count, dim, config.eps,
            )
            unless Qwen35Metal.encode_matmul_many_to_buffers(
                     encoder,
                     [weights.to_q, weights.to_k, weights.to_v],
                     norm1_buf,
                     [q_buf, k_buf, v_buf],
                     token_count,
                   )
              raise ArgumentError.new("no resident Metal route for Q/K/V projections")
            end
            encode_qk_rms_rope(
              encoder, q_buf, k_buf,
              q_norm_weight_bufs[layer_index], k_norm_weight_bufs[layer_index],
              positions_buf, token_count, config,
            )
            encode_attention(
              encoder, q_buf, k_buf, v_buf, image_ids_buf, key_valid_buf,
              attended_buf, token_count, config,
            )
            unless Qwen35Metal.encode_matmul_to_buffer(
                     encoder, weights.to_out, attended_buf, projected_buf, token_count
                   )
              raise ArgumentError.new("no resident Metal route for attention output projection")
            end
            encode_residual_layernorm_modulate_gate(
              encoder, current_hidden_buf, projected_buf, gate1_buf, modulation_buf,
              state_buf, norm2_buf, gate2_buf, token_count, dim, config.eps,
            )
            unless Qwen35Metal.encode_matmul_to_buffer(
                     encoder, weights.gate_up, norm2_buf, fused_buf, token_count
                   )
              raise ArgumentError.new("no resident Metal route for gate/up projection")
            end
            encode_swiglu(encoder, fused_buf, activated_buf, token_count, intermediate)
            unless Qwen35Metal.encode_matmul_to_buffer(
                     encoder, weights.mlp_out, activated_buf, mlp_buf, token_count
                   )
              raise ArgumentError.new("no resident Metal route for MLP output projection")
            end
            encode_residual_gate_add(
              encoder, state_buf, gate2_buf, mlp_buf, next_hidden_buf, hidden_count
            )

            previous_hidden_buf = current_hidden_buf
            current_hidden_buf = next_hidden_buf
            next_hidden_buf = previous_hidden_buf
          end
          encoder.end_encoding
          command.commit
          command.wait

          QwenImage21MetalBlockResult.new(
            current_hidden_buf.read(hidden_count),
            QwenImage21MetalBlockStats.new(1, layers.size * 6, 0, 1),
          )
        ensure
          buffers.each(&.release)
        end
      end

      private def self.validate_inputs(
        hidden : Array(Float32), token_count : Int32,
        modulation : Array(Float32),
        positions : Array(StaticArray(Int32, 3)), image_ids : Array(Int32),
        layers : Array(QwenImage21BlockWeights), config : QwenImage21BlockConfig,
        key_valid : Array(Bool)?,
      ) : Nil
        dim = config.hidden_dim
        raise ArgumentError.new("token_count must be positive") unless token_count > 0
        raise ArgumentError.new("hidden size mismatch") unless hidden.size == token_count * dim
        raise ArgumentError.new("modulation size mismatch") unless modulation.size == token_count * 4 * dim
        raise ArgumentError.new("positions size mismatch") unless positions.size == token_count
        raise ArgumentError.new("image_ids size mismatch") unless image_ids.size == token_count
        raise ArgumentError.new("key_valid size mismatch") if key_valid && key_valid.size != token_count
        raise ArgumentError.new("Metal attention supports head_dim <= 256") unless config.head_dim <= 256
        raise ArgumentError.new("layer stack must not be empty") if layers.empty?
        layers.each { |weights| validate_layer_weights(weights, config) }
      end

      private def self.validate_layer_weights(
        weights : QwenImage21BlockWeights, config : QwenImage21BlockConfig,
      ) : Nil
        dim = config.hidden_dim
        {weights.to_q, weights.to_k, weights.to_v, weights.to_out}.each do |weight|
          unless weight.in_dim == dim && weight.out_dim == dim
            raise ArgumentError.new("attention weight shape mismatch")
          end
        end
        unless weights.gate_up.in_dim == dim && weights.gate_up.out_dim == 2 * config.intermediate_dim
          raise ArgumentError.new("gate_up weight shape mismatch")
        end
        unless weights.mlp_out.in_dim == config.intermediate_dim && weights.mlp_out.out_dim == dim
          raise ArgumentError.new("MLP output weight shape mismatch")
        end
        raise ArgumentError.new("norm_q size mismatch") unless weights.norm_q.size == config.head_dim
        raise ArgumentError.new("norm_k size mismatch") unless weights.norm_k.size == config.head_dim
      end

      private def self.validate_attention_rows(image_ids : Array(Int32), key_valid : Array(Bool)) : Nil
        image_ids.each_index do |query|
          valid = image_ids.each_index.any? do |key|
            key_valid[key] && (query >= key || (image_ids[query] >= 0 && image_ids[query] == image_ids[key]))
          end
          raise ArgumentError.new("attention row has no valid keys") unless valid
        end
      end

      private def self.allocate(size : Int64, buffers : Array(ML::MetalBuffer)) : ML::MetalBuffer
        buffer = ML::MetalBuffer.new(size)
        buffers << buffer
        buffer
      end

      private def self.upload_f32(values : Array(Float32), buffers : Array(ML::MetalBuffer)) : ML::MetalBuffer
        buffer = ML::MetalBuffer.from_array(values)
        buffers << buffer
        buffer
      end

      private def self.upload_i32(values : Array(Int32), buffers : Array(ML::MetalBuffer)) : ML::MetalBuffer
        buffer = allocate(values.size.to_i64 * sizeof(Int32), buffers)
        buffer.write_bytes(values.to_unsafe.as(Pointer(UInt8)), values.size * sizeof(Int32))
        buffer
      end

      private def self.upload_u8(values : Array(UInt8), buffers : Array(ML::MetalBuffer)) : ML::MetalBuffer
        buffer = allocate(values.size.to_i64, buffers)
        buffer.write_bytes(values.to_unsafe, values.size)
        buffer
      end

      private def self.pipeline(name : String) : ML::Metal::ComputePipeline
        ML::Metal::PipelineCache.get(name) {
          ML::Metal::ComputePipeline.new(name, SOURCE)
        }
      end

      private def self.encode_layernorm_modulate_gate(
        encoder : ML::Metal::ComputeEncoder,
        input : ML::MetalBuffer, modulation : ML::MetalBuffer,
        normalized : ML::MetalBuffer, gate : ML::MetalBuffer,
        tokens : Int32, dim : Int32, eps : Float32,
      ) : Nil
        encoder.set_pipeline(pipeline("qi21_layernorm_modulate_gate"))
        encoder.set_buffer(input, 0)
        encoder.set_buffer(modulation, 1)
        encoder.set_buffer(normalized, 2, ML::Metal::BufferAccess::Write)
        encoder.set_buffer(gate, 3, ML::Metal::BufferAccess::Write)
        encoder.set_value(tokens.to_u32, 4)
        encoder.set_value(dim.to_u32, 5)
        encoder.set_value(eps, 6)
        encoder.dispatch_threadgroups({tokens, 1, 1}, {256, 1, 1})
      end

      private def self.encode_qk_rms_rope(
        encoder : ML::Metal::ComputeEncoder,
        q : ML::MetalBuffer, k : ML::MetalBuffer,
        q_weight : ML::MetalBuffer, k_weight : ML::MetalBuffer,
        positions : ML::MetalBuffer, tokens : Int32,
        config : QwenImage21BlockConfig,
      ) : Nil
        threads = head_threads(config.head_dim)
        encoder.set_pipeline(pipeline("qi21_qk_rms_rope"))
        encoder.set_buffer(q, 0, ML::Metal::BufferAccess::ReadWrite)
        encoder.set_buffer(k, 1, ML::Metal::BufferAccess::ReadWrite)
        encoder.set_buffer(q_weight, 2)
        encoder.set_buffer(k_weight, 3)
        encoder.set_buffer(positions, 4)
        encoder.set_value(tokens.to_u32, 5)
        encoder.set_value(config.heads.to_u32, 6)
        encoder.set_value(config.head_dim.to_u32, 7)
        encoder.set_value(config.axes_dims[0].to_u32, 8)
        encoder.set_value(config.axes_dims[1].to_u32, 9)
        encoder.set_value(config.axes_dims[2].to_u32, 10)
        encoder.set_value(config.eps, 11)
        encoder.set_value(config.rope_theta, 12)
        encoder.set_threadgroup_memory(2 * config.head_dim * sizeof(Float32), 0)
        encoder.dispatch_threadgroups({tokens * config.heads, 1, 1}, {threads, 1, 1})
      end

      private def self.encode_attention(
        encoder : ML::Metal::ComputeEncoder,
        q : ML::MetalBuffer, k : ML::MetalBuffer, v : ML::MetalBuffer,
        image_ids : ML::MetalBuffer, key_valid : ML::MetalBuffer,
        output : ML::MetalBuffer, tokens : Int32,
        config : QwenImage21BlockConfig,
      ) : Nil
        threads = head_threads(config.head_dim)
        encoder.set_pipeline(pipeline("qi21_block_causal_attention"))
        encoder.set_buffer(q, 0)
        encoder.set_buffer(k, 1)
        encoder.set_buffer(v, 2)
        encoder.set_buffer(image_ids, 3)
        encoder.set_buffer(key_valid, 4)
        encoder.set_buffer(output, 5, ML::Metal::BufferAccess::Write)
        encoder.set_value(tokens.to_u32, 6)
        encoder.set_value(config.heads.to_u32, 7)
        encoder.set_value(config.head_dim.to_u32, 8)
        encoder.set_value((1.0_f64 / Math.sqrt(config.head_dim)).to_f32, 9)
        encoder.dispatch_threadgroups({tokens * config.heads, 1, 1}, {threads, 1, 1})
      end

      private def self.encode_residual_layernorm_modulate_gate(
        encoder : ML::Metal::ComputeEncoder,
        hidden : ML::MetalBuffer, projected : ML::MetalBuffer,
        gate1 : ML::MetalBuffer, modulation : ML::MetalBuffer,
        state : ML::MetalBuffer, normalized : ML::MetalBuffer,
        gate2 : ML::MetalBuffer, tokens : Int32, dim : Int32, eps : Float32,
      ) : Nil
        encoder.set_pipeline(pipeline("qi21_residual_layernorm_modulate_gate"))
        encoder.set_buffer(hidden, 0)
        encoder.set_buffer(projected, 1)
        encoder.set_buffer(gate1, 2)
        encoder.set_buffer(modulation, 3)
        encoder.set_buffer(state, 4, ML::Metal::BufferAccess::Write)
        encoder.set_buffer(normalized, 5, ML::Metal::BufferAccess::Write)
        encoder.set_buffer(gate2, 6, ML::Metal::BufferAccess::Write)
        encoder.set_value(tokens.to_u32, 7)
        encoder.set_value(dim.to_u32, 8)
        encoder.set_value(eps, 9)
        encoder.dispatch_threadgroups({tokens, 1, 1}, {256, 1, 1})
      end

      private def self.encode_swiglu(
        encoder : ML::Metal::ComputeEncoder,
        fused : ML::MetalBuffer, output : ML::MetalBuffer,
        tokens : Int32, intermediate : Int32,
      ) : Nil
        encoder.set_pipeline(pipeline("qi21_swiglu"))
        encoder.set_buffer(fused, 0)
        encoder.set_buffer(output, 1, ML::Metal::BufferAccess::Write)
        encoder.set_value(tokens.to_u32, 2)
        encoder.set_value(intermediate.to_u32, 3)
        encoder.dispatch_1d(tokens * intermediate, 256)
      end

      private def self.encode_residual_gate_add(
        encoder : ML::Metal::ComputeEncoder,
        state : ML::MetalBuffer, gate : ML::MetalBuffer,
        projected : ML::MetalBuffer, output : ML::MetalBuffer,
        count : Int32,
      ) : Nil
        encoder.set_pipeline(pipeline("qi21_residual_gate_add"))
        encoder.set_buffer(state, 0)
        encoder.set_buffer(gate, 1)
        encoder.set_buffer(projected, 2)
        encoder.set_buffer(output, 3, ML::Metal::BufferAccess::Write)
        encoder.set_value(count.to_u32, 4)
        encoder.dispatch_1d(count, 256)
      end

      private def self.head_threads(head_dim : Int32) : Int32
        threads = 32
        while threads < head_dim
          threads *= 2
        end
        threads
      end
    {% end %}
  end

  class QwenImage21MetalLayerStackBackend
    include QwenImage21LayerStackBackend

    getter last_stats : QwenImage21MetalBlockStats?
    getter invocations = 0

    def initialize
      @last_stats = nil
    end

    def self.available? : Bool
      QwenImage21MetalBlock.available?
    end

    def forward_layers(
      hidden : Array(Float32), token_count : Int32,
      modulation : Array(Float32),
      positions : Array(StaticArray(Int32, 3)),
      image_ids : Array(Int32),
      layers : Array(QwenImage21BlockWeights),
      config : QwenImage21BlockConfig,
      key_valid : Array(Bool)?,
    ) : Array(Float32)
      result = QwenImage21MetalBlock.forward_layers(
        hidden, token_count, modulation, positions, image_ids,
        layers, config, key_valid,
      )
      @last_stats = result.stats
      @invocations += 1
      result.hidden
    end
  end
end
