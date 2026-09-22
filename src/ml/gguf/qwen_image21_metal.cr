require "./qwen_image21_transformer"
require "./qwen35_metal"

# Hybrid reference backend for Qwen-Image 2.1 block admission.
#
# Projection matmuls use the established Qwen 3.5 Metal K-quant kernels while
# normalization, RoPE, attention, and elementwise math remain on the CPU. This
# deliberately narrow boundary supports exact one-block CPU/Metal parity before
# the full block is fused or made resident on-device.
module ML::GGUF
  module QwenImage21MetalBF16
    {% if flag?(:cpu_only) %}
      def self.matmul(qw : QuantWeight, input : Array(Float32), rows : Int32) : Array(Float32)?
        nil
      end

      def self.project_text_layers(
        input : Array(Float32), rows : Int32,
        input_weight : QuantWeight, output_weight : QuantWeight,
      ) : Array(Float32)?
        nil
      end

      def self.project_timestep_layers(
        input : Array(Float32), rows : Int32,
        first_weight : QuantWeight, second_weight : QuantWeight,
        modulation_weight : QuantWeight, scale_weight : QuantWeight,
      ) : {Array(Float32), Array(Float32)}?
        nil
      end

      def self.encode_matmul_to_buffer(
        encoder : ML::Metal::ComputeEncoder, qw : QuantWeight,
        input_buf : ML::MetalBuffer, output_buf : ML::MetalBuffer, rows : Int32,
      ) : Bool
        false
      end

      def self.encode_silu_to_buffer(
        encoder : ML::Metal::ComputeEncoder, buffer : ML::MetalBuffer, count : Int32,
      ) : Nil
        raise "Metal disabled (cpu_only)"
      end
    {% else %}
      SOURCE = {{ read_file("#{__DIR__}/kernels/qwen_image21.metal") }}

      def self.matmul(qw : QuantWeight, input : Array(Float32), rows : Int32) : Array(Float32)?
        return nil unless qw.type.bf16? && rows > 0
        validate_input(input, rows, qw.in_dim)
        validate_weight(qw)
        ML::Metal::Device.init!
        buffers = [] of ML::MetalBuffer
        begin
          input_buf = upload(input, buffers)
          output_buf = allocate(rows.to_i64 * qw.out_dim * sizeof(Float32), buffers)
          command = ML::Metal::CommandBuffer.new
          encoder = ML::Metal::ComputeEncoder.new(command)
          encode_matmul(encoder, qw, input_buf, output_buf, rows)
          encoder.end_encoding
          command.commit
          command.wait
          output_buf.read(rows * qw.out_dim)
        ensure
          buffers.each(&.release)
        end
      end

      def self.project_text_layers(
        input : Array(Float32), rows : Int32,
        input_weight : QuantWeight, output_weight : QuantWeight,
      ) : Array(Float32)?
        return nil unless bf16_weights?(input_weight, output_weight)
        return [] of Float32 if rows == 0
        validate_input(input, rows, input_weight.in_dim)
        validate_chain(input_weight, output_weight)
        ML::Metal::Device.init!
        buffers = [] of ML::MetalBuffer
        begin
          input_buf = upload(input, buffers)
          hidden_buf = allocate(rows.to_i64 * input_weight.out_dim * sizeof(Float32), buffers)
          output_buf = allocate(rows.to_i64 * output_weight.out_dim * sizeof(Float32), buffers)
          command = ML::Metal::CommandBuffer.new
          encoder = ML::Metal::ComputeEncoder.new(command)
          encode_matmul(encoder, input_weight, input_buf, hidden_buf, rows)
          encode_activation(encoder, "qi21_gelu_inplace", hidden_buf, rows * input_weight.out_dim)
          encode_matmul(encoder, output_weight, hidden_buf, output_buf, rows)
          encoder.end_encoding
          command.commit
          command.wait
          output_buf.read(rows * output_weight.out_dim)
        ensure
          buffers.each(&.release)
        end
      end

      def self.project_timestep_layers(
        input : Array(Float32), rows : Int32,
        first_weight : QuantWeight, second_weight : QuantWeight,
        modulation_weight : QuantWeight, scale_weight : QuantWeight,
      ) : {Array(Float32), Array(Float32)}?
        return nil unless bf16_weights?(
                            first_weight, second_weight, modulation_weight, scale_weight
                          )
        return {[] of Float32, [] of Float32} if rows == 0
        validate_input(input, rows, first_weight.in_dim)
        validate_chain(first_weight, second_weight)
        unless second_weight.out_dim == modulation_weight.in_dim &&
               second_weight.out_dim == scale_weight.in_dim
          raise ArgumentError.new("BF16 timestep projection chain shape mismatch")
        end
        validate_weight(modulation_weight)
        validate_weight(scale_weight)
        ML::Metal::Device.init!
        buffers = [] of ML::MetalBuffer
        begin
          input_buf = upload(input, buffers)
          first_buf = allocate(rows.to_i64 * first_weight.out_dim * sizeof(Float32), buffers)
          second_buf = allocate(rows.to_i64 * second_weight.out_dim * sizeof(Float32), buffers)
          modulation_buf = allocate(rows.to_i64 * modulation_weight.out_dim * sizeof(Float32), buffers)
          scale_buf = allocate(rows.to_i64 * scale_weight.out_dim * sizeof(Float32), buffers)
          command = ML::Metal::CommandBuffer.new
          encoder = ML::Metal::ComputeEncoder.new(command)
          encode_matmul(encoder, first_weight, input_buf, first_buf, rows)
          encode_activation(encoder, "qi21_silu_inplace", first_buf, rows * first_weight.out_dim)
          encode_matmul(encoder, second_weight, first_buf, second_buf, rows)
          encode_activation(encoder, "qi21_silu_inplace", second_buf, rows * second_weight.out_dim)
          encode_matmul(encoder, modulation_weight, second_buf, modulation_buf, rows)
          encode_matmul(encoder, scale_weight, second_buf, scale_buf, rows)
          encoder.end_encoding
          command.commit
          command.wait
          {
            modulation_buf.read(rows * modulation_weight.out_dim),
            scale_buf.read(rows * scale_weight.out_dim),
          }
        ensure
          buffers.each(&.release)
        end
      end

      # Encode a BF16 batch projection into an existing command buffer. This is
      # the buffer-level contract used by resident Qwen-Image execution paths;
      # unsupported weights return false without committing partial work.
      def self.encode_matmul_to_buffer(
        encoder : ML::Metal::ComputeEncoder, qw : QuantWeight,
        input_buf : ML::MetalBuffer, output_buf : ML::MetalBuffer, rows : Int32,
      ) : Bool
        return false unless qw.type.bf16? && rows > 0
        validate_weight(qw)
        encode_matmul(encoder, qw, input_buf, output_buf, rows)
        true
      end

      def self.encode_silu_to_buffer(
        encoder : ML::Metal::ComputeEncoder, buffer : ML::MetalBuffer, count : Int32,
      ) : Nil
        encode_activation(encoder, "qi21_silu_inplace", buffer, count)
      end

      private def self.encode_matmul(
        encoder : ML::Metal::ComputeEncoder, qw : QuantWeight,
        input_buf : ML::MetalBuffer, output_buf : ML::MetalBuffer, rows : Int32,
      ) : Nil
        weight_buf, weight_offset = Qwen35Metal.weight_buffer_slot(qw)
        encoder.set_pipeline(matmul_pipeline)
        encoder.set_buffer(weight_buf, 0, offset: weight_offset)
        encoder.set_buffer(input_buf, 1)
        encoder.set_buffer(output_buf, 2, ML::Metal::BufferAccess::Write)
        encoder.set_value(qw.in_dim.to_u32, 3)
        encoder.set_value(qw.out_dim.to_u32, 4)
        encoder.set_value(rows.to_u32, 5)
        output_rows = rows * qw.out_dim
        encoder.dispatch_threadgroups({(output_rows + 1) // 2, 1, 1}, {64, 1, 1})
      end

      private def self.encode_activation(
        encoder : ML::Metal::ComputeEncoder, name : String,
        buffer : ML::MetalBuffer, count : Int32,
      ) : Nil
        encoder.set_pipeline(activation_pipeline(name))
        encoder.set_buffer(buffer, 0, ML::Metal::BufferAccess::ReadWrite)
        encoder.set_value(count.to_u32, 1)
        encoder.dispatch_1d(count, 256)
      end

      private def self.bf16_weights?(*weights : QuantWeight) : Bool
        weights.all?(&.type.bf16?)
      end

      private def self.validate_input(input : Array(Float32), rows : Int32, in_dim : Int32) : Nil
        unless input.size == rows * in_dim
          raise ArgumentError.new("BF16 projection input size mismatch")
        end
      end

      private def self.validate_chain(first : QuantWeight, second : QuantWeight) : Nil
        validate_weight(first)
        validate_weight(second)
        unless first.out_dim == second.in_dim
          raise ArgumentError.new("BF16 projection chain shape mismatch")
        end
      end

      private def self.validate_weight(qw : QuantWeight) : Nil
        expected_bytes = qw.out_dim.to_i64 * qw.in_dim * 2_i64
        unless qw.type.bf16? && qw.raw.size.to_i64 == expected_bytes
          raise ArgumentError.new("BF16 projection weight size mismatch")
        end
      end

      private def self.upload(values : Array(Float32), buffers : Array(ML::MetalBuffer)) : ML::MetalBuffer
        buffer = allocate(values.size.to_i64 * sizeof(Float32), buffers)
        buffer.write(values)
        buffer
      end

      private def self.allocate(bytes : Int64, buffers : Array(ML::MetalBuffer)) : ML::MetalBuffer
        buffer = ML::MetalBuffer.new(bytes)
        buffers << buffer
        buffer
      end

      private def self.matmul_pipeline : ML::Metal::ComputePipeline
        ML::Metal::PipelineCache.get("qi21_bf16_batch_matmul") do
          ML::Metal::ComputePipeline.new("qi21_bf16_batch_matmul", SOURCE)
        end
      end

      private def self.activation_pipeline(name : String) : ML::Metal::ComputePipeline
        ML::Metal::PipelineCache.get(name) do
          ML::Metal::ComputePipeline.new(name, SOURCE)
        end
      end
    {% end %}
  end

  # The Qwen 3.5 host-array matmul may switch Q5/Q6 batches to a separate GEMM
  # kernel. Qwen-Image's resident stack uses the buffer-level GEMV encoder, so
  # its hybrid reference must use that same numerical route at every batch size.
  module QwenImage21MetalQuantized
    {% if flag?(:cpu_only) %}
      def self.matmul(qw : QuantWeight, input : Array(Float32), rows : Int32) : Array(Float32)?
        nil
      end
    {% else %}
      def self.matmul(qw : QuantWeight, input : Array(Float32), rows : Int32) : Array(Float32)?
        return nil unless rows > 0
        input_buf = ML::MetalBuffer.from_array(input)
        output_buf = ML::MetalBuffer.new(rows.to_i64 * qw.out_dim * sizeof(Float32))
        begin
          return nil unless Qwen35Metal.matmul_to_buffer(qw, input_buf, output_buf, rows)
          output_buf.read(rows * qw.out_dim)
        ensure
          input_buf.release
          output_buf.release
        end
      end
    {% end %}
  end

  class QwenImage21MetalProjectionBackend
    include ComputeBackend
    include QwenImage21FusedProjectionBackend
    include QwenImage21ResidentInputProjectionBackend

    getter metal_projection_count = 0
    getter bf16_projection_count = 0
    getter fused_outer_command_count = 0

    def initialize(@strict : Bool = true, @resident_input : Bool = true)
      @cpu = F32Backend.new
    end

    def resident_input? : Bool
      @resident_input
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
      unless x.size == rows * qw.in_dim
        raise ArgumentError.new("projection input size mismatch")
      end
      return [] of Float32 if rows == 0

      {% if flag?(:cpu_only) %}
        if @strict
          raise ArgumentError.new("Metal disabled (cpu_only)")
        else
          @cpu.matmul(x, rows, qw, bias)
        end
      {% else %}
        result = if qw.type.bf16?
                   QwenImage21MetalBF16.matmul(qw, x, rows)
                 else
                   QwenImage21MetalQuantized.matmul(qw, x, rows)
                 end
        if result
          @metal_projection_count += 1
          @bf16_projection_count += 1 if qw.type.bf16?
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

    def project_text_layers(
      input : Array(Float32), rows : Int32,
      input_weight : QuantWeight, output_weight : QuantWeight,
    ) : Array(Float32)?
      return [] of Float32 if rows == 0
      result = QwenImage21MetalBF16.project_text_layers(
        input, rows, input_weight, output_weight
      )
      if result
        @metal_projection_count += 2
        @bf16_projection_count += 2
        @fused_outer_command_count += 1
      end
      result
    end

    def project_timestep_layers(
      input : Array(Float32), rows : Int32,
      first_weight : QuantWeight, second_weight : QuantWeight,
      modulation_weight : QuantWeight, scale_weight : QuantWeight,
    ) : {Array(Float32), Array(Float32)}?
      result = QwenImage21MetalBF16.project_timestep_layers(
        input, rows, first_weight, second_weight, modulation_weight, scale_weight
      )
      if result
        @metal_projection_count += 4
        @bf16_projection_count += 4
        @fused_outer_command_count += 1
      end
      result
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
    getter active_tokens : Int32

    def initialize(@command_buffers, @projection_dispatches,
                   @intermediate_readbacks, @final_readbacks, @active_tokens)
    end
  end

  class QwenImage21MetalBlockResult
    getter hidden : Array(Float32)
    getter stats : QwenImage21MetalBlockStats

    def initialize(@hidden, @stats)
    end
  end

  class QwenImage21MetalProjectedResult
    getter output : Array(Float32)
    getter stats : QwenImage21MetalBlockStats

    def initialize(@output, @stats)
    end
  end

  class QwenImage21MetalResidentInput
    getter image_input : Array(Float32)
    getter projected_text : Array(Float32)
    getter time_input : Array(Float32)
    getter source_rows : Array(Int32)
    getter target_mask : Array(Bool)
    getter image_weight : QuantWeight
    getter timestep_linear_1 : QuantWeight
    getter timestep_linear_2 : QuantWeight
    getter modulation_weight : QuantWeight
    getter scale_weight : QuantWeight

    def initialize(@image_input, @projected_text, @time_input, @source_rows,
                   @target_mask, @image_weight, @timestep_linear_1,
                   @timestep_linear_2, @modulation_weight, @scale_weight)
    end

    def image_rows : Int32
      @image_input.size // @image_weight.in_dim
    end

    def time_rows : Int32
      @time_input.size // @timestep_linear_1.in_dim
    end
  end

  class QwenImage21MetalPrefixCache
    getter prefix_tokens : Int32
    getter total_tokens : Int32
    getter k_buffers : Array(ML::MetalBuffer)
    getter v_buffers : Array(ML::MetalBuffer)
    getter prefix_output_buffer : ML::MetalBuffer
    property prefix_output : Array(Float32)

    @closed = false

    def initialize(
      hidden : Array(Float32), modulation : Array(Float32),
      positions : Array(StaticArray(Int32, 3)), image_ids : Array(Int32),
      key_valid : Array(Bool), layers : Array(QwenImage21BlockWeights),
      config : QwenImage21BlockConfig, @prefix_tokens : Int32,
      @total_tokens : Int32,
    )
      @dim = config.hidden_dim
      @heads = config.heads
      @head_dim = config.head_dim
      @intermediate_dim = config.intermediate_dim
      @axes_dims = config.axes_dims
      @eps = config.eps
      @rope_theta = config.rope_theta
      @layer_refs = layers.dup
      @prefix_hidden = hidden.first(@prefix_tokens * @dim)
      @prefix_modulation = modulation.first(@prefix_tokens * 4 * @dim)
      @prefix_positions = positions.first(@prefix_tokens)
      @prefix_image_ids = image_ids.first(@prefix_tokens)
      @prefix_key_valid = key_valid.first(@prefix_tokens)
      @prefix_output = Array(Float32).new(@prefix_tokens * @dim, 0.0_f32)
      @k_buffers = [] of ML::MetalBuffer
      @v_buffers = [] of ML::MetalBuffer
      bytes = @prefix_tokens.to_i64 * @dim * sizeof(Float32)
      @prefix_output_buffer = ML::MetalBuffer.new(bytes)
      begin
        layers.size.times do
          @k_buffers << ML::MetalBuffer.new(bytes)
          @v_buffers << ML::MetalBuffer.new(bytes)
        end
      rescue ex
        close
        raise ex
      end
    end

    def compatible?(
      hidden : Array(Float32), token_count : Int32,
      modulation : Array(Float32),
      positions : Array(StaticArray(Int32, 3)), image_ids : Array(Int32),
      key_valid : Array(Bool), layers : Array(QwenImage21BlockWeights),
      config : QwenImage21BlockConfig, target_start : Int32,
    ) : Bool
      return false if @closed
      return false unless target_start == @prefix_tokens && token_count == @total_tokens
      return false unless config.hidden_dim == @dim && config.heads == @heads
      return false unless config.head_dim == @head_dim && config.intermediate_dim == @intermediate_dim
      return false unless config.axes_dims == @axes_dims && config.eps == @eps && config.rope_theta == @rope_theta
      return false unless layers.size == @layer_refs.size
      return false unless layers.each_with_index.all? { |layer, index| layer.same?(@layer_refs[index]) }
      return false unless prefix_matches?(hidden, @prefix_hidden)
      return false unless prefix_matches?(modulation, @prefix_modulation)
      return false unless positions.first(@prefix_tokens) == @prefix_positions
      return false unless image_ids.first(@prefix_tokens) == @prefix_image_ids
      key_valid.first(@prefix_tokens) == @prefix_key_valid
    end

    def close : Nil
      return if @closed
      @closed = true
      @k_buffers.each(&.release)
      @v_buffers.each(&.release)
      @prefix_output_buffer.release
      @k_buffers.clear
      @v_buffers.clear
    end

    def finalize
      close
    rescue
      nil
    end

    private def prefix_matches?(values : Array(Float32), expected : Array(Float32)) : Bool
      return false if values.size < expected.size
      expected.each_with_index.all? { |value, index| values[index] == value }
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

      def self.forward_layers_projected(
        hidden : Array(Float32), token_count : Int32,
        modulation : Array(Float32),
        positions : Array(StaticArray(Int32, 3)),
        image_ids : Array(Int32),
        layers : Array(QwenImage21BlockWeights),
        config : QwenImage21BlockConfig,
        scales : Array(Float32), output_weight : QuantWeight,
        key_valid : Array(Bool)? = nil,
      ) : QwenImage21MetalProjectedResult
        raise "Metal disabled (cpu_only)"
      end

      def self.forward_layers_resident(
        input : QwenImage21MetalResidentInput, token_count : Int32,
        positions : Array(StaticArray(Int32, 3)), image_ids : Array(Int32),
        layers : Array(QwenImage21BlockWeights), config : QwenImage21BlockConfig,
        output_weight : QuantWeight, key_valid : Array(Bool)?,
      ) : QwenImage21MetalProjectedResult
        raise "Metal disabled (cpu_only)"
      end

      def self.forward_layers_capture_prefix(
        hidden : Array(Float32), token_count : Int32,
        modulation : Array(Float32),
        positions : Array(StaticArray(Int32, 3)),
        image_ids : Array(Int32),
        layers : Array(QwenImage21BlockWeights),
        config : QwenImage21BlockConfig,
        key_valid : Array(Bool)?, prefix_tokens : Int32,
      ) : Tuple(QwenImage21MetalBlockResult, QwenImage21MetalPrefixCache)
        raise "Metal disabled (cpu_only)"
      end

      def self.forward_layers_projected_capture_prefix(
        hidden : Array(Float32), token_count : Int32,
        modulation : Array(Float32),
        positions : Array(StaticArray(Int32, 3)),
        image_ids : Array(Int32),
        layers : Array(QwenImage21BlockWeights),
        config : QwenImage21BlockConfig,
        scales : Array(Float32), output_weight : QuantWeight,
        key_valid : Array(Bool)?, prefix_tokens : Int32,
      ) : Tuple(QwenImage21MetalProjectedResult, QwenImage21MetalPrefixCache)
        raise "Metal disabled (cpu_only)"
      end

      def self.forward_layers_from_prefix_cache(
        hidden : Array(Float32), token_count : Int32,
        modulation : Array(Float32),
        positions : Array(StaticArray(Int32, 3)),
        image_ids : Array(Int32),
        layers : Array(QwenImage21BlockWeights),
        config : QwenImage21BlockConfig,
        key_valid : Array(Bool)?, cache : QwenImage21MetalPrefixCache,
      ) : QwenImage21MetalBlockResult
        raise "Metal disabled (cpu_only)"
      end

      def self.forward_layers_projected_from_prefix_cache(
        hidden : Array(Float32), token_count : Int32,
        modulation : Array(Float32),
        positions : Array(StaticArray(Int32, 3)),
        image_ids : Array(Int32),
        layers : Array(QwenImage21BlockWeights),
        config : QwenImage21BlockConfig,
        key_valid : Array(Bool)?, cache : QwenImage21MetalPrefixCache,
        scales : Array(Float32), output_weight : QuantWeight,
      ) : QwenImage21MetalProjectedResult
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
        forward_layers_impl(
          hidden, token_count, modulation, positions, image_ids,
          layers, config, key_valid, nil, nil, nil,
        )
      end

      def self.forward_layers_projected(
        hidden : Array(Float32), token_count : Int32,
        modulation : Array(Float32),
        positions : Array(StaticArray(Int32, 3)),
        image_ids : Array(Int32),
        layers : Array(QwenImage21BlockWeights),
        config : QwenImage21BlockConfig,
        scales : Array(Float32), output_weight : QuantWeight,
        key_valid : Array(Bool)? = nil,
      ) : QwenImage21MetalProjectedResult
        result = forward_layers_impl(
          hidden, token_count, modulation, positions, image_ids,
          layers, config, key_valid, nil, scales, output_weight,
        )
        QwenImage21MetalProjectedResult.new(result.hidden, result.stats)
      end

      def self.forward_layers_resident(
        input : QwenImage21MetalResidentInput, token_count : Int32,
        positions : Array(StaticArray(Int32, 3)), image_ids : Array(Int32),
        layers : Array(QwenImage21BlockWeights), config : QwenImage21BlockConfig,
        output_weight : QuantWeight, key_valid : Array(Bool)?,
      ) : QwenImage21MetalProjectedResult
        result = forward_layers_impl(
          nil, token_count, nil, positions, image_ids,
          layers, config, key_valid, nil, nil, output_weight, input,
        )
        QwenImage21MetalProjectedResult.new(result.hidden, result.stats)
      end

      def self.forward_layers_capture_prefix(
        hidden : Array(Float32), token_count : Int32,
        modulation : Array(Float32),
        positions : Array(StaticArray(Int32, 3)),
        image_ids : Array(Int32),
        layers : Array(QwenImage21BlockWeights),
        config : QwenImage21BlockConfig,
        key_valid : Array(Bool)? = nil, prefix_tokens : Int32 = 0,
      ) : Tuple(QwenImage21MetalBlockResult, QwenImage21MetalPrefixCache)
        unless prefix_tokens > 0 && prefix_tokens < token_count
          raise ArgumentError.new("prefix_tokens must split a non-empty prefix and target")
        end
        valid = key_valid || Array(Bool).new(token_count, true)
        cache = QwenImage21MetalPrefixCache.new(
          hidden, modulation, positions, image_ids, valid, layers, config,
          prefix_tokens, token_count,
        )
        begin
          result = forward_layers_impl(
            hidden, token_count, modulation, positions, image_ids,
            layers, config, valid, cache, nil, nil,
          )
          cache.prefix_output = result.hidden.first(prefix_tokens * config.hidden_dim)
          {result, cache}
        rescue ex
          cache.close
          raise ex
        end
      end

      def self.forward_layers_projected_capture_prefix(
        hidden : Array(Float32), token_count : Int32,
        modulation : Array(Float32),
        positions : Array(StaticArray(Int32, 3)),
        image_ids : Array(Int32),
        layers : Array(QwenImage21BlockWeights),
        config : QwenImage21BlockConfig,
        scales : Array(Float32), output_weight : QuantWeight,
        key_valid : Array(Bool)? = nil, prefix_tokens : Int32 = 0,
      ) : Tuple(QwenImage21MetalProjectedResult, QwenImage21MetalPrefixCache)
        unless prefix_tokens > 0 && prefix_tokens < token_count
          raise ArgumentError.new("prefix_tokens must split a non-empty prefix and target")
        end
        valid = key_valid || Array(Bool).new(token_count, true)
        cache = QwenImage21MetalPrefixCache.new(
          hidden, modulation, positions, image_ids, valid, layers, config,
          prefix_tokens, token_count,
        )
        begin
          result = forward_layers_impl(
            hidden, token_count, modulation, positions, image_ids,
            layers, config, valid, cache, scales, output_weight,
          )
          {
            QwenImage21MetalProjectedResult.new(result.hidden, result.stats),
            cache,
          }
        rescue ex
          cache.close
          raise ex
        end
      end

      def self.forward_layers_from_prefix_cache(
        hidden : Array(Float32), token_count : Int32,
        modulation : Array(Float32),
        positions : Array(StaticArray(Int32, 3)),
        image_ids : Array(Int32),
        layers : Array(QwenImage21BlockWeights),
        config : QwenImage21BlockConfig,
        key_valid : Array(Bool)?, cache : QwenImage21MetalPrefixCache,
      ) : QwenImage21MetalBlockResult
        forward_layers_from_prefix_cache_impl(
          hidden, token_count, modulation, positions, image_ids,
          layers, config, key_valid, cache, nil, nil,
        )
      end

      def self.forward_layers_projected_from_prefix_cache(
        hidden : Array(Float32), token_count : Int32,
        modulation : Array(Float32),
        positions : Array(StaticArray(Int32, 3)),
        image_ids : Array(Int32),
        layers : Array(QwenImage21BlockWeights),
        config : QwenImage21BlockConfig,
        key_valid : Array(Bool)?, cache : QwenImage21MetalPrefixCache,
        scales : Array(Float32), output_weight : QuantWeight,
      ) : QwenImage21MetalProjectedResult
        result = forward_layers_from_prefix_cache_impl(
          hidden, token_count, modulation, positions, image_ids,
          layers, config, key_valid, cache, scales, output_weight,
        )
        QwenImage21MetalProjectedResult.new(result.hidden, result.stats)
      end

      private def self.forward_layers_from_prefix_cache_impl(
        hidden : Array(Float32), token_count : Int32,
        modulation : Array(Float32),
        positions : Array(StaticArray(Int32, 3)),
        image_ids : Array(Int32),
        layers : Array(QwenImage21BlockWeights),
        config : QwenImage21BlockConfig,
        key_valid : Array(Bool)?, cache : QwenImage21MetalPrefixCache,
        final_scales : Array(Float32)?, output_weight : QuantWeight?,
      ) : QwenImage21MetalBlockResult
        validate_inputs(hidden, token_count, modulation, positions, image_ids,
          layers, config, key_valid)
        validate_final_head(final_scales, output_weight, token_count, config.hidden_dim)
        valid = key_valid || Array(Bool).new(token_count, true)
        unless cache.compatible?(
                 hidden, token_count, modulation, positions, image_ids,
                 valid, layers, config, cache.prefix_tokens,
               )
          raise ArgumentError.new("prefix cache input mismatch")
        end
        unless cache.total_tokens == token_count && cache.k_buffers.size == layers.size && cache.v_buffers.size == layers.size
          raise ArgumentError.new("prefix cache shape mismatch")
        end
        prefix_tokens = cache.prefix_tokens
        unless prefix_tokens > 0 && prefix_tokens < token_count
          raise ArgumentError.new("prefix cache must split a non-empty prefix and target")
        end
        validate_attention_rows(image_ids, valid)
        ML::Metal::Device.init!

        dim = config.hidden_dim
        intermediate = config.intermediate_dim
        active_tokens = token_count - prefix_tokens
        active_count = active_tokens * dim
        active_bytes = active_count.to_i64 * sizeof(Float32)
        full_hidden_bytes = token_count.to_i64 * dim * sizeof(Float32)
        active_hidden = hidden[prefix_tokens * dim, active_count]
        active_modulation = modulation[prefix_tokens * 4 * dim, active_tokens * 4 * dim]
        active_positions = positions[prefix_tokens, active_tokens]
        buffers = [] of ML::MetalBuffer

        begin
          current_hidden_buf = upload_f32(active_hidden, buffers)
          next_hidden_buf = allocate(active_bytes, buffers)
          modulation_buf = upload_f32(active_modulation, buffers)
          positions_buf = upload_i32(active_positions.flat_map(&.to_a), buffers)
          image_ids_buf = upload_i32(image_ids, buffers)
          key_valid_buf = upload_u8(valid.map { |value| value ? 1_u8 : 0_u8 }, buffers)
          q_norm_weight_bufs = layers.map { |weights| upload_f32(weights.norm_q, buffers) }
          k_norm_weight_bufs = layers.map { |weights| upload_f32(weights.norm_k, buffers) }

          norm1_buf = allocate(active_bytes, buffers)
          gate1_buf = allocate(active_bytes, buffers)
          q_buf = allocate(active_bytes, buffers)
          active_k_buf = allocate(active_bytes, buffers)
          active_v_buf = allocate(active_bytes, buffers)
          full_k_buf = allocate(full_hidden_bytes, buffers)
          full_v_buf = allocate(full_hidden_bytes, buffers)
          attended_buf = allocate(active_bytes, buffers)
          projected_buf = allocate(active_bytes, buffers)
          state_buf = allocate(active_bytes, buffers)
          norm2_buf = allocate(active_bytes, buffers)
          gate2_buf = allocate(active_bytes, buffers)
          fused_buf = allocate(active_tokens.to_i64 * 2_i64 * intermediate * sizeof(Float32), buffers)
          activated_buf = allocate(active_tokens.to_i64 * intermediate * sizeof(Float32), buffers)
          mlp_buf = allocate(active_bytes, buffers)
          final_hidden_buf = final_scales ? allocate(full_hidden_bytes, buffers) : nil
          final_scales_buf = final_scales.try { |values| upload_f32(values, buffers) }
          final_norm_buf = final_scales ? allocate(full_hidden_bytes, buffers) : nil
          final_output_buf = if weight = output_weight
                               allocate(token_count.to_i64 * weight.out_dim * sizeof(Float32), buffers)
                             end

          command = ML::Metal::CommandBuffer.new
          encoder = ML::Metal::ComputeEncoder.new(command)
          prefix_values = prefix_tokens * dim
          layers.each_with_index do |weights, layer_index|
            encode_layernorm_modulate_gate(
              encoder, current_hidden_buf, modulation_buf, norm1_buf, gate1_buf,
              active_tokens, dim, config.eps,
            )
            unless Qwen35Metal.encode_matmul_many_to_buffers(
                     encoder,
                     [weights.to_q, weights.to_k, weights.to_v],
                     norm1_buf,
                     [q_buf, active_k_buf, active_v_buf],
                     active_tokens,
                   )
              raise ArgumentError.new("no resident Metal route for Q/K/V projections")
            end
            encode_qk_rms_rope(
              encoder, q_buf, active_k_buf,
              q_norm_weight_bufs[layer_index], k_norm_weight_bufs[layer_index],
              positions_buf, active_tokens, config,
            )
            encode_copy_f32(encoder, cache.k_buffers[layer_index], full_k_buf, prefix_values)
            encode_copy_f32(encoder, cache.v_buffers[layer_index], full_v_buf, prefix_values)
            encode_copy_f32(
              encoder, active_k_buf, full_k_buf, active_count,
              destination_offset: prefix_values,
            )
            encode_copy_f32(
              encoder, active_v_buf, full_v_buf, active_count,
              destination_offset: prefix_values,
            )
            encode_attention(
              encoder, q_buf, full_k_buf, full_v_buf, image_ids_buf, key_valid_buf,
              attended_buf, token_count, active_tokens, prefix_tokens, config,
            )
            unless Qwen35Metal.encode_matmul_to_buffer(
                     encoder, weights.to_out, attended_buf, projected_buf, active_tokens
                   )
              raise ArgumentError.new("no resident Metal route for attention output projection")
            end
            encode_residual_layernorm_modulate_gate(
              encoder, current_hidden_buf, projected_buf, gate1_buf, modulation_buf,
              state_buf, norm2_buf, gate2_buf, active_tokens, dim, config.eps,
            )
            unless Qwen35Metal.encode_matmul_to_buffer(
                     encoder, weights.gate_up, norm2_buf, fused_buf, active_tokens
                   )
              raise ArgumentError.new("no resident Metal route for gate/up projection")
            end
            encode_swiglu(encoder, fused_buf, activated_buf, active_tokens, intermediate)
            unless Qwen35Metal.encode_matmul_to_buffer(
                     encoder, weights.mlp_out, activated_buf, mlp_buf, active_tokens
                   )
              raise ArgumentError.new("no resident Metal route for MLP output projection")
            end
            encode_residual_gate_add(
              encoder, state_buf, gate2_buf, mlp_buf, next_hidden_buf, active_count
            )

            previous_hidden_buf = current_hidden_buf
            current_hidden_buf = next_hidden_buf
            next_hidden_buf = previous_hidden_buf
          end
          if scales_buf = final_scales_buf
            hidden_buf = final_hidden_buf.not_nil!
            normalized_buf = final_norm_buf.not_nil!
            output_buf = final_output_buf.not_nil!
            weight = output_weight.not_nil!
            encode_copy_f32(
              encoder, cache.prefix_output_buffer, hidden_buf, prefix_values
            )
            encode_copy_f32(
              encoder, current_hidden_buf, hidden_buf, active_count,
              destination_offset: prefix_values,
            )
            encode_final_layernorm_scale(
              encoder, hidden_buf, scales_buf, normalized_buf,
              token_count, dim, config.eps,
            )
            unless QwenImage21MetalBF16.encode_matmul_to_buffer(
                     encoder, weight, normalized_buf, output_buf, token_count
                   )
              raise ArgumentError.new("no resident Metal route for final output projection")
            end
          end
          encoder.end_encoding
          command.commit
          command.wait

          values = if output_buf = final_output_buf
                     output_buf.read(token_count * output_weight.not_nil!.out_dim)
                   else
                     cache.prefix_output + current_hidden_buf.read(active_count)
                   end
          projection_dispatches = layers.size * 6 + (final_output_buf ? 1 : 0)
          QwenImage21MetalBlockResult.new(
            values,
            QwenImage21MetalBlockStats.new(1, projection_dispatches, 0, 1, active_tokens),
          )
        ensure
          buffers.each(&.release)
        end
      end

      private def self.forward_layers_impl(
        hidden : Array(Float32)?, token_count : Int32,
        modulation : Array(Float32)?,
        positions : Array(StaticArray(Int32, 3)),
        image_ids : Array(Int32),
        layers : Array(QwenImage21BlockWeights),
        config : QwenImage21BlockConfig,
        key_valid : Array(Bool)?, capture_cache : QwenImage21MetalPrefixCache?,
        final_scales : Array(Float32)?, output_weight : QuantWeight?,
        resident_input : QwenImage21MetalResidentInput? = nil,
      ) : QwenImage21MetalBlockResult
        if input = resident_input
          validate_resident_input(input, token_count, positions, image_ids,
            layers, config, key_valid, output_weight)
        else
          validate_inputs(hidden.not_nil!, token_count, modulation.not_nil!, positions, image_ids,
            layers, config, key_valid)
          validate_final_head(final_scales, output_weight, token_count, config.hidden_dim)
        end
        valid = key_valid || Array(Bool).new(token_count, true)
        validate_attention_rows(image_ids, valid)
        ML::Metal::Device.init!

        dim = config.hidden_dim
        intermediate = config.intermediate_dim
        hidden_count = token_count * dim
        hidden_bytes = hidden_count.to_i64 * sizeof(Float32)
        buffers = [] of ML::MetalBuffer

        begin
          current_hidden_buf = if resident_input
                                 allocate(hidden_bytes, buffers)
                               else
                                 upload_f32(hidden.not_nil!, buffers)
                               end
          next_hidden_buf = allocate(hidden_bytes, buffers)
          modulation_buf = if resident_input
                             allocate(token_count.to_i64 * 4_i64 * dim * sizeof(Float32), buffers)
                           else
                             upload_f32(modulation.not_nil!, buffers)
                           end
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
          final_scales_buf = if resident_input
                               allocate(hidden_bytes, buffers)
                             else
                               final_scales.try { |values| upload_f32(values, buffers) }
                             end
          final_norm_buf = final_scales_buf ? allocate(hidden_bytes, buffers) : nil
          final_output_buf = if weight = output_weight
                               allocate(token_count.to_i64 * weight.out_dim * sizeof(Float32), buffers)
                             end

          command = ML::Metal::CommandBuffer.new
          encoder = ML::Metal::ComputeEncoder.new(command)
          if input = resident_input
            encode_resident_input(
              encoder, input, current_hidden_buf, modulation_buf,
              final_scales_buf.not_nil!, token_count, dim, buffers,
            )
          end
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
            if cache = capture_cache
              prefix_values = cache.prefix_tokens * dim
              encode_copy_f32(encoder, k_buf, cache.k_buffers[layer_index], prefix_values)
              encode_copy_f32(encoder, v_buf, cache.v_buffers[layer_index], prefix_values)
            end
            encode_attention(
              encoder, q_buf, k_buf, v_buf, image_ids_buf, key_valid_buf,
              attended_buf, token_count, token_count, 0, config,
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
          if cache = capture_cache
            encode_copy_f32(
              encoder,
              current_hidden_buf,
              cache.prefix_output_buffer,
              cache.prefix_tokens * dim,
            )
          end
          if scales_buf = final_scales_buf
            normalized_buf = final_norm_buf.not_nil!
            projected_buf = final_output_buf.not_nil!
            weight = output_weight.not_nil!
            encode_final_layernorm_scale(
              encoder, current_hidden_buf, scales_buf, normalized_buf,
              token_count, dim, config.eps,
            )
            unless QwenImage21MetalBF16.encode_matmul_to_buffer(
                     encoder, weight, normalized_buf, projected_buf, token_count
                   )
              raise ArgumentError.new("no resident Metal route for final output projection")
            end
          end
          encoder.end_encoding
          command.commit
          command.wait

          values = if output_buf = final_output_buf
                     output_buf.read(token_count * output_weight.not_nil!.out_dim)
                   else
                     current_hidden_buf.read(hidden_count)
                   end
          projection_dispatches = layers.size * 6 + (final_output_buf ? 1 : 0) + (resident_input ? 5 : 0)
          QwenImage21MetalBlockResult.new(
            values,
            QwenImage21MetalBlockStats.new(1, projection_dispatches, 0, 1, token_count),
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

      private def self.validate_resident_input(
        input : QwenImage21MetalResidentInput, token_count : Int32,
        positions : Array(StaticArray(Int32, 3)), image_ids : Array(Int32),
        layers : Array(QwenImage21BlockWeights), config : QwenImage21BlockConfig,
        key_valid : Array(Bool)?, output_weight : QuantWeight?,
      ) : Nil
        dim = config.hidden_dim
        raise ArgumentError.new("token_count must be positive") unless token_count > 0
        raise ArgumentError.new("resident image input dimension must be positive") unless input.image_weight.in_dim > 0
        raise ArgumentError.new("resident timestep input dimension must be positive") unless input.timestep_linear_1.in_dim > 0
        raise ArgumentError.new("positions size mismatch") unless positions.size == token_count
        raise ArgumentError.new("image_ids size mismatch") unless image_ids.size == token_count
        raise ArgumentError.new("key_valid size mismatch") if key_valid && key_valid.size != token_count
        raise ArgumentError.new("Metal attention supports head_dim <= 256") unless config.head_dim <= 256
        raise ArgumentError.new("layer stack must not be empty") if layers.empty?
        layers.each { |weights| validate_layer_weights(weights, config) }
        raise ArgumentError.new("resident source map size mismatch") unless input.source_rows.size == token_count
        raise ArgumentError.new("resident target mask size mismatch") unless input.target_mask.size == token_count
        raise ArgumentError.new("resident image input size mismatch") unless input.image_input.size.divisible_by?(input.image_weight.in_dim)
        image_row = 0
        input.source_rows.each do |row|
          next unless row < 0
          raise ArgumentError.new("resident image source order mismatch") unless row == -(image_row + 1)
          image_row += 1
        end
        raise ArgumentError.new("resident image row count mismatch") unless image_row == input.image_rows
        raise ArgumentError.new("resident text input size mismatch") unless input.projected_text.size.divisible_by?(dim)
        text_rows = input.projected_text.size // dim
        input.source_rows.each do |row|
          raise ArgumentError.new("resident source map out of bounds") unless row < 0 ? -row - 1 < input.image_rows : row < text_rows
        end
        raise ArgumentError.new("resident timestep input size mismatch") unless input.time_input.size.divisible_by?(input.timestep_linear_1.in_dim)
        raise ArgumentError.new("resident timestep row count mismatch") unless input.time_rows == 1 || input.time_rows == 2
        weights = {input.image_weight, input.timestep_linear_1, input.timestep_linear_2,
                   input.modulation_weight, input.scale_weight, output_weight.not_nil!}
        raise ArgumentError.new("resident inputs require BF16 weights") unless weights.all?(&.type.bf16?)
        unless input.image_weight.out_dim == dim && input.timestep_linear_1.out_dim == dim &&
               input.timestep_linear_2.in_dim == dim && input.timestep_linear_2.out_dim == dim &&
               input.modulation_weight.in_dim == dim && input.modulation_weight.out_dim == 4 * dim &&
               input.scale_weight.in_dim == dim && input.scale_weight.out_dim == dim &&
               output_weight.not_nil!.in_dim == dim
          raise ArgumentError.new("resident projection shape mismatch")
        end
      end

      private def self.encode_resident_input(
        encoder : ML::Metal::ComputeEncoder,
        input : QwenImage21MetalResidentInput,
        hidden_buf : ML::MetalBuffer, modulation_buf : ML::MetalBuffer,
        scales_buf : ML::MetalBuffer, token_count : Int32, dim : Int32,
        buffers : Array(ML::MetalBuffer),
      ) : Nil
        image_input_buf = upload_f32(input.image_input, buffers)
        text_buf = input.projected_text.empty? ? allocate(sizeof(Float32).to_i64, buffers) : upload_f32(input.projected_text, buffers)
        time_input_buf = upload_f32(input.time_input, buffers)
        source_rows_buf = upload_i32(input.source_rows, buffers)
        target_mask_buf = upload_u8(input.target_mask.map { |value| value ? 1_u8 : 0_u8 }, buffers)
        image_buf = allocate(input.image_rows.to_i64 * dim * sizeof(Float32), buffers)
        first_buf = allocate(input.time_rows.to_i64 * dim * sizeof(Float32), buffers)
        second_buf = allocate(input.time_rows.to_i64 * dim * sizeof(Float32), buffers)
        modulation_rows_buf = allocate(input.time_rows.to_i64 * 4_i64 * dim * sizeof(Float32), buffers)
        scale_rows_buf = allocate(input.time_rows.to_i64 * dim * sizeof(Float32), buffers)

        unless QwenImage21MetalBF16.encode_matmul_to_buffer(
                 encoder, input.image_weight, image_input_buf, image_buf, input.image_rows
               )
          raise ArgumentError.new("no resident Metal route for image input projection")
        end
        encoder.set_pipeline(pipeline("qi21_assemble_joint"))
        encoder.set_buffer(source_rows_buf, 0)
        encoder.set_buffer(text_buf, 1)
        encoder.set_buffer(image_buf, 2)
        encoder.set_buffer(hidden_buf, 3, ML::Metal::BufferAccess::Write)
        encoder.set_value(token_count.to_u32, 4)
        encoder.set_value(dim.to_u32, 5)
        encoder.dispatch_1d(token_count * dim, 256)

        unless QwenImage21MetalBF16.encode_matmul_to_buffer(
                 encoder, input.timestep_linear_1, time_input_buf, first_buf, input.time_rows
               )
          raise ArgumentError.new("no resident Metal route for timestep projection")
        end
        QwenImage21MetalBF16.encode_silu_to_buffer(encoder, first_buf, input.time_rows * dim)
        unless QwenImage21MetalBF16.encode_matmul_to_buffer(
                 encoder, input.timestep_linear_2, first_buf, second_buf, input.time_rows
               )
          raise ArgumentError.new("no resident Metal route for timestep projection")
        end
        QwenImage21MetalBF16.encode_silu_to_buffer(encoder, second_buf, input.time_rows * dim)
        unless QwenImage21MetalBF16.encode_matmul_to_buffer(
                 encoder, input.modulation_weight, second_buf, modulation_rows_buf, input.time_rows
               ) && QwenImage21MetalBF16.encode_matmul_to_buffer(
                 encoder, input.scale_weight, second_buf, scale_rows_buf, input.time_rows
               )
          raise ArgumentError.new("no resident Metal route for timestep output projections")
        end
        encode_select_time_rows(
          encoder, modulation_rows_buf, target_mask_buf, modulation_buf,
          token_count, 4 * dim, input.time_rows,
        )
        encode_select_time_rows(
          encoder, scale_rows_buf, target_mask_buf, scales_buf,
          token_count, dim, input.time_rows,
        )
      end

      private def self.encode_select_time_rows(
        encoder : ML::Metal::ComputeEncoder,
        rows : ML::MetalBuffer, target_mask : ML::MetalBuffer,
        output : ML::MetalBuffer, token_count : Int32, width : Int32,
        row_count : Int32,
      ) : Nil
        encoder.set_pipeline(pipeline("qi21_select_time_rows"))
        encoder.set_buffer(rows, 0)
        encoder.set_buffer(target_mask, 1)
        encoder.set_buffer(output, 2, ML::Metal::BufferAccess::Write)
        encoder.set_value(token_count.to_u32, 3)
        encoder.set_value(width.to_u32, 4)
        encoder.set_value(row_count.to_u32, 5)
        encoder.dispatch_1d(token_count * width, 256)
      end

      private def self.validate_final_head(
        scales : Array(Float32)?, output_weight : QuantWeight?,
        token_count : Int32, hidden_dim : Int32,
      ) : Nil
        if values = scales
          weight = output_weight || raise ArgumentError.new("output weight is required with final scales")
          unless values.size == token_count * hidden_dim
            raise ArgumentError.new("final scale size mismatch")
          end
          unless weight.type.bf16? && weight.in_dim == hidden_dim
            raise ArgumentError.new("resident final projection requires BF16 hidden_dim input")
          end
        elsif output_weight
          raise ArgumentError.new("final scales are required with output weight")
        end
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
        output : ML::MetalBuffer, total_tokens : Int32,
        query_tokens : Int32, query_offset : Int32,
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
        encoder.set_value(total_tokens.to_u32, 6)
        encoder.set_value(query_tokens.to_u32, 7)
        encoder.set_value(query_offset.to_u32, 8)
        encoder.set_value(config.heads.to_u32, 9)
        encoder.set_value(config.head_dim.to_u32, 10)
        encoder.set_value((1.0_f64 / Math.sqrt(config.head_dim)).to_f32, 11)
        encoder.dispatch_threadgroups({query_tokens * config.heads, 1, 1}, {threads, 1, 1})
      end

      private def self.encode_copy_f32(
        encoder : ML::Metal::ComputeEncoder,
        source : ML::MetalBuffer, destination : ML::MetalBuffer,
        count : Int32, source_offset : Int32 = 0, destination_offset : Int32 = 0,
      ) : Nil
        encoder.set_pipeline(pipeline("qi21_copy_f32"))
        encoder.set_buffer(source, 0, offset: source_offset.to_i64 * sizeof(Float32))
        encoder.set_buffer(
          destination, 1, ML::Metal::BufferAccess::Write,
          offset: destination_offset.to_i64 * sizeof(Float32),
        )
        encoder.set_value(count.to_u32, 2)
        encoder.dispatch_1d(count, 256)
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

      private def self.encode_final_layernorm_scale(
        encoder : ML::Metal::ComputeEncoder,
        hidden : ML::MetalBuffer, scales : ML::MetalBuffer,
        output : ML::MetalBuffer,
        tokens : Int32, dim : Int32, eps : Float32,
      ) : Nil
        encoder.set_pipeline(pipeline("qi21_final_layernorm_scale"))
        encoder.set_buffer(hidden, 0)
        encoder.set_buffer(scales, 1)
        encoder.set_buffer(output, 2, ML::Metal::BufferAccess::Write)
        encoder.set_value(tokens.to_u32, 3)
        encoder.set_value(dim.to_u32, 4)
        encoder.set_value(eps, 5)
        encoder.dispatch_threadgroups({tokens, 1, 1}, {256, 1, 1})
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
    include QwenImage21FusedLayerStackBackend
    include QwenImage21ResidentInputStackBackend

    getter last_stats : QwenImage21MetalBlockStats?
    getter invocations = 0
    getter prefix_cache_builds = 0
    getter prefix_cache_hits = 0
    getter resident_head_invocations = 0
    getter resident_input_invocations = 0

    def initialize
      @last_stats = nil
      @prefix_cache = nil.as(QwenImage21MetalPrefixCache?)
      @closed = false
    end

    def self.available? : Bool
      QwenImage21MetalBlock.available?
    end

    def forward_resident_input(
      image_input : Array(Float32), projected_text : Array(Float32),
      time_input : Array(Float32), img_mask : Array(Bool),
      layout : QwenImage21TokenLayout,
      weights : QwenImage21TransformerWeights,
      config : QwenImage21TransformerConfig,
    ) : Array(Float32)?
      raise ArgumentError.new("Qwen-Image Metal layer stack is closed") if @closed
      return nil unless QwenImage21MetalBlock.available?
      return nil unless {weights.img_in, weights.timestep_linear_1,
                         weights.timestep_linear_2, weights.modulation,
                         weights.norm_out_linear, weights.proj_out}.all?(&.type.bf16?)
      source_rows = [] of Int32
      image_row = 0
      img_mask.each_with_index do |is_image, base_row|
        if is_image
          QwenImage21TransformerCPU::IMG_TOKENS_PER_SLOT.times do
            source_rows << -(image_row + 1)
            image_row += 1
          end
        else
          source_rows << base_row
        end
      end
      input = QwenImage21MetalResidentInput.new(
        image_input, projected_text, time_input, source_rows,
        layout.target_token_mask, weights.img_in, weights.timestep_linear_1,
        weights.timestep_linear_2, weights.modulation, weights.norm_out_linear,
      )
      invalidate_prefix_cache
      result = QwenImage21MetalBlock.forward_layers_resident(
        input, layout.token_count, layout.positions, layout.image_ids,
        weights.layers, config.block, weights.proj_out, layout.key_valid,
      )
      @last_stats = result.stats
      @invocations += 1
      @resident_head_invocations += 1
      @resident_input_invocations += 1
      result.output
    end

    def forward_layers(
      hidden : Array(Float32), token_count : Int32,
      modulation : Array(Float32),
      positions : Array(StaticArray(Int32, 3)),
      image_ids : Array(Int32),
      layers : Array(QwenImage21BlockWeights),
      config : QwenImage21BlockConfig,
      key_valid : Array(Bool)?,
      target_start : Int32?,
    ) : Array(Float32)
      raise ArgumentError.new("Qwen-Image Metal layer stack is closed") if @closed
      valid = key_valid || Array(Bool).new(token_count, true)
      result = if prefix_tokens = target_start
                 if prefix_tokens > 0 && prefix_tokens < token_count
                   if cache = @prefix_cache
                     if cache.compatible?(
                          hidden, token_count, modulation, positions, image_ids,
                          valid, layers, config, prefix_tokens,
                        )
                       @prefix_cache_hits += 1
                       QwenImage21MetalBlock.forward_layers_from_prefix_cache(
                         hidden, token_count, modulation, positions, image_ids,
                         layers, config, valid, cache,
                       )
                     else
                       rebuild_prefix_cache(
                         hidden, token_count, modulation, positions, image_ids,
                         layers, config, valid, prefix_tokens,
                       )
                     end
                   else
                     rebuild_prefix_cache(
                       hidden, token_count, modulation, positions, image_ids,
                       layers, config, valid, prefix_tokens,
                     )
                   end
                 else
                   invalidate_prefix_cache
                   QwenImage21MetalBlock.forward_layers(
                     hidden, token_count, modulation, positions, image_ids,
                     layers, config, valid,
                   )
                 end
               else
                 invalidate_prefix_cache
                 QwenImage21MetalBlock.forward_layers(
                   hidden, token_count, modulation, positions, image_ids,
                   layers, config, valid,
                 )
               end
      @last_stats = result.stats
      @invocations += 1
      result.hidden
    end

    def forward_layers_projected(
      hidden : Array(Float32), token_count : Int32,
      modulation : Array(Float32),
      positions : Array(StaticArray(Int32, 3)),
      image_ids : Array(Int32),
      layers : Array(QwenImage21BlockWeights),
      config : QwenImage21BlockConfig,
      key_valid : Array(Bool)?, target_start : Int32?,
      scales : Array(Float32), output_weight : QuantWeight,
    ) : Array(Float32)?
      return nil unless output_weight.type.bf16?
      return nil unless output_weight.in_dim == config.hidden_dim
      return nil unless scales.size == token_count * config.hidden_dim
      raise ArgumentError.new("Qwen-Image Metal layer stack is closed") if @closed
      valid = key_valid || Array(Bool).new(token_count, true)
      result = if prefix_tokens = target_start
                 if prefix_tokens > 0 && prefix_tokens < token_count
                   if cache = @prefix_cache
                     if cache.compatible?(
                          hidden, token_count, modulation, positions, image_ids,
                          valid, layers, config, prefix_tokens,
                        )
                       @prefix_cache_hits += 1
                       QwenImage21MetalBlock.forward_layers_projected_from_prefix_cache(
                         hidden, token_count, modulation, positions, image_ids,
                         layers, config, valid, cache, scales, output_weight,
                       )
                     else
                       rebuild_prefix_cache_projected(
                         hidden, token_count, modulation, positions, image_ids,
                         layers, config, valid, prefix_tokens, scales, output_weight,
                       )
                     end
                   else
                     rebuild_prefix_cache_projected(
                       hidden, token_count, modulation, positions, image_ids,
                       layers, config, valid, prefix_tokens, scales, output_weight,
                     )
                   end
                 else
                   invalidate_prefix_cache
                   QwenImage21MetalBlock.forward_layers_projected(
                     hidden, token_count, modulation, positions, image_ids,
                     layers, config, scales, output_weight, valid,
                   )
                 end
               else
                 invalidate_prefix_cache
                 QwenImage21MetalBlock.forward_layers_projected(
                   hidden, token_count, modulation, positions, image_ids,
                   layers, config, scales, output_weight, valid,
                 )
               end
      @last_stats = result.stats
      @invocations += 1
      @resident_head_invocations += 1
      result.output
    end

    def close : Nil
      return if @closed
      invalidate_prefix_cache
      @closed = true
    end

    def finalize
      close
    rescue
      nil
    end

    private def rebuild_prefix_cache(
      hidden : Array(Float32), token_count : Int32,
      modulation : Array(Float32),
      positions : Array(StaticArray(Int32, 3)), image_ids : Array(Int32),
      layers : Array(QwenImage21BlockWeights), config : QwenImage21BlockConfig,
      key_valid : Array(Bool), prefix_tokens : Int32,
    ) : QwenImage21MetalBlockResult
      invalidate_prefix_cache
      result, cache = QwenImage21MetalBlock.forward_layers_capture_prefix(
        hidden, token_count, modulation, positions, image_ids,
        layers, config, key_valid, prefix_tokens,
      )
      @prefix_cache = cache
      @prefix_cache_builds += 1
      result
    end

    private def rebuild_prefix_cache_projected(
      hidden : Array(Float32), token_count : Int32,
      modulation : Array(Float32),
      positions : Array(StaticArray(Int32, 3)), image_ids : Array(Int32),
      layers : Array(QwenImage21BlockWeights), config : QwenImage21BlockConfig,
      key_valid : Array(Bool), prefix_tokens : Int32,
      scales : Array(Float32), output_weight : QuantWeight,
    ) : QwenImage21MetalProjectedResult
      invalidate_prefix_cache
      result, cache = QwenImage21MetalBlock.forward_layers_projected_capture_prefix(
        hidden, token_count, modulation, positions, image_ids,
        layers, config, scales, output_weight, key_valid, prefix_tokens,
      )
      @prefix_cache = cache
      @prefix_cache_builds += 1
      result
    end

    private def invalidate_prefix_cache : Nil
      @prefix_cache.try(&.close)
      @prefix_cache = nil
    end
  end
end
