# Bounded graphless CPU reference for the TRELLIS.2 DINOv3 embedding boundary.
# This slice covers patch projection, prefix-token concatenation, dynamic patch
# coordinates, and RoPE cos/sin only. It behaviorally follows the pinned
# Apache-2.0 Transformers 5.8.1 source identified below, without linking or
# executing it. It does not execute attention, blocks, checkpoints, mixed
# precision, accelerators, or Metal.

require "../../core/tensor"
require "digest/sha256"

module ML::Vision::DinoV3
  TRELLIS2_SOURCE_REVISION     = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
  TRANSFORMERS_MODELING_SHA256 = "6073b7665eea50fb2260d86984011e6af8ed68cbd37ae57a6095e5e90e0eea34"
  TRANSFORMERS_CONFIG_SHA256   = "9a13d3c9ea8020aaed7057db28a7ffe0ffd9bb094a6178261bcc45ed04e9bbfc"

  class EmbeddingError < Exception
  end

  class EmbeddingBudgetError < EmbeddingError
  end

  struct EmbeddingConfig
    getter patch_size : Int32
    getter num_channels : Int32
    getter hidden_size : Int32
    getter num_attention_heads : Int32
    getter num_register_tokens : Int32
    getter rope_theta : Float32
    getter head_dim : Int32

    def initialize(
      @patch_size : Int32,
      @num_channels : Int32,
      @hidden_size : Int32,
      @num_attention_heads : Int32,
      @num_register_tokens : Int32,
      @rope_theta : Float32,
    )
      unless @patch_size == 16
        raise EmbeddingError.new("DINOv3 patch size must be 16 in the admitted TRELLIS.2 boundary")
      end
      unless @num_channels == 3
        raise EmbeddingError.new("DINOv3 input channel count must be 3")
      end
      unless 0 < @hidden_size <= 1024
        raise EmbeddingError.new("DINOv3 hidden size must be in 1..1024")
      end
      unless 0 < @num_attention_heads <= 64
        raise EmbeddingError.new("DINOv3 attention head count must be in 1..64")
      end
      unless @hidden_size % @num_attention_heads == 0
        raise EmbeddingError.new("DINOv3 hidden size must be divisible by attention heads")
      end
      @head_dim = @hidden_size // @num_attention_heads
      unless {4_i32, 8_i32, 16_i32, 32_i32, 64_i32}.includes?(@head_dim)
        raise EmbeddingError.new(
          "DINOv3 head dimension must be a supported power of two in 4..64 for exact 2D RoPE"
        )
      end
      unless 0 <= @num_register_tokens <= 16
        raise EmbeddingError.new("DINOv3 register token count must be in 0..16")
      end
      unless @rope_theta.finite? && @rope_theta > 0.0_f32
        raise EmbeddingError.new("DINOv3 RoPE theta must be finite and positive")
      end
    end
  end

  # Parameter arrays are retained and read without copying. This is a
  # single-owner reference API: the caller must keep input and parameter owners
  # alive and must not mutate them or call forward concurrently while a call is
  # in progress.
  class EmbeddingParameters
    MAX_PARAMETER_BYTES = 64_i64 * 1024_i64 * 1024_i64

    getter config : EmbeddingConfig
    getter patch_weight : Array(Float32)
    getter patch_bias : Array(Float32)
    getter cls_token : Array(Float32)
    getter register_tokens : Array(Float32)

    def initialize(
      @config : EmbeddingConfig,
      @patch_weight : Array(Float32),
      @patch_bias : Array(Float32),
      @cls_token : Array(Float32),
      @register_tokens : Array(Float32),
    )
      validate!
    end

    def validate! : Nil
      patch_elements = @config.hidden_size.to_i64 * @config.num_channels *
                       @config.patch_size * @config.patch_size
      total_elements = patch_elements + @config.hidden_size.to_i64 * 2_i64 +
                       @config.num_register_tokens.to_i64 * @config.hidden_size
      total_bytes = total_elements * 4_i64
      if total_bytes > MAX_PARAMETER_BYTES
        raise EmbeddingBudgetError.new(
          "DINOv3 embedding parameters require #{total_bytes} bytes, limit is #{MAX_PARAMETER_BYTES}"
        )
      end
      validate_length!(@patch_weight, patch_elements, "patch weight")
      validate_length!(@patch_bias, @config.hidden_size, "patch bias")
      validate_length!(@cls_token, @config.hidden_size, "CLS token")
      validate_length!(
        @register_tokens,
        @config.num_register_tokens.to_i64 * @config.hidden_size,
        "register tokens"
      )
      validate_finite!(@patch_weight, "patch weight")
      validate_finite!(@patch_bias, "patch bias")
      validate_finite!(@cls_token, "CLS token")
      validate_finite!(@register_tokens, "register tokens")
    end

    def f32le_sha256 : String
      digest = Digest::SHA256.new
      bytes = Bytes.new(4, 0_u8)
      {
        @patch_weight,
        @patch_bias,
        @cls_token,
        @register_tokens,
      }.each do |values|
        values.each do |value|
          IO::ByteFormat::LittleEndian.encode(value, bytes)
          digest.update(bytes)
        end
      end
      digest.final.hexstring
    end

    private def validate_length!(values : Array(Float32), expected : Int64, name : String) : Nil
      unless values.size.to_i64 == expected
        raise EmbeddingError.new(
          "DINOv3 #{name} has #{values.size} elements, expected #{expected}"
        )
      end
    end

    private def validate_finite!(values : Array(Float32), name : String) : Nil
      values.each_with_index do |value, index|
        unless value.finite?
          raise EmbeddingError.new("DINOv3 #{name}[#{index}] must be finite")
        end
      end
    end
  end

  record EmbeddingResult,
    patches : Tensor,
    embeddings : Tensor,
    patch_coordinates : Tensor,
    rope_cos : Tensor,
    rope_sin : Tensor,
    config : EmbeddingConfig,
    parameter_f32le_sha256 : String,
    source_revision : String,
    transformers_modeling_sha256 : String,
    transformers_config_sha256 : String

  class EmbeddingCPU
    MAX_INPUT_BYTES      = 16_i64 * 1024_i64 * 1024_i64
    MAX_OUTPUT_BYTES     = 64_i64 * 1024_i64 * 1024_i64
    MAX_MULTIPLY_ADDS    = 64_i64 * 1024_i64 * 1024_i64
    ADMITTED_RESOLUTIONS = {512_i32, 1024_i32}

    getter config : EmbeddingConfig
    getter parameters : EmbeddingParameters

    def initialize(@parameters : EmbeddingParameters)
      @config = @parameters.config
    end

    def forward(input : Tensor) : EmbeddingResult
      edge = validate_input!(input)
      patch_edge = edge // @config.patch_size
      patch_count = patch_edge.to_i64 * patch_edge
      preflight!(input, patch_count)
      @parameters.validate!
      parameter_f32le_sha256 = @parameters.f32le_sha256

      input_values = input.cpu_read
      validate_finite_input!(input_values)
      patches = Tensor.new(
        Shape.new(1_i32, patch_count.to_i32, @config.hidden_size),
        device: Tensor::Device::CPU
      )
      project_patches!(
        input_values,
        edge,
        patch_edge,
        patches.cpu_data.not_nil!
      )

      prefix_count = 1_i64 + @config.num_register_tokens
      embeddings = Tensor.new(
        Shape.new(
          1_i32,
          (prefix_count + patch_count).to_i32,
          @config.hidden_size
        ),
        device: Tensor::Device::CPU
      )
      concatenate_prefix!(patches.cpu_data.not_nil!, embeddings.cpu_data.not_nil!)

      coordinates = Tensor.new(
        Shape.new(patch_count.to_i32, 2_i32),
        device: Tensor::Device::CPU
      )
      build_coordinates!(patch_edge, coordinates.cpu_data.not_nil!)

      rope_shape = Shape.new(
        patch_count.to_i32,
        @config.head_dim
      )
      rope_cos = Tensor.new(rope_shape, device: Tensor::Device::CPU)
      rope_sin = Tensor.new(rope_shape, device: Tensor::Device::CPU)
      build_rope!(
        coordinates.cpu_data.not_nil!,
        patch_count.to_i32,
        rope_cos.cpu_data.not_nil!,
        rope_sin.cpu_data.not_nil!
      )

      EmbeddingResult.new(
        patches,
        embeddings,
        coordinates,
        rope_cos,
        rope_sin,
        @config,
        parameter_f32le_sha256,
        TRELLIS2_SOURCE_REVISION,
        TRANSFORMERS_MODELING_SHA256,
        TRANSFORMERS_CONFIG_SHA256
      )
    end

    private def validate_input!(input : Tensor) : Int32
      unless input.on_cpu?
        raise EmbeddingError.new("DINOv3 embedding reference accepts CPU tensors only")
      end
      unless input.contiguous?
        raise EmbeddingError.new("DINOv3 embedding input must be contiguous NCHW")
      end
      shape = input.shape
      unless shape.ndim == 4
        raise EmbeddingError.new(
          "DINOv3 embedding input must have rank 4 [1, 3, H, W], got #{shape}"
        )
      end
      unless shape[0] == 1
        raise EmbeddingError.new("DINOv3 embedding reference admits batch size 1 only")
      end
      unless shape[1] == @config.num_channels
        raise EmbeddingError.new(
          "DINOv3 embedding input must have #{@config.num_channels} channels"
        )
      end
      unless shape[2] == shape[3]
        raise EmbeddingError.new("DINOv3 embedding input must be square")
      end
      edge = shape[2]
      unless ADMITTED_RESOLUTIONS.includes?(edge)
        raise EmbeddingError.new("DINOv3 embedding resolution must be 512 or 1024")
      end
      unless edge % @config.patch_size == 0
        raise EmbeddingError.new(
          "DINOv3 embedding resolution must be divisible by patch size #{@config.patch_size}"
        )
      end
      edge
    end

    private def validate_finite_input!(values : Indexable(Float32)) : Nil
      values.each_with_index do |value, index|
        unless value.finite?
          raise EmbeddingError.new("DINOv3 embedding input[#{index}] must be finite")
        end
      end
    end

    private def preflight!(input : Tensor, patch_count : Int64) : Nil
      input_bytes = input.numel.to_i64 * 4_i64
      if input_bytes > MAX_INPUT_BYTES
        raise EmbeddingBudgetError.new(
          "DINOv3 embedding input requires #{input_bytes} bytes, limit is #{MAX_INPUT_BYTES}"
        )
      end

      patch_elements = patch_count * @config.hidden_size
      embedding_elements = (
        patch_count + 1_i64 + @config.num_register_tokens
      ) * @config.hidden_size
      coordinate_elements = patch_count * 2_i64
      rope_elements = patch_count * @config.head_dim * 2_i64
      output_bytes = (
        patch_elements + embedding_elements + coordinate_elements + rope_elements
      ) * 4_i64
      if output_bytes > MAX_OUTPUT_BYTES
        raise EmbeddingBudgetError.new(
          "DINOv3 embedding outputs require #{output_bytes} bytes, limit is #{MAX_OUTPUT_BYTES}"
        )
      end

      multiply_adds = patch_count * @config.hidden_size * @config.num_channels *
                      @config.patch_size * @config.patch_size
      if multiply_adds > MAX_MULTIPLY_ADDS
        raise EmbeddingBudgetError.new(
          "DINOv3 patch projection requires #{multiply_adds} multiply-adds, limit is #{MAX_MULTIPLY_ADDS}"
        )
      end
    end

    private def project_patches!(
      input : Indexable(Float32),
      edge : Int32,
      patch_edge : Int32,
      output : Array(Float32),
    ) : Nil
      patch_size = @config.patch_size
      hidden = @config.hidden_size
      channels = @config.num_channels
      weights = @parameters.patch_weight
      bias = @parameters.patch_bias

      patch_edge.times do |patch_y|
        patch_edge.times do |patch_x|
          patch_index = patch_y * patch_edge + patch_x
          hidden.times do |feature|
            accumulator = bias[feature]
            channels.times do |channel|
              patch_size.times do |kernel_y|
                source_y = patch_y * patch_size + kernel_y
                patch_size.times do |kernel_x|
                  source_x = patch_x * patch_size + kernel_x
                  input_offset = channel * edge * edge + source_y * edge + source_x
                  weight_offset = (
                    ((feature * channels + channel) * patch_size + kernel_y) *
                    patch_size + kernel_x
                  )
                  accumulator += input[input_offset] * weights[weight_offset]
                end
              end
            end
            unless accumulator.finite?
              raise EmbeddingError.new(
                "DINOv3 patch projection produced a non-finite value at patch #{patch_index}, feature #{feature}"
              )
            end
            output[patch_index * hidden + feature] = accumulator
          end
        end
      end
    end

    private def concatenate_prefix!(patches : Array(Float32), output : Array(Float32)) : Nil
      hidden = @config.hidden_size
      @parameters.cls_token.each_with_index do |value, index|
        output[index] = value
      end
      @parameters.register_tokens.each_with_index do |value, index|
        output[hidden + index] = value
      end
      prefix_elements = (1 + @config.num_register_tokens) * hidden
      patches.each_with_index do |value, index|
        output[prefix_elements + index] = value
      end
    end

    private def build_coordinates!(patch_edge : Int32, output : Array(Float32)) : Nil
      patch_edge.times do |patch_y|
        coordinate_y = (
          (patch_y.to_f32 + 0.5_f32) / patch_edge.to_f32
        ) * 2.0_f32 - 1.0_f32
        patch_edge.times do |patch_x|
          coordinate_x = (
            (patch_x.to_f32 + 0.5_f32) / patch_edge.to_f32
          ) * 2.0_f32 - 1.0_f32
          patch_index = patch_y * patch_edge + patch_x
          output[patch_index * 2] = coordinate_y
          output[patch_index * 2 + 1] = coordinate_x
        end
      end
    end

    private def build_rope!(
      coordinates : Array(Float32),
      patch_count : Int32,
      cos_output : Array(Float32),
      sin_output : Array(Float32),
    ) : Nil
      head_dim = @config.head_dim
      frequency_count = head_dim // 4
      half_dim = head_dim // 2
      exponent_step = 4.0_f32 / head_dim.to_f32
      inverse_frequencies = Array(Float32).new(frequency_count) do |index|
        exponent = index.to_f32 * exponent_step
        inverse = (
          1.0_f64 / (@config.rope_theta.to_f64 ** exponent.to_f64)
        ).to_f32
        unless inverse.finite?
          raise EmbeddingError.new("DINOv3 RoPE inverse frequency must be finite")
        end
        inverse
      end
      two_pi = (2.0_f64 * Math::PI).to_f32

      patch_count.times do |patch_index|
        head_dim.times do |dimension|
          half_index = dimension % half_dim
          axis = half_index // frequency_count
          frequency = half_index % frequency_count
          coordinate = coordinates[patch_index * 2 + axis]
          angle = two_pi * coordinate * inverse_frequencies[frequency]
          offset = patch_index * head_dim + dimension
          cos_output[offset] = Math.cos(angle).to_f32
          sin_output[offset] = Math.sin(angle).to_f32
        end
      end
    end
  end
end
