# Bounded graphless CPU reference for one TRELLIS.2 DINOv3 transformer block.
#
# This implementation intentionally owns no graph, autograd, accelerator, or
# checkpoint machinery. Mutable input and RoPE storage is snapshotted at the
# execution boundary; parameter arrays are retained without copying. The caller
# is the single parameter owner and must not mutate arrays or call forward
# concurrently while a call is in progress. Parameters are validated before
# each call, and their byte digest is checked both before and after execution.
#
# Scalar reductions accumulate in Float64 and round each materialized boundary
# to Float32. The pinned PyTorch Float32 oracle is therefore checked with a
# declared absolute/relative tolerance, not claimed as byte-identical output.

require "digest/sha256"
require "./embeddings"

module ML::Vision::DinoV3
  TRELLIS2_EXTRACTOR_SHA256 = "12530b23e8b6a2cc6b87d8cd01922c7b0085199a365b731f19dd0e7ef4919150"

  class BlockError < EmbeddingError
  end

  class BlockBudgetError < BlockError
  end

  struct BlockConfig
    getter hidden_size : Int32
    getter intermediate_size : Int32
    getter num_attention_heads : Int32
    getter num_register_tokens : Int32
    getter layer_norm_eps : Float32
    getter hidden_act : String
    getter query_bias : Bool
    getter key_bias : Bool
    getter value_bias : Bool
    getter proj_bias : Bool
    getter mlp_bias : Bool
    getter attention_dropout : Float32
    getter drop_path_rate : Float32
    getter use_gated_mlp : Bool
    getter attention_backend : String
    getter training : Bool
    getter extractor_final_layer_norm_eps : Float32
    getter head_dim : Int32

    def initialize(
      @hidden_size : Int32,
      @intermediate_size : Int32,
      @num_attention_heads : Int32,
      @num_register_tokens : Int32,
      @layer_norm_eps : Float32,
      @hidden_act : String = "gelu",
      @query_bias : Bool = true,
      @key_bias : Bool = false,
      @value_bias : Bool = true,
      @proj_bias : Bool = true,
      @mlp_bias : Bool = true,
      @attention_dropout : Float32 = 0.0_f32,
      @drop_path_rate : Float32 = 0.0_f32,
      @use_gated_mlp : Bool = false,
      @attention_backend : String = "eager",
      @training : Bool = false,
      @extractor_final_layer_norm_eps : Float32 = 1.0e-5_f32,
    )
      unless 0 < @hidden_size <= 64
        raise BlockError.new("DINOv3 block hidden size must be in 1..64")
      end
      unless 0 < @intermediate_size <= 256
        raise BlockError.new("DINOv3 block intermediate size must be in 1..256")
      end
      unless 0 < @num_attention_heads <= 8
        raise BlockError.new("DINOv3 block attention heads must be in 1..8")
      end
      unless @hidden_size % @num_attention_heads == 0
        raise BlockError.new("DINOv3 block hidden size must be divisible by attention heads")
      end
      @head_dim = @hidden_size // @num_attention_heads
      unless {4_i32, 8_i32, 16_i32, 32_i32, 64_i32}.includes?(@head_dim)
        raise BlockError.new(
          "DINOv3 block head dimension must be one of 4, 8, 16, 32, or 64"
        )
      end
      unless 0 <= @num_register_tokens <= 16
        raise BlockError.new("DINOv3 block register token count must be in 0..16")
      end
      unless @layer_norm_eps.finite? && @layer_norm_eps > 0.0_f32
        raise BlockError.new("DINOv3 block layer norm epsilon must be finite and positive")
      end
      unless @hidden_act == "gelu"
        raise BlockError.new("DINOv3 block requires exact GELU")
      end
      unless @query_bias && !@key_bias && @value_bias && @proj_bias && @mlp_bias
        raise BlockError.new(
          "DINOv3 block requires q/v/projection/MLP bias and no key bias"
        )
      end
      unless @attention_dropout.finite? && @attention_dropout == 0.0_f32
        raise BlockError.new("DINOv3 block requires zero attention dropout")
      end
      unless @drop_path_rate.finite? && @drop_path_rate == 0.0_f32
        raise BlockError.new("DINOv3 block requires zero drop path")
      end
      if @use_gated_mlp
        raise BlockError.new("DINOv3 block reference does not support gated MLP")
      end
      unless @attention_backend == "eager"
        raise BlockError.new("DINOv3 block reference requires eager attention")
      end
      if @training
        raise BlockError.new("DINOv3 block reference requires evaluation mode")
      end
      unless @extractor_final_layer_norm_eps.finite? &&
             @extractor_final_layer_norm_eps == 1.0e-5_f32
        raise BlockError.new(
          "DINOv3 TRELLIS extractor final layer norm epsilon must equal 1e-5"
        )
      end
    end
  end

  class BlockParameters
    # Defense-in-depth for a later enlarged envelope. The structural limits in
    # BlockConfig are intentionally tighter for this synthetic CPU slice.
    MAX_PARAMETER_BYTES = 64_i64 * 1024_i64 * 1024_i64

    getter config : BlockConfig
    getter norm1_weight : Array(Float32)
    getter norm1_bias : Array(Float32)
    getter q_weight : Array(Float32)
    getter q_bias : Array(Float32)
    getter k_weight : Array(Float32)
    getter v_weight : Array(Float32)
    getter v_bias : Array(Float32)
    getter o_weight : Array(Float32)
    getter o_bias : Array(Float32)
    getter layer_scale1 : Array(Float32)
    getter norm2_weight : Array(Float32)
    getter norm2_bias : Array(Float32)
    getter up_weight : Array(Float32)
    getter up_bias : Array(Float32)
    getter down_weight : Array(Float32)
    getter down_bias : Array(Float32)
    getter layer_scale2 : Array(Float32)

    def initialize(
      @config : BlockConfig,
      *,
      @norm1_weight : Array(Float32),
      @norm1_bias : Array(Float32),
      @q_weight : Array(Float32),
      @q_bias : Array(Float32),
      @k_weight : Array(Float32),
      @v_weight : Array(Float32),
      @v_bias : Array(Float32),
      @o_weight : Array(Float32),
      @o_bias : Array(Float32),
      @layer_scale1 : Array(Float32),
      @norm2_weight : Array(Float32),
      @norm2_bias : Array(Float32),
      @up_weight : Array(Float32),
      @up_bias : Array(Float32),
      @down_weight : Array(Float32),
      @down_bias : Array(Float32),
      @layer_scale2 : Array(Float32),
    )
      validate!
    end

    # Revalidate retained arrays at each execution boundary. This is public so
    # callers that retain a parameter owner can explicitly preflight it too.
    def validate! : Nil
      c = @config.hidden_size.to_i64
      i = @config.intermediate_size.to_i64
      expected = {
        {@norm1_weight, c, "norm1 weight"},
        {@norm1_bias, c, "norm1 bias"},
        {@q_weight, c * c, "query weight"},
        {@q_bias, c, "query bias"},
        {@k_weight, c * c, "key weight"},
        {@v_weight, c * c, "value weight"},
        {@v_bias, c, "value bias"},
        {@o_weight, c * c, "output weight"},
        {@o_bias, c, "output bias"},
        {@layer_scale1, c, "layer scale 1"},
        {@norm2_weight, c, "norm2 weight"},
        {@norm2_bias, c, "norm2 bias"},
        {@up_weight, i * c, "MLP up weight"},
        {@up_bias, i, "MLP up bias"},
        {@down_weight, c * i, "MLP down weight"},
        {@down_bias, c, "MLP down bias"},
        {@layer_scale2, c, "layer scale 2"},
      }
      total_elements = 0_i64
      expected.each do |entry|
        values = entry[0]
        expected_length = entry[1]
        name = entry[2]
        unless values.size.to_i64 == expected_length
          raise BlockError.new(
            "DINOv3 block #{name} has #{values.size} elements, expected #{expected_length}"
          )
        end
        values.each_with_index do |value, index|
          unless value.finite?
            raise BlockError.new("DINOv3 block #{name}[#{index}] must be finite")
          end
        end
        total_elements += expected_length
      end
      total_bytes = total_elements * 4_i64
      if total_bytes > MAX_PARAMETER_BYTES
        raise BlockBudgetError.new(
          "DINOv3 block parameters require #{total_bytes} bytes, limit is #{MAX_PARAMETER_BYTES}"
        )
      end
    end

    def f32le_sha256 : String
      digest = Digest::SHA256.new
      bytes = Bytes.new(4, 0_u8)
      parameter_arrays.each do |values|
        values.each do |value|
          IO::ByteFormat::LittleEndian.encode(value, bytes)
          digest.update(bytes)
        end
      end
      digest.final.hexstring
    end

    private def parameter_arrays : Array(Array(Float32))
      [
        @norm1_weight,
        @norm1_bias,
        @q_weight,
        @q_bias,
        @k_weight,
        @v_weight,
        @v_bias,
        @o_weight,
        @o_bias,
        @layer_scale1,
        @norm2_weight,
        @norm2_bias,
        @up_weight,
        @up_bias,
        @down_weight,
        @down_bias,
        @layer_scale2,
      ]
    end
  end

  record BlockTrace,
    input : Tensor,
    norm1 : Tensor,
    q_heads : Tensor,
    k_heads : Tensor,
    v_heads : Tensor,
    q_rope : Tensor,
    k_rope : Tensor,
    scores : Tensor,
    probabilities : Tensor,
    attention_context : Tensor,
    output_projection : Tensor,
    layer_scale1 : Tensor,
    first_residual : Tensor,
    norm2 : Tensor,
    mlp_up : Tensor,
    exact_gelu : Tensor,
    mlp_down : Tensor,
    layer_scale2 : Tensor,
    block_output : Tensor,
    extractor_final : Tensor,
    parameter_f32le_sha256 : String,
    trellis_extractor_sha256 : String,
    source_revision : String,
    transformers_modeling_sha256 : String,
    transformers_config_sha256 : String

  class BlockCPU
    MAX_TOKENS  = 64_i64
    MAX_PATCHES = 32_i64
    # Secondary defense-in-depth. The structural caps above and in BlockConfig
    # are the active synthetic envelope, not a measured production budget.
    MAX_SCORE_ELEMENTS = 1_i64 * 1024_i64 * 1024_i64
    MAX_OUTPUT_BYTES   = 64_i64 * 1024_i64 * 1024_i64
    MAX_MULTIPLY_ADDS  = 64_i64 * 1024_i64 * 1024_i64

    getter config : BlockConfig
    getter parameters : BlockParameters

    def initialize(@parameters : BlockParameters)
      @config = @parameters.config
    end

    def forward_with_trace(
      input : Tensor,
      rope_cos : Tensor,
      rope_sin : Tensor,
    ) : BlockTrace
      token_count = validate_input!(input)
      patch_count = validate_rope!(rope_cos, rope_sin, token_count)
      preflight!(token_count, patch_count)

      # The owner contract above permits no concurrent mutation. Revalidate and
      # digest immediately before reading any retained parameter values.
      @parameters.validate!
      parameter_sha = @parameters.f32le_sha256

      # Tensor CPUReadView is a live borrow. Snapshot the bounded mutable
      # boundary so concurrent tensor writes cannot change one forward midway.
      input_values = snapshot_finite!(input.cpu_read, "input")
      rope_cos_values = snapshot_finite!(rope_cos.cpu_read, "rope cos")
      rope_sin_values = snapshot_finite!(rope_sin.cpu_read, "rope sin")

      input_tensor = tensor_from_values(input_values, input.shape.to_a)
      norm1_tensor = affine_layer_norm(
        input_values,
        token_count,
        @config.hidden_size,
        @parameters.norm1_weight,
        @parameters.norm1_bias,
        @config.layer_norm_eps
      )
      norm1_values = norm1_tensor.cpu_data.not_nil!

      q_values = linear(norm1_values, token_count, @config.hidden_size,
        @parameters.q_weight, @parameters.q_bias,
        @config.hidden_size)
      k_values = linear(norm1_values, token_count, @config.hidden_size,
        @parameters.k_weight, nil,
        @config.hidden_size)
      v_values = linear(norm1_values, token_count, @config.hidden_size,
        @parameters.v_weight, @parameters.v_bias,
        @config.hidden_size)
      q_heads_values = split_heads(q_values, token_count)
      k_heads_values = split_heads(k_values, token_count)
      v_heads_values = split_heads(v_values, token_count)
      q_heads_tensor = tensor_from_values(q_heads_values, [1_i32, @config.num_attention_heads, token_count, @config.head_dim])
      k_heads_tensor = tensor_from_values(k_heads_values, [1_i32, @config.num_attention_heads, token_count, @config.head_dim])
      v_heads_tensor = tensor_from_values(v_heads_values, [1_i32, @config.num_attention_heads, token_count, @config.head_dim])

      q_rope_values = apply_rope(q_heads_values, rope_cos_values, rope_sin_values, token_count, patch_count)
      k_rope_values = apply_rope(k_heads_values, rope_cos_values, rope_sin_values, token_count, patch_count)
      q_rope_tensor = tensor_from_values(q_rope_values, [1_i32, @config.num_attention_heads, token_count, @config.head_dim])
      k_rope_tensor = tensor_from_values(k_rope_values, [1_i32, @config.num_attention_heads, token_count, @config.head_dim])

      scores_values = attention_scores(q_rope_values, k_rope_values, token_count)
      scores_tensor = tensor_from_values(scores_values, [1_i32, @config.num_attention_heads, token_count, token_count])
      probabilities_values = stable_softmax(scores_values, token_count)
      probabilities_tensor = tensor_from_values(probabilities_values, [1_i32, @config.num_attention_heads, token_count, token_count])

      context_values = attention_context(probabilities_values, v_heads_values, token_count)
      context_tensor = tensor_from_values(context_values, [1_i32, token_count, @config.num_attention_heads, @config.head_dim])
      context_flat_values = flatten_context(context_values, token_count)
      output_projection_values = linear(context_flat_values, token_count, @config.hidden_size,
        @parameters.o_weight, @parameters.o_bias,
        @config.hidden_size)
      output_projection_tensor = tensor_from_values(output_projection_values, [1_i32, token_count, @config.hidden_size])

      layer_scale1_values = feature_scale(output_projection_values, @parameters.layer_scale1, token_count)
      layer_scale1_tensor = tensor_from_values(layer_scale1_values, [1_i32, token_count, @config.hidden_size])
      first_residual_values = add_values(layer_scale1_values, input_values, token_count * @config.hidden_size)
      first_residual_tensor = tensor_from_values(first_residual_values, [1_i32, token_count, @config.hidden_size])

      norm2_tensor = affine_layer_norm(
        first_residual_values,
        token_count,
        @config.hidden_size,
        @parameters.norm2_weight,
        @parameters.norm2_bias,
        @config.layer_norm_eps
      )
      norm2_values = norm2_tensor.cpu_data.not_nil!
      mlp_up_values = linear(norm2_values, token_count, @config.hidden_size,
        @parameters.up_weight, @parameters.up_bias,
        @config.intermediate_size)
      mlp_up_tensor = tensor_from_values(mlp_up_values, [1_i32, token_count, @config.intermediate_size])
      exact_gelu_values = exact_gelu(mlp_up_values)
      exact_gelu_tensor = tensor_from_values(exact_gelu_values, [1_i32, token_count, @config.intermediate_size])
      mlp_down_values = linear(exact_gelu_values, token_count, @config.intermediate_size,
        @parameters.down_weight, @parameters.down_bias,
        @config.hidden_size)
      mlp_down_tensor = tensor_from_values(mlp_down_values, [1_i32, token_count, @config.hidden_size])
      layer_scale2_values = feature_scale(mlp_down_values, @parameters.layer_scale2, token_count)
      layer_scale2_tensor = tensor_from_values(layer_scale2_values, [1_i32, token_count, @config.hidden_size])
      block_output_values = add_values(layer_scale2_values, first_residual_values, token_count * @config.hidden_size)
      block_output_tensor = tensor_from_values(block_output_values, [1_i32, token_count, @config.hidden_size])
      extractor_final_tensor = non_affine_layer_norm(
        block_output_values,
        token_count,
        @config.hidden_size,
        @config.extractor_final_layer_norm_eps
      )

      # Zero-copy retention is safe only under the single-owner contract. This
      # second digest rejects ordinary in-flight mutation instead of certifying
      # a trace for parameters different from those observed at the start.
      unless @parameters.f32le_sha256 == parameter_sha
        raise BlockError.new("DINOv3 block parameters mutated during forward")
      end

      BlockTrace.new(
        input_tensor,
        norm1_tensor,
        q_heads_tensor,
        k_heads_tensor,
        v_heads_tensor,
        q_rope_tensor,
        k_rope_tensor,
        scores_tensor,
        probabilities_tensor,
        context_tensor,
        output_projection_tensor,
        layer_scale1_tensor,
        first_residual_tensor,
        norm2_tensor,
        mlp_up_tensor,
        exact_gelu_tensor,
        mlp_down_tensor,
        layer_scale2_tensor,
        block_output_tensor,
        extractor_final_tensor,
        parameter_sha,
        TRELLIS2_EXTRACTOR_SHA256,
        TRELLIS2_SOURCE_REVISION,
        TRANSFORMERS_MODELING_SHA256,
        TRANSFORMERS_CONFIG_SHA256
      )
    rescue ex : OverflowError
      raise BlockError.new(
        "DINOv3 block intermediate arithmetic overflow: #{ex.message}"
      )
    end

    private def validate_input!(input : Tensor) : Int32
      unless input.on_cpu?
        raise BlockError.new("DINOv3 block reference accepts CPU tensors only")
      end
      unless input.dtype.f32?
        raise BlockError.new("DINOv3 block input must use F32")
      end
      unless input.contiguous?
        raise BlockError.new("DINOv3 block input must be contiguous")
      end
      shape = input.shape
      unless shape.ndim == 3 && shape[0] == 1 && shape[2] == @config.hidden_size
        raise BlockError.new(
          "DINOv3 block input must have shape [1, T, #{@config.hidden_size}], got #{shape}"
        )
      end
      token_count = shape[1]
      unless 1 <= token_count <= MAX_TOKENS
        raise BlockError.new("DINOv3 block token count must be in 1..#{MAX_TOKENS}")
      end
      token_count
    end

    private def validate_rope!(rope_cos : Tensor, rope_sin : Tensor, token_count : Int32) : Int32
      prefix_count = 1 + @config.num_register_tokens
      patch_count = token_count - prefix_count
      unless patch_count > 0 && patch_count <= MAX_PATCHES
        raise BlockError.new(
          "DINOv3 block patch count #{patch_count} must be in 1..#{MAX_PATCHES} " \
          "after prefix count #{prefix_count}"
        )
      end
      { {rope_cos, "rope cos"}, {rope_sin, "rope sin"} }.each do |entry|
        tensor = entry[0]
        name = entry[1]
        unless tensor.on_cpu?
          raise BlockError.new("DINOv3 block #{name} must be on CPU")
        end
        unless tensor.dtype.f32?
          raise BlockError.new("DINOv3 block #{name} must use F32")
        end
        unless tensor.contiguous?
          raise BlockError.new("DINOv3 block #{name} must be contiguous")
        end
        unless tensor.shape.ndim == 2 && tensor.shape[0] == patch_count && tensor.shape[1] == @config.head_dim
          raise BlockError.new(
            "DINOv3 block #{name} must have shape [#{patch_count}, #{@config.head_dim}], got #{tensor.shape}"
          )
        end
      end
      patch_count
    end

    private def preflight!(token_count : Int32, patch_count : Int32) : Nil
      heads = @config.num_attention_heads.to_i64
      tokens = token_count.to_i64
      hidden = @config.hidden_size.to_i64
      intermediate = @config.intermediate_size.to_i64
      score_elements = heads * tokens * tokens
      if score_elements > MAX_SCORE_ELEMENTS
        raise BlockBudgetError.new(
          "DINOv3 block scores require #{score_elements} elements, limit is #{MAX_SCORE_ELEMENTS}"
        )
      end
      trace_elements = 17_i64 * tokens * hidden + 2_i64 * tokens * intermediate + 2_i64 * score_elements
      trace_bytes = trace_elements * 4_i64
      if trace_bytes > MAX_OUTPUT_BYTES
        raise BlockBudgetError.new(
          "DINOv3 block trace requires #{trace_bytes} bytes, limit is #{MAX_OUTPUT_BYTES}"
        )
      end
      multiply_adds = 4_i64 * tokens * hidden * hidden +
                      2_i64 * tokens * hidden * intermediate +
                      2_i64 * tokens * tokens * hidden
      if multiply_adds > MAX_MULTIPLY_ADDS
        raise BlockBudgetError.new(
          "DINOv3 block multiply-add budget #{multiply_adds} exceeds #{MAX_MULTIPLY_ADDS}"
        )
      end
      # Keep the argument live in the arithmetic so malformed callers cannot
      # bypass the patch bound before any output tensor is allocated.
      raise BlockError.new("DINOv3 block patch/token boundary mismatch") unless patch_count == token_count - (1 + @config.num_register_tokens)
    end

    private def validate_finite!(values : Indexable(Float32), name : String) : Nil
      values.each_with_index do |value, index|
        unless value.finite?
          raise BlockError.new("DINOv3 block #{name}[#{index}] must be finite")
        end
      end
    end

    private def snapshot_finite!(values : Tensor::CPUReadView, name : String) : Array(Float32)
      snapshot = Array(Float32).new(values.size) { |index| values[index] }
      validate_finite!(snapshot, name)
      snapshot
    end

    private def tensor_from_values(values : Array(Float32), dimensions : Array(Int32)) : Tensor
      validate_finite!(values, "intermediate")
      Tensor.from_array(values, Shape.new(dimensions))
    end

    private def affine_layer_norm(
      values : Indexable(Float32),
      rows : Int32,
      width : Int32,
      weight : Array(Float32),
      bias : Array(Float32),
      eps : Float32,
    ) : Tensor
      result = Array(Float32).new(rows * width, 0.0_f32)
      rows.times do |row|
        offset = row * width
        sum = 0.0_f64
        width.times { |index| sum += values[offset + index].to_f64 }
        mean = sum / width.to_f64
        squared = 0.0_f64
        width.times do |index|
          delta = values[offset + index].to_f64 - mean
          squared += delta * delta
        end
        inv_std = 1.0_f64 / Math.sqrt(squared / width.to_f64 + eps.to_f64)
        width.times do |index|
          normalized = (values[offset + index].to_f64 - mean) * inv_std
          result[offset + index] = (normalized * weight[index].to_f64 + bias[index].to_f64).to_f32
        end
      end
      tensor_from_values(result, [1_i32, rows, width])
    end

    private def non_affine_layer_norm(
      values : Indexable(Float32),
      rows : Int32,
      width : Int32,
      eps : Float32,
    ) : Tensor
      result = Array(Float32).new(rows * width, 0.0_f32)
      rows.times do |row|
        offset = row * width
        sum = 0.0_f64
        width.times { |index| sum += values[offset + index].to_f64 }
        mean = sum / width.to_f64
        squared = 0.0_f64
        width.times do |index|
          delta = values[offset + index].to_f64 - mean
          squared += delta * delta
        end
        inv_std = 1.0_f64 / Math.sqrt(squared / width.to_f64 + eps.to_f64)
        width.times do |index|
          result[offset + index] = ((values[offset + index].to_f64 - mean) * inv_std).to_f32
        end
      end
      tensor_from_values(result, [1_i32, rows, width])
    end

    private def linear(
      values : Indexable(Float32),
      rows : Int32,
      input_width : Int32,
      weight : Array(Float32),
      bias : Array(Float32)?,
      output_width : Int32,
    ) : Array(Float32)
      result = Array(Float32).new(rows * output_width, 0.0_f32)
      rows.times do |row|
        input_offset = row * input_width
        output_offset = row * output_width
        output_width.times do |out_index|
          sum = bias ? bias.not_nil![out_index].to_f64 : 0.0_f64
          input_width.times do |in_index|
            sum += values[input_offset + in_index].to_f64 * weight[out_index * input_width + in_index].to_f64
          end
          result[output_offset + out_index] = sum.to_f32
        end
      end
      result
    end

    private def split_heads(values : Array(Float32), token_count : Int32) : Array(Float32)
      heads = @config.num_attention_heads
      dim = @config.head_dim
      result = Array(Float32).new(heads * token_count * dim, 0.0_f32)
      heads.times do |head|
        token_count.times do |token|
          dim.times do |index|
            result[(head * token_count + token) * dim + index] =
              values[token * @config.hidden_size + head * dim + index]
          end
        end
      end
      result
    end

    private def apply_rope(
      values : Array(Float32),
      rope_cos : Indexable(Float32),
      rope_sin : Indexable(Float32),
      token_count : Int32,
      patch_count : Int32,
    ) : Array(Float32)
      heads = @config.num_attention_heads
      dim = @config.head_dim
      half = dim // 2
      prefix = 1 + @config.num_register_tokens
      result = values.dup
      heads.times do |head|
        patch_count.times do |patch|
          token = prefix + patch
          base = (head * token_count + token) * dim
          rope_base = patch * dim
          dim.times do |index|
            rotated = index < half ? -values[base + half + index].to_f64 : values[base + index - half].to_f64
            result[base + index] = (
              values[base + index].to_f64 * rope_cos[rope_base + index].to_f64 +
              rotated * rope_sin[rope_base + index].to_f64
            ).to_f32
          end
        end
      end
      result
    end

    private def attention_scores(q : Array(Float32), k : Array(Float32), token_count : Int32) : Array(Float32)
      heads = @config.num_attention_heads
      dim = @config.head_dim
      scale = 1.0_f64 / Math.sqrt(dim.to_f64)
      result = Array(Float32).new(heads * token_count * token_count, 0.0_f32)
      heads.times do |head|
        token_count.times do |query|
          token_count.times do |key|
            sum = 0.0_f64
            dim.times do |index|
              q_index = (head * token_count + query) * dim + index
              k_index = (head * token_count + key) * dim + index
              sum += q[q_index].to_f64 * k[k_index].to_f64
            end
            result[(head * token_count + query) * token_count + key] = (sum * scale).to_f32
          end
        end
      end
      result
    end

    private def stable_softmax(scores : Array(Float32), token_count : Int32) : Array(Float32)
      heads = @config.num_attention_heads
      result = Array(Float32).new(scores.size, 0.0_f32)
      heads.times do |head|
        token_count.times do |query|
          offset = (head * token_count + query) * token_count
          maximum = scores[offset].to_f64
          token_count.times do |key|
            maximum = scores[offset + key].to_f64 if scores[offset + key].to_f64 > maximum
          end
          sum = 0.0_f64
          token_count.times do |key|
            value = Math.exp(scores[offset + key].to_f64 - maximum)
            result[offset + key] = value.to_f32
            sum += value
          end
          token_count.times do |key|
            result[offset + key] = (result[offset + key].to_f64 / sum).to_f32
          end
        end
      end
      result
    end

    private def attention_context(probabilities : Array(Float32), values : Array(Float32), token_count : Int32) : Array(Float32)
      heads = @config.num_attention_heads
      dim = @config.head_dim
      result = Array(Float32).new(token_count * heads * dim, 0.0_f32)
      token_count.times do |query|
        heads.times do |head|
          dim.times do |index|
            sum = 0.0_f64
            token_count.times do |key|
              probability = probabilities[(head * token_count + query) * token_count + key].to_f64
              value_index = (head * token_count + key) * dim + index
              sum += probability * values[value_index].to_f64
            end
            result[(query * heads + head) * dim + index] = sum.to_f32
          end
        end
      end
      result
    end

    private def flatten_context(values : Array(Float32), token_count : Int32) : Array(Float32)
      values.dup
    end

    private def feature_scale(values : Indexable(Float32), scale : Array(Float32), rows : Int32) : Array(Float32)
      result = Array(Float32).new(values.size, 0.0_f32)
      width = scale.size
      rows.times do |row|
        width.times do |index|
          result[row * width + index] = (values[row * width + index].to_f64 * scale[index].to_f64).to_f32
        end
      end
      result
    end

    private def add_values(left : Indexable(Float32), right : Indexable(Float32), count : Int32) : Array(Float32)
      result = Array(Float32).new(count, 0.0_f32)
      count.times do |index|
        result[index] = (left[index].to_f64 + right[index].to_f64).to_f32
      end
      result
    end

    private def exact_gelu(values : Array(Float32)) : Array(Float32)
      inverse_sqrt_two = 1.0_f64 / Math.sqrt(2.0_f64)
      values.map do |value|
        x = value.to_f64
        (0.5_f64 * x * (1.0_f64 + Math.erf(x * inverse_sqrt_two))).to_f32
      end
    end
  end
end
