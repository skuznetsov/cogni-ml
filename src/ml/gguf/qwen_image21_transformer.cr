# Exact batch-one CPU reference for the Qwen-Image 2.1 outer transformer.
#
# The implementation mirrors the upstream sequence construction, timestep and
# text projections, shared modulation, transformer-block loop, and final head.
# It is parameterized so small fixtures can be checked independently before the
# fixed-width execution path is made fully Metal-resident.

require "./qwen_image21_block"

module ML::GGUF
  struct QwenImage21TransformerConfig
    getter input_dim : Int32
    getter output_dim : Int32
    getter context_dim : Int32
    getter time_input_dim : Int32
    getter block : QwenImage21BlockConfig
    getter causal_condition : Bool

    def initialize(@input_dim, @output_dim, @context_dim, @time_input_dim,
                   @block, @causal_condition = true)
      raise ArgumentError.new("input_dim must be positive") unless @input_dim > 0
      raise ArgumentError.new("output_dim must be positive") unless @output_dim > 0
      raise ArgumentError.new("context_dim must be positive") unless @context_dim > 0
      raise ArgumentError.new("time_input_dim must be at least two") unless @time_input_dim >= 2
    end

    def hidden_dim : Int32
      @block.hidden_dim
    end
  end

  class QwenImage21TransformerWeights
    getter img_in : QuantWeight
    getter modulation : QuantWeight
    getter norm_out_linear : QuantWeight
    getter proj_out : QuantWeight
    getter timestep_linear_1 : QuantWeight
    getter timestep_linear_2 : QuantWeight
    getter text_in_layer : QuantWeight
    getter text_out_layer : QuantWeight
    getter text_norm : Array(Float32)
    getter layers : Array(QwenImage21BlockWeights)

    def initialize(@img_in, @modulation, @norm_out_linear, @proj_out,
                   @timestep_linear_1, @timestep_linear_2,
                   @text_in_layer, @text_out_layer, @text_norm, @layers)
    end
  end

  class QwenImage21TokenLayout
    getter image_pad_mask : Array(Bool)
    getter image_ids : Array(Int32)
    getter target_token_mask : Array(Bool)
    getter positions : Array(StaticArray(Int32, 3))
    getter key_valid : Array(Bool)

    def initialize(@image_pad_mask, @image_ids, @target_token_mask,
                   @positions, @key_valid)
    end

    def token_count : Int32
      @image_pad_mask.size
    end
  end

  class QwenImage21TransformerResult
    getter output : Array(Float32)
    getter layout : QwenImage21TokenLayout

    def initialize(@output, @layout)
    end
  end

  module QwenImage21TransformerCPU
    IMG_TOKENS_PER_SLOT = 4

    # `img_mask` spans encoder tokens followed by target placeholder slots.
    # Each true slot expands to a 2x2 group of latent tokens. `img_shapes`
    # lists condition images first and the target image last.
    def self.build_layout(
      img_mask : Array(Bool),
      img_shapes : Array(StaticArray(Int32, 3)),
      encoder_token_count : Int32,
      encoder_hidden_states_mask : Array(Bool)? = nil,
    ) : QwenImage21TokenLayout
      raise ArgumentError.new("img_shapes must contain a target image") if img_shapes.empty?
      img_shapes.each do |shape|
        raise ArgumentError.new("only single-frame image shapes are supported") unless shape[0] == 1
        raise ArgumentError.new("image height and width must be positive") unless shape[1] > 0 && shape[2] > 0
      end

      target_tokens = shape_tokens(img_shapes.last)
      unless target_tokens.divisible_by?(IMG_TOKENS_PER_SLOT)
        raise ArgumentError.new("target image token count must be divisible by four")
      end
      target_slots = target_tokens // IMG_TOKENS_PER_SLOT
      unless img_mask.size == encoder_token_count + target_slots
        raise ArgumentError.new(
          "img_mask has #{img_mask.size} entries, expected #{encoder_token_count + target_slots}"
        )
      end
      unless img_mask.last(target_slots).all?
        raise ArgumentError.new("target placeholder slots must be marked as image slots")
      end

      image_pad_mask = [] of Bool
      img_mask.each do |is_image|
        (is_image ? IMG_TOKENS_PER_SLOT : 1).times { image_pad_mask << is_image }
      end

      block_lengths = img_shapes.map { |shape| shape_tokens(shape) }
      image_positions = [] of Int32
      image_pad_mask.each_with_index { |is_image, index| image_positions << index if is_image }
      unless block_lengths.sum == image_positions.size
        raise ArgumentError.new(
          "img_shapes account for #{block_lengths.sum} image tokens but img_mask expands to #{image_positions.size}"
        )
      end

      image_ids = Array(Int32).new(image_pad_mask.size, -1)
      image_offset = 0
      block_lengths.each_with_index do |length, block_id|
        length.times do
          image_ids[image_positions[image_offset]] = block_id
          image_offset += 1
        end
      end
      target_token_mask = Array(Bool).new(image_pad_mask.size, false)
      target_tokens.times do |offset|
        target_token_mask[image_positions[image_positions.size - target_tokens + offset]] = true
      end

      positions = build_positions(image_pad_mask, img_shapes)
      key_valid = build_key_valid(
        image_pad_mask, img_mask, encoder_token_count, encoder_hidden_states_mask
      )
      QwenImage21TokenLayout.new(
        image_pad_mask, image_ids, target_token_mask, positions, key_valid
      )
    end

    def self.forward(
      hidden_states : Array(Float32),
      encoder_hidden_states : Array(Float32),
      timestep : Float32,
      img_shapes : Array(StaticArray(Int32, 3)),
      img_mask : Array(Bool),
      weights : QwenImage21TransformerWeights,
      config : QwenImage21TransformerConfig,
      encoder_hidden_states_mask : Array(Bool)? = nil,
      backend : ComputeBackend = F32Backend.new,
    ) : QwenImage21TransformerResult
      validate_weights(weights, config)
      image_token_count = img_shapes.sum { |shape| shape_tokens(shape) }
      unless hidden_states.size == image_token_count * config.input_dim
        raise ArgumentError.new("hidden_states size mismatch")
      end
      unless encoder_hidden_states.size.divisible_by?(config.context_dim)
        raise ArgumentError.new("encoder_hidden_states size mismatch")
      end
      encoder_token_count = encoder_hidden_states.size // config.context_dim
      layout = build_layout(
        img_mask, img_shapes, encoder_token_count, encoder_hidden_states_mask
      )

      projected_images = backend.matmul(
        hidden_states, image_token_count, weights.img_in, zeros(config.hidden_dim)
      )
      projected_text = project_text(
        encoder_hidden_states, encoder_token_count, weights, config, backend
      )
      joint = build_joint_hidden(
        projected_text, encoder_token_count, projected_images, img_mask,
        layout.image_pad_mask, config.hidden_dim
      )

      time_rows = config.causal_condition ? [timestep, 0.0_f32] : [timestep]
      temb = time_embedding(time_rows, config.time_input_dim)
      temb = backend.matmul(
        temb, time_rows.size, weights.timestep_linear_1, zeros(config.hidden_dim)
      )
      temb.map! { |value| silu(value) }
      temb = backend.matmul(
        temb, time_rows.size, weights.timestep_linear_2, zeros(config.hidden_dim)
      )

      modulation_input = temb.map { |value| silu(value) }
      modulation_rows = backend.matmul(
        modulation_input, time_rows.size, weights.modulation, zeros(4 * config.hidden_dim)
      )
      modulation = select_rows(
        modulation_rows, time_rows.size, 4 * config.hidden_dim,
        layout.target_token_mask, config.causal_condition
      )

      weights.layers.each do |layer|
        joint = QwenImage21BlockCPU.forward(
          joint,
          layout.token_count,
          modulation,
          layout.positions,
          layout.image_ids,
          layer,
          config.block,
          key_valid: layout.key_valid,
          backend: backend,
        )
      end

      scale_rows = backend.matmul(
        temb.map { |value| silu(value) },
        time_rows.size,
        weights.norm_out_linear,
        zeros(config.hidden_dim),
      )
      scales = select_rows(
        scale_rows, time_rows.size, config.hidden_dim,
        layout.target_token_mask, config.causal_condition
      )
      normalized = layer_norm(joint, layout.token_count, config.hidden_dim, config.block.eps)
      normalized.size.times { |index| normalized[index] *= 1.0_f32 + scales[index] }
      output = backend.matmul(
        normalized, layout.token_count, weights.proj_out, zeros(config.output_dim)
      )
      QwenImage21TransformerResult.new(output, layout)
    end

    private def self.project_text(
      input : Array(Float32), rows : Int32,
      weights : QwenImage21TransformerWeights,
      config : QwenImage21TransformerConfig,
      backend : ComputeBackend,
    ) : Array(Float32)
      normalized = zero_center_rms_norm(
        input, rows, config.context_dim, weights.text_norm, config.block.eps
      )
      projected = backend.matmul(
        normalized, rows, weights.text_in_layer, zeros(config.hidden_dim)
      )
      projected.map! { |value| backend.gelu(value) }
      backend.matmul(projected, rows, weights.text_out_layer, zeros(config.hidden_dim))
    end

    private def self.build_joint_hidden(
      projected_text : Array(Float32), encoder_tokens : Int32,
      projected_images : Array(Float32), img_mask : Array(Bool),
      image_pad_mask : Array(Bool), dim : Int32,
    ) : Array(Float32)
      base = projected_text.dup
      (img_mask.size - encoder_tokens).times { dim.times { base << 0.0_f32 } }
      joint = Array(Float32).new(image_pad_mask.size * dim, 0.0_f32)
      joint_row = 0
      img_mask.each_with_index do |is_image, base_row|
        repeat = is_image ? IMG_TOKENS_PER_SLOT : 1
        repeat.times do
          dim.times { |column| joint[joint_row * dim + column] = base[base_row * dim + column] }
          joint_row += 1
        end
      end

      image_row = 0
      image_pad_mask.each_with_index do |is_image, row|
        next unless is_image
        dim.times { |column| joint[row * dim + column] = projected_images[image_row * dim + column] }
        image_row += 1
      end
      joint
    end

    private def self.build_positions(
      image_pad_mask : Array(Bool),
      img_shapes : Array(StaticArray(Int32, 3)),
    ) : Array(StaticArray(Int32, 3))
      positions = [] of StaticArray(Int32, 3)
      cursor = 0
      position = 0
      img_shapes.each do |shape|
        height = shape[1]
        width = shape[2]
        block_start = cursor
        while block_start < image_pad_mask.size && !image_pad_mask[block_start]
          block_start += 1
        end
        raise ArgumentError.new("img_mask does not contain the declared image blocks") if block_start == image_pad_mask.size

        text_len = block_start - cursor
        text_len.times do |offset|
          value = position + offset
          positions << StaticArray[value, value, value]
        end
        position += text_len
        (-(height - height // 2)...(height // 2)).each do |height_index|
          (-(width - width // 2)...(width // 2)).each do |width_index|
            positions << StaticArray[position, height_index, width_index]
          end
        end
        cursor = block_start + height * width
        position += Math.max(height, width)
      end
      if cursor < image_pad_mask.size
        (image_pad_mask.size - cursor).times do |offset|
          value = position + offset
          positions << StaticArray[value, value, value]
        end
      end
      unless positions.size == image_pad_mask.size
        raise ArgumentError.new("image shapes do not align with expanded img_mask")
      end
      positions
    end

    private def self.build_key_valid(
      image_pad_mask : Array(Bool), img_mask : Array(Bool),
      encoder_token_count : Int32,
      encoder_hidden_states_mask : Array(Bool)?,
    ) : Array(Bool)
      key_valid = Array(Bool).new(image_pad_mask.size, true)
      return key_valid unless encoder_hidden_states_mask
      unless encoder_hidden_states_mask.size == encoder_token_count
        raise ArgumentError.new("encoder_hidden_states_mask size mismatch")
      end

      text_positions = [] of Int32
      image_pad_mask.each_with_index { |is_image, index| text_positions << index unless is_image }
      text_valid = [] of Bool
      encoder_token_count.times do |index|
        text_valid << encoder_hidden_states_mask[index] unless img_mask[index]
      end
      unless text_positions.size == text_valid.size
        raise ArgumentError.new("encoder text positions do not align with expanded img_mask")
      end
      text_positions.each_with_index { |position, index| key_valid[position] = text_valid[index] }
      key_valid
    end

    private def self.time_embedding(timesteps : Array(Float32), dim : Int32) : Array(Float32)
      half = dim // 2
      output = Array(Float32).new(timesteps.size * dim, 0.0_f32)
      timesteps.each_with_index do |timestep, row|
        half.times do |index|
          frequency = Math.exp(-Math.log(10_000.0_f64) * index / half)
          angle = 1000.0_f64 * timestep * frequency
          output[row * dim + index] = Math.cos(angle).to_f32
          output[row * dim + half + index] = Math.sin(angle).to_f32
        end
      end
      output
    end

    private def self.select_rows(
      rows : Array(Float32), row_count : Int32, width : Int32,
      target_mask : Array(Bool), causal_condition : Bool,
    ) : Array(Float32)
      unless rows.size == row_count * width
        raise ArgumentError.new("modulation row size mismatch")
      end
      selected = Array(Float32).new(target_mask.size * width, 0.0_f32)
      target_mask.each_with_index do |is_target, token|
        source_row = causal_condition && !is_target ? row_count - 1 : 0
        width.times do |column|
          selected[token * width + column] = rows[source_row * width + column]
        end
      end
      selected
    end

    private def self.zero_center_rms_norm(
      input : Array(Float32), rows : Int32, dim : Int32,
      weight : Array(Float32), eps : Float32,
    ) : Array(Float32)
      output = Array(Float32).new(input.size, 0.0_f32)
      rows.times do |row|
        offset = row * dim
        mean_square = 0.0_f64
        dim.times { |column| mean_square += input[offset + column].to_f64 ** 2 }
        inv_rms = 1.0_f64 / Math.sqrt(mean_square / dim + eps)
        dim.times do |column|
          output[offset + column] = (
            input[offset + column] * inv_rms * (weight[column] + 1.0_f32)
          ).to_f32
        end
      end
      output
    end

    private def self.layer_norm(
      input : Array(Float32), rows : Int32, dim : Int32, eps : Float32,
    ) : Array(Float32)
      output = Array(Float32).new(input.size, 0.0_f32)
      rows.times do |row|
        offset = row * dim
        mean = 0.0_f64
        dim.times { |column| mean += input[offset + column] }
        mean /= dim
        variance = 0.0_f64
        dim.times do |column|
          delta = input[offset + column] - mean
          variance += delta * delta
        end
        inv_std = 1.0_f64 / Math.sqrt(variance / dim + eps)
        dim.times do |column|
          output[offset + column] = ((input[offset + column] - mean) * inv_std).to_f32
        end
      end
      output
    end

    private def self.validate_weights(
      weights : QwenImage21TransformerWeights,
      config : QwenImage21TransformerConfig,
    ) : Nil
      hidden = config.hidden_dim
      validate_projection(weights.img_in, config.input_dim, hidden, "img_in")
      validate_projection(weights.modulation, hidden, 4 * hidden, "modulation")
      validate_projection(weights.norm_out_linear, hidden, hidden, "norm_out.linear")
      validate_projection(weights.proj_out, hidden, config.output_dim, "proj_out")
      validate_projection(weights.timestep_linear_1, config.time_input_dim, hidden, "timestep linear 1")
      validate_projection(weights.timestep_linear_2, hidden, hidden, "timestep linear 2")
      validate_projection(weights.text_in_layer, config.context_dim, hidden, "text in")
      validate_projection(weights.text_out_layer, hidden, hidden, "text out")
      raise ArgumentError.new("text_norm size mismatch") unless weights.text_norm.size == config.context_dim
    end

    private def self.validate_projection(
      weight : QuantWeight, expected_in : Int32, expected_out : Int32, label : String,
    ) : Nil
      unless weight.in_dim == expected_in && weight.out_dim == expected_out
        raise ArgumentError.new("#{label} weight shape mismatch")
      end
    end

    private def self.shape_tokens(shape : StaticArray(Int32, 3)) : Int32
      shape[0] * shape[1] * shape[2]
    end

    private def self.zeros(size : Int32) : Array(Float32)
      Array(Float32).new(size, 0.0_f32)
    end

    @[AlwaysInline]
    private def self.silu(value : Float32) : Float32
      (value / (1.0_f32 + Math.exp(-value))).to_f32
    end
  end
end
