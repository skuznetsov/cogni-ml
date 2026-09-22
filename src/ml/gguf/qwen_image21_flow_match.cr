# Qwen-Image 2.1's deterministic FlowMatch Euler schedule.
#
# This is the exact configured path used by the official pipeline: linear input
# sigmas, exponential resolution-dependent shifting, terminal stretching, and
# first-order Euler updates. Optional stochastic, Karras, beta, and per-token
# variants from the generic Diffusers scheduler are intentionally out of scope.

require "./qwen_image21_transformer"

module ML::GGUF
  struct QwenImage21FlowMatchConfig
    getter num_train_timesteps : Int32
    getter base_image_seq_len : Int32
    getter max_image_seq_len : Int32
    getter base_shift : Float32
    getter max_shift : Float32
    getter shift_terminal : Float32

    def initialize(
      @num_train_timesteps = 1000,
      @base_image_seq_len = 256,
      @max_image_seq_len = 8192,
      @base_shift = 0.5_f32,
      @max_shift = 0.9_f32,
      @shift_terminal = 0.02_f32,
    )
      raise ArgumentError.new("num_train_timesteps must be positive") unless @num_train_timesteps > 0
      unless @base_image_seq_len > 0 && @max_image_seq_len > @base_image_seq_len
        raise ArgumentError.new("image sequence bounds are invalid")
      end
      unless @shift_terminal >= 0.0_f32 && @shift_terminal < 1.0_f32
        raise ArgumentError.new("shift_terminal must be in [0, 1)")
      end
    end
  end

  class QwenImage21FlowMatchSchedule
    getter sigmas : Array(Float32)
    getter timesteps : Array(Float32)
    getter mu : Float32

    def initialize(@sigmas, @timesteps, @mu)
      unless @sigmas.size == @timesteps.size + 1
        raise ArgumentError.new("sigma schedule must include one terminal value")
      end
    end

    def step_count : Int32
      @timesteps.size
    end

    # The transformer consumes scheduler timestep / num_train_timesteps, which
    # is the shifted sigma at this step.
    def model_timestep(index : Int32) : Float32
      check_index(index)
      @sigmas[index]
    end

    def step(sample : Array(Float32), model_output : Array(Float32), index : Int32) : Array(Float32)
      check_index(index)
      unless sample.size == model_output.size
        raise ArgumentError.new("sample and model output sizes differ")
      end
      dt = @sigmas[index + 1] - @sigmas[index]
      Array(Float32).new(sample.size) do |value_index|
        sample[value_index] + dt * model_output[value_index]
      end
    end

    private def check_index(index : Int32) : Nil
      unless index >= 0 && index < step_count
        raise IndexError.new("flow step #{index} is outside 0...#{step_count}")
      end
    end
  end

  module QwenImage21FlowMatch
    def self.calculate_mu(
      image_seq_len : Int32,
      config : QwenImage21FlowMatchConfig = QwenImage21FlowMatchConfig.new,
    ) : Float32
      raise ArgumentError.new("image_seq_len must be positive") unless image_seq_len > 0
      slope = (config.max_shift - config.base_shift).to_f64 /
              (config.max_image_seq_len - config.base_image_seq_len)
      intercept = config.base_shift - slope * config.base_image_seq_len
      (image_seq_len * slope + intercept).to_f32
    end

    def self.schedule(
      num_inference_steps : Int32,
      image_seq_len : Int32,
      config : QwenImage21FlowMatchConfig = QwenImage21FlowMatchConfig.new,
    ) : QwenImage21FlowMatchSchedule
      unless num_inference_steps >= 2
        raise ArgumentError.new("num_inference_steps must be at least two")
      end
      mu = calculate_mu(image_seq_len, config)

      sigmas = Array(Float32).new(num_inference_steps) do |index|
        fraction = index.to_f64 / (num_inference_steps - 1)
        (1.0_f64 + fraction * (1.0_f64 / num_inference_steps - 1.0_f64)).to_f32
      end

      exponential = Math.exp(mu.to_f64)
      sigmas.map! do |sigma|
        (exponential / (exponential + (1.0_f64 / sigma - 1.0_f64))).to_f32
      end

      one_minus_last = 1.0_f64 - sigmas.last
      terminal_scale = one_minus_last / (1.0_f64 - config.shift_terminal)
      sigmas.map! do |sigma|
        (1.0_f64 - (1.0_f64 - sigma) / terminal_scale).to_f32
      end

      timesteps = sigmas.map { |sigma| sigma * config.num_train_timesteps }
      sigmas << 0.0_f32
      QwenImage21FlowMatchSchedule.new(sigmas, timesteps, mu)
    end

    def self.denoise(
      initial_latents : Array(Float32),
      schedule : QwenImage21FlowMatchSchedule,
      &predictor : Array(Float32), Float32, Int32 -> Array(Float32)
    ) : Array(Float32)
      latents = initial_latents.dup
      schedule.step_count.times do |index|
        model_output = yield latents, schedule.model_timestep(index), index
        latents = schedule.step(latents, model_output, index)
      end
      latents
    end
  end

  class QwenImage21DenoisingResult
    getter latents : Array(Float32)
    getter schedule : QwenImage21FlowMatchSchedule
    getter transformer_evaluations : Int32

    def initialize(@latents, @schedule, @transformer_evaluations)
    end
  end

  # Batch-one latent denoising driver for the admitted transformer reference.
  # The caller supplies VLM embeddings and optional condition-image latents;
  # target placeholder slots are appended exactly as in the official pipeline.
  module QwenImage21LatentDenoiser
    def self.run(
      initial_target_latents : Array(Float32),
      condition_latents : Array(Float32),
      encoder_hidden_states : Array(Float32),
      img_shapes : Array(StaticArray(Int32, 3)),
      encoder_img_mask : Array(Bool),
      weights : QwenImage21TransformerWeights,
      config : QwenImage21TransformerConfig,
      num_inference_steps = 40,
      encoder_hidden_states_mask : Array(Bool)? = nil,
      scheduler_config : QwenImage21FlowMatchConfig = QwenImage21FlowMatchConfig.new,
      backend : ComputeBackend = F32Backend.new,
    ) : QwenImage21DenoisingResult
      raise ArgumentError.new("img_shapes must contain a target image") if img_shapes.empty?
      unless config.input_dim == config.output_dim
        raise ArgumentError.new("denoising requires equal transformer input and output dimensions")
      end
      target_tokens = shape_tokens(img_shapes.last)
      condition_tokens = img_shapes.first(img_shapes.size - 1).sum { |shape| shape_tokens(shape) }
      unless initial_target_latents.size == target_tokens * config.input_dim
        raise ArgumentError.new("target latent size mismatch")
      end
      unless condition_latents.size == condition_tokens * config.input_dim
        raise ArgumentError.new("condition latent size mismatch")
      end
      unless encoder_hidden_states.size.divisible_by?(config.context_dim)
        raise ArgumentError.new("encoder hidden state size mismatch")
      end
      encoder_tokens = encoder_hidden_states.size // config.context_dim
      unless encoder_img_mask.size == encoder_tokens
        raise ArgumentError.new("encoder image mask size mismatch")
      end
      unless target_tokens.divisible_by?(QwenImage21TransformerCPU::IMG_TOKENS_PER_SLOT)
        raise ArgumentError.new("target image token count must be divisible by four")
      end

      transformer_mask = encoder_img_mask.dup
      (target_tokens // QwenImage21TransformerCPU::IMG_TOKENS_PER_SLOT).times do
        transformer_mask << true
      end
      schedule = QwenImage21FlowMatch.schedule(
        num_inference_steps, target_tokens, scheduler_config
      )
      evaluations = 0
      latents = QwenImage21FlowMatch.denoise(
        initial_target_latents, schedule
      ) do |target_latents, timestep, _index|
        result = QwenImage21TransformerCPU.forward(
          condition_latents + target_latents,
          encoder_hidden_states,
          timestep,
          img_shapes,
          transformer_mask,
          weights,
          config,
          encoder_hidden_states_mask: encoder_hidden_states_mask,
          backend: backend,
        )
        evaluations += 1
        result.output.last(target_tokens * config.output_dim)
      end
      QwenImage21DenoisingResult.new(latents, schedule, evaluations)
    end

    private def self.shape_tokens(shape : StaticArray(Int32, 3)) : Int32
      shape[0] * shape[1] * shape[2]
    end
  end
end
