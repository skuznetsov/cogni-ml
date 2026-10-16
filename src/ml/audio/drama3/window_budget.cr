require "json"
require "./backbone"
require "./tower"
require "./codec"

module ML::Audio::Drama3
  # Advisory admission guard for the resident, all-GPU DRAMA-3 path. These are
  # advisory allocation estimates, not reservations or physical peak bounds.
  # Ordinary host temporaries may remain live until Crystal GC runs, and driver
  # allocations can change after the snapshot; neither is certified here.
  module Drama3WindowBudget
    extend self

    MAX_TOTAL_POSITIONS  =   4096
    GRAPH_DEPTH          =      2
    FLOAT_BYTES          =  4_i64
    HOST_RESERVE_PERCENT = 15_i64

    enum Failure
      UnknownTelemetry
      InvalidTelemetry
      InvalidRequest
      PromptExceedsPositionCap
      InsufficientFrames
      InsufficientBudget
      PerBufferLimit
      Overflow
    end

    # Nullable fields let callers represent unavailable OS/Metal observations.
    # An incomplete snapshot is rejected; zero is not substituted for unknown.
    struct Telemetry
      getter physical_bytes : Int64?
      getter free_reclaimable_bytes : Int64?
      getter metal_recommended_bytes : Int64?
      getter metal_current_allocated_bytes : Int64?
      getter max_buffer_length : Int64?

      def initialize(
        @physical_bytes : Int64? = nil,
        @free_reclaimable_bytes : Int64? = nil,
        @metal_recommended_bytes : Int64? = nil,
        @metal_current_allocated_bytes : Int64? = nil,
        @max_buffer_length : Int64? = nil,
      )
      end
    end

    # The MoE widths must come from the pinned checkpoint's text_config. They
    # cannot be reconstructed safely from hidden size or expert count.
    struct Request
      getter backbone_config : BackboneConfig
      getter tower_config : TowerConfig
      getter codec_config : CodecConfig
      getter prompt_frames : Int32
      getter min_frames : Int32
      getter frame_cap : Int32
      getter checkpoint_mapped_bytes : Int64
      getter largest_checkpoint_mapped_buffer_bytes : Int64
      getter converted_static_bytes : Int64
      getter largest_converted_buffer_bytes : Int64
      getter additional_margin_bytes : Int64
      getter moe_intermediate_size : Int32?
      getter shared_expert_intermediate_size : Int32?
      getter reference_encoder_peak_bytes : Int64?

      def initialize(
        @backbone_config : BackboneConfig,
        @tower_config : TowerConfig,
        @codec_config : CodecConfig,
        @prompt_frames : Int32,
        @min_frames : Int32,
        @frame_cap : Int32,
        @checkpoint_mapped_bytes : Int64,
        @largest_checkpoint_mapped_buffer_bytes : Int64,
        @converted_static_bytes : Int64,
        @largest_converted_buffer_bytes : Int64,
        @additional_margin_bytes : Int64,
        @moe_intermediate_size : Int32?,
        @shared_expert_intermediate_size : Int32?,
        @reference_encoder_peak_bytes : Int64? = nil,
      )
      end
    end

    struct Breakdown
      include JSON::Serializable

      getter checkpoint_mapped_bytes : Int64
      getter converted_static_bytes : Int64
      getter backbone_cache_bytes : Int64
      getter tower_cache_bytes : Int64
      getter recurrent_state_bytes : Int64
      getter generation_buffers_bytes : Int64
      getter decoder_scratch_bytes : Int64
      # Subset already included in codec_temporaries_bytes; do not add it again.
      getter codec_weightnorm_temporaries_included_bytes : Int64
      getter codec_temporaries_bytes : Int64
      getter generation_temporaries_bytes : Int64
      getter reference_encoder_peak_bytes : Int64
      getter estimated_total_bytes : Int64
      getter largest_buffer_bytes : Int64

      def initialize(
        @checkpoint_mapped_bytes : Int64,
        @converted_static_bytes : Int64,
        @backbone_cache_bytes : Int64,
        @tower_cache_bytes : Int64,
        @recurrent_state_bytes : Int64,
        @generation_buffers_bytes : Int64,
        @decoder_scratch_bytes : Int64,
        @codec_weightnorm_temporaries_included_bytes : Int64,
        @codec_temporaries_bytes : Int64,
        @generation_temporaries_bytes : Int64,
        @reference_encoder_peak_bytes : Int64,
        @estimated_total_bytes : Int64,
        @largest_buffer_bytes : Int64,
      )
      end
    end

    struct Plan
      include JSON::Serializable

      getter generated_frames : Int32
      getter total_positions : Int32
      getter effective_frame_cap : Int32
      getter estimated_bytes : Int64
      getter budget_bytes : Int64
      getter host_budget_bytes : Int64
      getter metal_budget_bytes : Int64
      getter host_reserved_bytes : Int64
      getter additional_margin_bytes : Int64
      getter max_buffer_length : Int64
      getter breakdown : Breakdown
      getter advisory_only : Bool
      getter host_gc_retention_certified : Bool

      def initialize(
        @generated_frames : Int32,
        @total_positions : Int32,
        @effective_frame_cap : Int32,
        @estimated_bytes : Int64,
        @budget_bytes : Int64,
        @host_budget_bytes : Int64,
        @metal_budget_bytes : Int64,
        @host_reserved_bytes : Int64,
        @additional_margin_bytes : Int64,
        @max_buffer_length : Int64,
        @breakdown : Breakdown,
      )
        @advisory_only = true
        @host_gc_retention_certified = false
      end
    end

    struct PlanResult
      getter plan : Plan?
      getter failure : Failure?

      def initialize(@plan : Plan?, @failure : Failure?)
      end

      def success? : Bool
        !!@plan
      end

      def plan? : Plan?
        @plan
      end
    end

    private class Rejected < Exception
      getter failure : Failure

      def initialize(@failure : Failure)
      end
    end

    # Choose the largest generated-frame count admitted by both independent
    # headroom checks, the explicit frame cap, the 4096-position ceiling and
    # every known single-buffer limit. No allocator or device API is called.
    def experimental_plan(request : Request, telemetry : Telemetry) : PlanResult
      begin
        validate_request!(request)
        physical, reclaimable, recommended, allocated, max_buffer = read_telemetry!(telemetry)

        host_reserved = ceil_percent(physical, HOST_RESERVE_PERCENT)
        host_budget = subtract_budget(reclaimable, host_reserved, request.additional_margin_bytes)
        metal_budget = recommended - allocated
        budget = Math.min(host_budget, metal_budget)
        raise Rejected.new(Failure::InsufficientBudget) unless budget > 0

        max_positions = MAX_TOTAL_POSITIONS - request.prompt_frames + 1
        max_frames = Math.min(request.frame_cap.to_i64, max_positions.to_i64).to_i32
        raise Rejected.new(Failure::InsufficientFrames) if request.min_frames > max_frames

        min_estimate = estimate(request, request.min_frames)
        unless fits?(min_estimate, budget, max_buffer)
          failure = min_estimate.largest_buffer_bytes > max_buffer ? Failure::PerBufferLimit : Failure::InsufficientBudget
          raise Rejected.new(failure)
        end

        max_estimate = estimate(request, max_frames)
        chosen_frames, chosen_estimate = if fits?(max_estimate, budget, max_buffer)
                                           {max_frames, max_estimate}
                                         else
                                           search_largest(request, request.min_frames, max_frames, budget, max_buffer, min_estimate)
                                         end
        positions = request.prompt_frames + chosen_frames - 1
        plan = Plan.new(
          chosen_frames,
          positions,
          max_frames,
          chosen_estimate.estimated_total_bytes,
          budget,
          host_budget,
          metal_budget,
          host_reserved,
          request.additional_margin_bytes,
          max_buffer,
          chosen_estimate,
        )
        PlanResult.new(plan, nil)
      rescue ex : Rejected
        PlanResult.new(nil, ex.failure)
      rescue OverflowError
        PlanResult.new(nil, Failure::Overflow)
      end
    end

    private def search_largest(request : Request, lower : Int32, upper : Int32,
                               budget : Int64, max_buffer : Int64,
                               lower_estimate : Breakdown) : Tuple(Int32, Breakdown)
      best_frames = lower
      best_estimate = lower_estimate
      low = lower.to_i64 + 1
      high = upper.to_i64
      while low <= high
        middle = low + (high - low) // 2
        candidate = estimate(request, middle.to_i32)
        if fits?(candidate, budget, max_buffer)
          best_frames = middle.to_i32
          best_estimate = candidate
          low = middle + 1
        else
          high = middle - 1
        end
      end
      {best_frames, best_estimate}
    end

    private def fits?(estimate : Breakdown, budget : Int64, max_buffer : Int64) : Bool
      estimate.estimated_total_bytes <= budget && estimate.largest_buffer_bytes <= max_buffer
    end

    private def estimate(request : Request, frames : Int32) : Breakdown
      prompt = request.prompt_frames.to_i64
      generated = frames.to_i64
      positions = checked_add(prompt, generated) - 1
      backbone = request.backbone_config
      tower = request.tower_config
      codec = request.codec_config
      moe_width = request.moe_intermediate_size.not_nil!.to_i64
      shared_width = request.shared_expert_intermediate_size.not_nil!.to_i64
      full_layers = (backbone.layers // backbone.full_interval).to_i64
      linear_layers = backbone.layers.to_i64 - full_layers
      bytes = FLOAT_BYTES

      # Persistent resident KV, full-attention score, convolution and recurrent
      # state allocations. The cache capacity is prompt + generated - 1.
      backbone_cache = checked_bytes(
        positions, backbone.kv_heads.to_i64, backbone.head_dim.to_i64,
        2_i64, full_layers, bytes,
      )
      full_score_buffer = if positions > 1024 && full_layers > 0
                            checked_bytes(positions, backbone.heads.to_i64, bytes)
                          else
                            0_i64
                          end
      backbone_cache = checked_add(backbone_cache, full_score_buffer)

      qkv_width = checked_add(
        checked_product(2_i64, backbone.key_heads.to_i64, backbone.key_dim.to_i64),
        checked_product(backbone.value_heads.to_i64, backbone.value_dim.to_i64),
      )
      recurrent = checked_bytes(
        linear_layers, backbone.conv_kernel.to_i64, qkv_width, bytes,
      )
      recurrent = checked_add(recurrent, checked_bytes(
        linear_layers, backbone.value_heads.to_i64, backbone.value_dim.to_i64,
        backbone.key_dim.to_i64, bytes,
      ))

      tower_cache = checked_bytes(
        2_i64, tower.layer_count.to_i64, positions, tower.n_local_heads.to_i64,
        tower.head_dim.to_i64, bytes,
      )

      # Persistent prompt/generated output, random draws and repeat-avoidance
      # window. Prompt rows are uploaded separately but summed here.
      frame_width = tower.num_codebooks.to_i64 + 1
      draw_width = frame_width
      generation_buffers = checked_bytes(generated, frame_width, bytes)
      generation_buffers = checked_add(generation_buffers, checked_bytes(generated, draw_width, bytes))
      # sampling_draws is a host Float32 array copied into the resident draw buffer.
      generation_buffers = checked_add(generation_buffers, checked_bytes(generated, draw_width, bytes))
      generation_buffers = checked_add(generation_buffers, checked_bytes(prompt, frame_width, bytes))
      generation_buffers = checked_add(generation_buffers, checked_bytes(Math.min(prompt, 16_i64), bytes))
      generation_buffers = checked_add(generation_buffers, checked_bytes(backbone.hidden.to_i64, bytes))
      generation_buffers = checked_add(generation_buffers, checked_bytes(backbone.key_dim.to_i64, bytes))

      # DAC scratch is three reusable decoder slabs. The waveform is a separate
      # output allocation and is accounted in codec temporaries below.
      dac_activation_elements = max_dac_activation_elements(codec, generated)
      decoder_scratch = checked_bytes(3_i64, dac_activation_elements, bytes)

      # Non-decoder DAC commands. Each command's allocations stay live until
      # completion; graph depth two therefore admits at most the two largest
      # command groups simultaneously. Decoder slabs are counted separately.
      codec_commands, codec_weightnorm_commands = codec_command_sizes(codec, generated)
      codec_temporaries = largest_n_sum(codec_commands, GRAPH_DEPTH)
      effective_weight_temporaries = largest_n_sum(codec_weightnorm_commands, GRAPH_DEPTH)
      waveform_elements = waveform_elements_per_frame(codec, generated)
      codec_temporaries = checked_add(codec_temporaries, checked_bytes(waveform_elements, bytes))

      # Resident generation keeps each submitted row's allocations alive until
      # the bounded queue retires it. MoE widths are explicit request inputs.
      generation_command = checked_add(
        backbone_step_bytes(backbone, positions, tower, moe_width, shared_width),
        tower_frame_bytes(tower, positions),
      )
      generation_temporaries = checked_bytes(generation_command, GRAPH_DEPTH.to_i64)

      reference_peak = request.reference_encoder_peak_bytes || 0_i64
      estimate_total = 0_i64
      {
        request.checkpoint_mapped_bytes,
        request.converted_static_bytes,
        backbone_cache,
        tower_cache,
        recurrent,
        generation_buffers,
        decoder_scratch,
        codec_temporaries,
        generation_temporaries,
        reference_peak,
      }.each { |part| estimate_total = checked_add(estimate_total, part) }

      largest_buffer = request.largest_checkpoint_mapped_buffer_bytes
      largest_buffer = Math.max(largest_buffer, request.largest_converted_buffer_bytes)
      largest_buffer = Math.max(largest_buffer, checked_bytes(
        positions, backbone.kv_heads.to_i64, backbone.head_dim.to_i64, bytes,
      )) if full_layers > 0
      largest_buffer = Math.max(largest_buffer, checked_bytes(
        2_i64, backbone.heads.to_i64, backbone.head_dim.to_i64, bytes,
      )) if full_layers > 0
      largest_buffer = Math.max(largest_buffer, checked_bytes(
        qkv_width, backbone.conv_kernel.to_i64, bytes,
      )) if linear_layers > 0
      largest_buffer = Math.max(largest_buffer, checked_bytes(positions, tower.n_local_heads.to_i64, tower.head_dim.to_i64, bytes))
      largest_buffer = Math.max(largest_buffer, full_score_buffer)
      largest_buffer = Math.max(largest_buffer, checked_bytes(generated, frame_width, bytes))
      largest_buffer = Math.max(largest_buffer, checked_bytes(generated, draw_width, bytes))
      largest_buffer = Math.max(largest_buffer, checked_bytes(dac_activation_elements, bytes))
      largest_buffer = Math.max(largest_buffer, checked_bytes(waveform_elements, bytes))
      largest_buffer = Math.max(largest_buffer, largest_codec_buffer(codec, generated))
      largest_buffer = Math.max(largest_buffer, largest_generation_buffer(backbone, positions, tower, moe_width, shared_width))

      Breakdown.new(
        request.checkpoint_mapped_bytes,
        request.converted_static_bytes,
        backbone_cache,
        tower_cache,
        recurrent,
        generation_buffers,
        decoder_scratch,
        effective_weight_temporaries,
        codec_temporaries,
        generation_temporaries,
        reference_peak,
        estimate_total,
        largest_buffer,
      )
    end

    private def backbone_step_bytes(config : BackboneConfig, positions : Int64, tower : TowerConfig,
                                    moe_width : Int64, shared_width : Int64) : Int64
      full_layers = (config.layers // config.full_interval).to_i64
      linear_layers = config.layers.to_i64 - full_layers
      full_projection = checked_add(
        checked_product(2_i64, config.heads.to_i64, config.head_dim.to_i64),
        checked_product(2_i64, config.kv_heads.to_i64, config.head_dim.to_i64),
      )
      full_per_layer = checked_add(
        checked_product(20_i64, config.hidden.to_i64),
        checked_add(full_projection, checked_product(config.heads.to_i64, positions)),
      )
      linear_width = checked_add(
        checked_product(2_i64, config.key_heads.to_i64, config.key_dim.to_i64),
        checked_product(config.value_heads.to_i64, config.value_dim.to_i64),
      )
      linear_per_layer = checked_add(
        checked_product(24_i64, config.hidden.to_i64),
        checked_product(8_i64, linear_width),
      )
      routed = checked_add(
        checked_product(3_i64, config.top_k.to_i64, moe_width),
        checked_product(config.top_k.to_i64, config.hidden.to_i64),
      )
      shared = checked_product(4_i64, shared_width)
      common_per_layer = checked_add(
        checked_product(24_i64, config.hidden.to_i64),
        checked_add(config.experts.to_i64, checked_add(routed, shared)),
      )
      elements = checked_add(
        checked_product(full_layers, full_per_layer),
        checked_add(checked_product(linear_layers, linear_per_layer),
          checked_product(config.layers.to_i64, common_per_layer)),
      )
      # Main-token logits and one top-p/top-k sampling workspace.
      elements = checked_add(elements, tower.vocab_size.to_i64)
      checked_bytes(elements, FLOAT_BYTES)
    end

    private def tower_frame_bytes(config : TowerConfig, positions : Int64) : Int64
      q_width = checked_product(config.n_head.to_i64, config.head_dim.to_i64)
      kv_width = checked_product(config.n_local_heads.to_i64, config.head_dim.to_i64)
      layer_elements = checked_add(
        checked_product(28_i64, config.dim.to_i64),
        checked_add(
          checked_product(4_i64, config.intermediate_size.to_i64),
          checked_add(
            checked_product(4_i64, q_width),
            checked_add(checked_product(2_i64, kv_width), checked_product(config.n_head.to_i64, positions)),
          ),
        ),
      )
      elements = checked_product(config.layer_count.to_i64, layer_elements)
      # Codebook embedding accumulation, logits, and sampler output buffers.
      elements = checked_add(elements, checked_product(config.num_codebooks.to_i64 + 2_i64, config.dim.to_i64))
      elements = checked_add(elements, checked_product(config.num_codebooks.to_i64 - 1_i64, config.vocab_size.to_i64))
      checked_bytes(elements, FLOAT_BYTES)
    end

    private def codec_command_sizes(config : CodecConfig, frames : Int64) : Tuple(Array(Int64), Array(Int64))
      commands = [] of Int64
      weightnorm_commands = [] of Int64
      bytes = FLOAT_BYTES
      books = config.residual_codebooks.to_i64 + 1_i64
      latent = checked_product(frames, config.latent_channels.to_i64)
      code_dim = checked_product(frames, config.codebook_dim.to_i64)
      codebook_elements = checked_add(latent, checked_product(books, checked_add(code_dim, checked_product(2_i64, latent))))
      codebook_weightnorm = checked_bytes(
        books,
        weightnorm_bytes(config.latent_channels.to_i64, config.codebook_dim.to_i64),
      )
      commands << checked_add(checked_bytes(codebook_elements, bytes), codebook_weightnorm)
      weightnorm_commands << codebook_weightnorm

      channels = config.latent_channels.to_i64
      ff = config.transformer_ff_dim.to_i64
      time = frames
      score_elements = checked_product(time, config.transformer_heads.to_i64, config.transformer_window.to_i64)
      # Transformer command allocates Q/K/V, scores/probabilities, attention
      # outputs, residuals, and gated feed-forward intermediates.
      transformer_elements = checked_add(
        checked_product(24_i64, time, channels),
        checked_add(checked_product(4_i64, time, ff), checked_product(2_i64, score_elements)),
      )
      config.post_layers.times { commands << checked_bytes(transformer_elements, bytes) }

      rows = time
      config.quantizer_upsample_rates.reverse_each do |rate32|
        previous_rows = rows
        rows = checked_product(rows, rate32.to_i64)
        # Deconvolution output/intermediates and the following ConvNeXt
        # command. They are separate submissions and may overlap at depth two.
        deconv = checked_bytes(checked_product(3_i64, rows, channels), bytes)
        commands << deconv
        convnext = checked_bytes(checked_product(13_i64, rows, channels), bytes)
        commands << convnext
        # Retain the prior activation explicitly as a live input to each stage.
        commands[-1] = checked_add(commands[-1], checked_bytes(previous_rows, channels, bytes))
        commands[-2] = checked_add(commands[-2], checked_bytes(previous_rows, channels, bytes))
      end

      # DAC's weight-normalized convolutions materialize per-call scale and
      # effective-weight buffers. They are separate allocations from the
      # converted static checkpoint tensors and stay bound through submission.
      initial_weightnorm = weightnorm_bytes(
        config.decoder_channels.to_i64,
        checked_product(config.latent_channels.to_i64, 7_i64),
      )
      commands << initial_weightnorm
      weightnorm_commands << initial_weightnorm

      channels = config.decoder_channels.to_i64
      config.decoder_rates.each do |rate32|
        rate = rate32.to_i64
        output_channels = channels // 2_i64
        transpose = weightnorm_bytes(
          channels,
          checked_product(output_channels, checked_product(rate, 2_i64)),
        )
        commands << transpose
        weightnorm_commands << transpose

        # Each residual unit submits two normalized convolutions together.
        residual_pair = checked_add(
          weightnorm_bytes(channels, checked_product(channels, 7_i64)),
          weightnorm_bytes(channels, channels),
        )
        3.times do
          commands << residual_pair
          weightnorm_commands << residual_pair
        end
        channels = output_channels
      end

      final_weightnorm = weightnorm_bytes(1_i64, checked_product(channels, 7_i64))
      commands << final_weightnorm
      weightnorm_commands << final_weightnorm
      {commands, weightnorm_commands}
    end

    private def largest_codec_buffer(config : CodecConfig, frames : Int64) : Int64
      bytes = FLOAT_BYTES
      largest = checked_bytes(frames, config.transformer_ff_dim.to_i64, bytes)
      largest = Math.max(largest, checked_bytes(frames, config.latent_channels.to_i64, 3_i64, bytes))
      largest = Math.max(largest, checked_bytes(
        frames, config.transformer_heads.to_i64, config.transformer_window.to_i64, bytes,
      ))
      rows = frames
      config.quantizer_upsample_rates.each do |rate|
        rows = checked_product(rows, rate.to_i64)
        largest = Math.max(largest, checked_bytes(rows, config.latent_channels.to_i64, 4_i64, bytes))
      end
      largest = Math.max(largest, largest_codec_weightnorm_buffer(config))
      largest
    end

    private def largest_codec_weightnorm_buffer(config : CodecConfig) : Int64
      channels = config.decoder_channels.to_i64
      largest_elements = checked_product(channels, config.latent_channels.to_i64, 7_i64)
      config.decoder_rates.each do |rate32|
        rate = rate32.to_i64
        output_channels = channels // 2_i64
        largest_elements = Math.max(largest_elements, checked_product(channels, output_channels, rate, 2_i64))
        largest_elements = Math.max(largest_elements, checked_product(channels, channels, 7_i64))
        largest_elements = Math.max(largest_elements, checked_product(channels, channels))
        channels = output_channels
      end
      largest_elements = Math.max(largest_elements, checked_product(channels, 7_i64))
      largest_elements = Math.max(largest_elements, checked_product(config.latent_channels.to_i64, config.codebook_dim.to_i64))
      checked_bytes(largest_elements, FLOAT_BYTES)
    end

    private def weightnorm_bytes(channels : Int64, elements_per_channel : Int64) : Int64
      checked_add(
        checked_bytes(channels, FLOAT_BYTES),
        checked_bytes(channels, elements_per_channel, FLOAT_BYTES),
      )
    end

    private def largest_generation_buffer(backbone : BackboneConfig, positions : Int64, tower : TowerConfig,
                                          moe_width : Int64, shared_width : Int64) : Int64
      largest = checked_bytes(checked_product(backbone.heads.to_i64, positions), FLOAT_BYTES)
      largest = Math.max(largest, checked_bytes(
        checked_product(2_i64, backbone.top_k.to_i64, moe_width), FLOAT_BYTES,
      ))
      largest = Math.max(largest, checked_bytes(checked_product(4_i64, shared_width), FLOAT_BYTES))
      largest = Math.max(largest, checked_bytes(
        checked_add(
          checked_product(2_i64, backbone.key_heads.to_i64, backbone.key_dim.to_i64),
          checked_product(backbone.value_heads.to_i64, backbone.value_dim.to_i64),
        ),
        FLOAT_BYTES,
      ))
      largest = Math.max(largest, checked_bytes(
        backbone.value_heads.to_i64, backbone.value_dim.to_i64,
        backbone.key_dim.to_i64, FLOAT_BYTES,
      ))
      largest = Math.max(largest, checked_bytes(
        checked_product(tower.n_head.to_i64, positions), FLOAT_BYTES,
      ))
      largest = Math.max(largest, checked_bytes(
        checked_product(4_i64, tower.intermediate_size.to_i64), FLOAT_BYTES,
      ))
      Math.max(largest, checked_bytes(tower.vocab_size.to_i64, FLOAT_BYTES))
    end

    private def max_dac_activation_elements(config : CodecConfig, frames : Int64) : Int64
      rows = frames
      config.quantizer_upsample_rates.each { |rate| rows = checked_product(rows, rate.to_i64) }
      channels = config.decoder_channels.to_i64
      maximum = checked_product(rows, channels)
      config.decoder_rates.each do |rate|
        rows = checked_product(rows, rate.to_i64)
        channels //= 2
        raise Rejected.new(Failure::InvalidRequest) unless channels > 0
        maximum = Math.max(maximum, checked_product(rows, channels))
      end
      maximum
    end

    private def waveform_elements_per_frame(config : CodecConfig, frames : Int64) : Int64
      elements = frames
      config.quantizer_upsample_rates.each { |rate| elements = checked_product(elements, rate.to_i64) }
      config.decoder_rates.each { |rate| elements = checked_product(elements, rate.to_i64) }
      elements
    end

    private def largest_n_sum(values : Array(Int64), count : Int32) : Int64
      sorted = values.sort.reverse
      total = 0_i64
      Math.min(count, sorted.size).times do |index|
        total = checked_add(total, sorted[index])
      end
      total
    end

    private def validate_request!(request : Request) : Nil
      raise Rejected.new(Failure::InvalidRequest) unless request.prompt_frames > 0 && request.min_frames > 0 && request.frame_cap > 0
      raise Rejected.new(Failure::PromptExceedsPositionCap) if request.prompt_frames > MAX_TOTAL_POSITIONS
      raise Rejected.new(Failure::InvalidRequest) unless request.checkpoint_mapped_bytes >= 0 && request.largest_checkpoint_mapped_buffer_bytes >= 0
      raise Rejected.new(Failure::InvalidRequest) unless request.converted_static_bytes >= 0 && request.largest_converted_buffer_bytes >= 0
      raise Rejected.new(Failure::InvalidRequest) unless request.largest_checkpoint_mapped_buffer_bytes <= request.checkpoint_mapped_bytes
      raise Rejected.new(Failure::InvalidRequest) unless request.largest_converted_buffer_bytes <= request.converted_static_bytes
      raise Rejected.new(Failure::InvalidRequest) unless request.additional_margin_bytes >= 0
      validate_model_configs!(request)
      raise Rejected.new(Failure::InvalidRequest) unless request.moe_intermediate_size && request.moe_intermediate_size.not_nil! > 0
      raise Rejected.new(Failure::InvalidRequest) unless request.shared_expert_intermediate_size && request.shared_expert_intermediate_size.not_nil! > 0
      if peak = request.reference_encoder_peak_bytes
        raise Rejected.new(Failure::InvalidRequest) if peak < 0
      end
    end

    private def validate_model_configs!(request : Request) : Nil
      backbone = request.backbone_config
      tower = request.tower_config
      codec = request.codec_config
      raise Rejected.new(Failure::InvalidRequest) unless backbone.layers > 0 && backbone.full_interval > 0
      raise Rejected.new(Failure::InvalidRequest) unless backbone.hidden > 0 && backbone.heads > 0 && backbone.kv_heads > 0 && backbone.head_dim > 0
      raise Rejected.new(Failure::InvalidRequest) unless backbone.experts > 0 && backbone.top_k > 0 && backbone.top_k <= backbone.experts
      raise Rejected.new(Failure::InvalidRequest) unless backbone.key_heads > 0 && backbone.value_heads > 0 && backbone.key_dim > 0 && backbone.value_dim > 0 && backbone.conv_kernel > 0
      raise Rejected.new(Failure::InvalidRequest) unless tower.vocab_size > 0 && tower.num_codebooks > 0 && tower.dim > 0
      raise Rejected.new(Failure::InvalidRequest) unless tower.n_head > 0 && tower.n_local_heads > 0 && tower.head_dim > 0 && tower.intermediate_size > 0
      raise Rejected.new(Failure::InvalidRequest) unless tower.layer_count > 0
      raise Rejected.new(Failure::InvalidRequest) unless codec.latent_channels > 0 && codec.codebook_dim > 0 && codec.residual_codebooks > 0
      raise Rejected.new(Failure::InvalidRequest) unless codec.semantic_codebook_size > 0 && codec.residual_codebook_size > 0
      raise Rejected.new(Failure::InvalidRequest) unless codec.transformer_heads > 0 && codec.transformer_head_dim > 0 && codec.transformer_ff_dim > 0 && codec.transformer_window > 0
      raise Rejected.new(Failure::InvalidRequest) unless codec.decoder_channels > 0 && !codec.quantizer_upsample_rates.empty? && !codec.decoder_rates.empty?
      raise Rejected.new(Failure::InvalidRequest) if codec.quantizer_upsample_rates.any? { |rate| rate <= 0 }
      raise Rejected.new(Failure::InvalidRequest) if codec.decoder_rates.any? { |rate| rate <= 0 }
    end

    private def read_telemetry!(telemetry : Telemetry) : Tuple(Int64, Int64, Int64, Int64, Int64)
      unless physical = telemetry.physical_bytes
        raise Rejected.new(Failure::UnknownTelemetry)
      end
      unless reclaimable = telemetry.free_reclaimable_bytes
        raise Rejected.new(Failure::UnknownTelemetry)
      end
      unless recommended = telemetry.metal_recommended_bytes
        raise Rejected.new(Failure::UnknownTelemetry)
      end
      unless allocated = telemetry.metal_current_allocated_bytes
        raise Rejected.new(Failure::UnknownTelemetry)
      end
      unless max_buffer = telemetry.max_buffer_length
        raise Rejected.new(Failure::UnknownTelemetry)
      end
      unless physical > 0 && reclaimable >= 0 && reclaimable <= physical && recommended > 0 && allocated >= 0 && max_buffer > 0
        raise Rejected.new(Failure::InvalidTelemetry)
      end
      raise Rejected.new(Failure::InvalidTelemetry) if allocated > recommended
      {physical, reclaimable, recommended, allocated, max_buffer}
    end

    private def subtract_budget(available : Int64, reserve : Int64, margin : Int64) : Int64
      protected_bytes = checked_add(reserve, margin)
      raise Rejected.new(Failure::InsufficientBudget) if available <= protected_bytes
      available - protected_bytes
    end

    private def ceil_percent(value : Int64, percent : Int64) : Int64
      quotient = value // 100_i64
      remainder = value % 100_i64
      checked_add(checked_product(quotient, percent), (remainder * percent + 99_i64) // 100_i64)
    end

    private def checked_bytes(*factors : Int64) : Int64
      checked_product(*factors)
    end

    private def checked_product(*factors : Int64) : Int64
      result = 1_i64
      factors.each do |factor|
        raise Rejected.new(Failure::InvalidRequest) if factor < 0
        if factor != 0 && result > Int64::MAX // factor
          raise Rejected.new(Failure::Overflow)
        end
        result *= factor
      end
      result
    end

    private def checked_add(left : Int64, right : Int64) : Int64
      raise Rejected.new(Failure::InvalidRequest) if left < 0 || right < 0
      raise Rejected.new(Failure::Overflow) if left > Int64::MAX - right
      left + right
    end
  end
end
