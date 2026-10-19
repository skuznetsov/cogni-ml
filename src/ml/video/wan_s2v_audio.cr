# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Apache-2.0. Native F32 MotionEncoder/CausalAudioEncoder path for Wan S2V.
require "./checkpoint"

module ML::Video
  # Fixed single-sample geometry for the upstream Wan S2V causal audio
  # MotionEncoder. The input is the resident layer-major XLSR bucket tensor
  # [layers, channels, bucket_frames]. `prefix_frames` repeats its first audio
  # frame; `drop_frames` removes the initial motion-latent audio frames.
  record WanS2VCausalAudioConfig,
    feature_layers : Int32 = 25,
    input_dim : Int32 = 1_024,
    hidden_dim : Int32 = 5_120,
    local_tokens : Int32 = 4,
    bucket_frames : Int32 = 84,
    prefix_frames : Int32 = 17,
    drop_frames : Int32 = 5 do
    def validate! : Nil
      unless feature_layers > 0 && input_dim > 0 && hidden_dim > 0 &&
             local_tokens > 0 && bucket_frames > 0 && prefix_frames >= 0 && drop_frames >= 0
        raise ArgumentError.new("Invalid Wan S2V causal audio dimensions")
      end
      unless bucket_frames.to_i64 + prefix_frames <= Int32::MAX
        raise ArgumentError.new("Wan S2V causal audio input frame count exceeds Int32")
      end
      unless hidden_dim % 4 == 0
        raise ArgumentError.new("Wan S2V causal audio hidden dimension must divide into four channels")
      end
      unless input_frames > 0 && video_frames > 0
        raise ArgumentError.new("Wan S2V causal audio geometry produces no frames")
      end
      validate_count!(feature_values.to_i64)
      validate_count!(input_frames.to_i64 * input_dim)
      validate_count!(local_conv1_values.to_i64)
      validate_count!(local_conv2_values.to_i64)
      validate_count!(local_conv3_values.to_i64)
      validate_count!(global_conv1_values.to_i64)
      validate_count!(global_conv2_values.to_i64)
      validate_count!(global_conv3_values.to_i64)
      validate_count!(local_output_values.to_i64)
      validate_count!(global_output_values.to_i64)
    end

    # A cheap setup fingerprint guards the dimensions used to bind checkpoint
    # weights and allocate reusable buffers. No mutable collections are held.
    def signature : UInt64
      hash = 14_695_981_039_346_656_037_u64
      hash = signature_word(hash, feature_layers)
      hash = signature_word(hash, input_dim)
      hash = signature_word(hash, hidden_dim)
      hash = signature_word(hash, local_tokens)
      hash = signature_word(hash, bucket_frames)
      hash = signature_word(hash, prefix_frames)
      signature_word(hash, drop_frames)
    end

    private def signature_word(hash : UInt64, value : Int32) : UInt64
      (hash ^ value.to_u32.to_u64) &* 1_099_511_628_211_u64
    end

    def input_frames : Int32
      (bucket_frames.to_i64 + prefix_frames).to_i32
    end

    # Causal pad adds K-1 samples on the left, then Conv1d uses no right pad.
    # For kernel 3 this is ceil(input / stride).
    def conv1_frames : Int32
      input_frames
    end

    def conv2_frames : Int32
      ceil_div(conv1_frames, 2)
    end

    def conv3_frames : Int32
      ceil_div(conv2_frames, 2)
    end

    def video_frames : Int32
      conv3_frames - drop_frames
    end

    def quarter_dim : Int32
      hidden_dim // 4
    end

    def half_dim : Int32
      hidden_dim // 2
    end

    def feature_values : Int32
      checked_count(feature_layers, input_dim, bucket_frames)
    end

    def local_conv1_values : Int32
      checked_count(local_tokens, conv1_frames, quarter_dim)
    end

    def local_conv2_values : Int32
      checked_count(local_tokens, conv2_frames, half_dim)
    end

    def local_conv3_values : Int32
      checked_count(local_tokens, conv3_frames, hidden_dim)
    end

    def global_conv1_values : Int32
      checked_count(conv1_frames, quarter_dim)
    end

    def global_conv2_values : Int32
      checked_count(conv2_frames, half_dim)
    end

    def global_conv3_values : Int32
      checked_count(conv3_frames, hidden_dim)
    end

    def local_output_values : Int32
      checked_count(video_frames, local_tokens + 1, hidden_dim)
    end

    def global_output_values : Int32
      checked_count(video_frames, hidden_dim)
    end

    def max_activation_values : Int32
      maximum = local_conv1_values
      maximum = local_conv2_values if local_conv2_values > maximum
      maximum = local_conv3_values if local_conv3_values > maximum
      maximum = global_conv1_values if global_conv1_values > maximum
      maximum = global_conv2_values if global_conv2_values > maximum
      maximum = global_conv3_values if global_conv3_values > maximum
      maximum
    end

    def scratch_bytes : Int64
      (input_frames.to_i64 * input_dim + 2_i64 * max_activation_values) * sizeof(Float32)
    end

    # These are caller-allocated optional trace buffers. Production leaves the
    # trace absent, so encode only emits the actual neural operators.
    def trace_plan : Array(Tuple(String, Int32))
      local1 = local_conv1_values
      local2 = local_conv2_values
      local3 = local_conv3_values
      global1 = global_conv1_values
      global2 = global_conv2_values
      global3 = global_conv3_values
      [
        {"weighted", input_frames * input_dim},
        {"local_conv1", local1}, {"local_norm1", local1}, {"local_act1", local1},
        {"local_conv2", local2}, {"local_norm2", local2}, {"local_act2", local2},
        {"local_conv3", local3}, {"local_norm3", local3}, {"local_act3", local3},
        {"local_output", local_output_values},
        {"global_conv1", global1}, {"global_norm1", global1}, {"global_act1", global1},
        {"global_conv2", global2}, {"global_norm2", global2}, {"global_act2", global2},
        {"global_conv3", global3}, {"global_norm3", global3}, {"global_act3", global3},
        {"global_linear", global3}, {"global_output", global_output_values},
      ] of Tuple(String, Int32)
    end

    private def ceil_div(value : Int32, divisor : Int32) : Int32
      ((value.to_i64 + divisor - 1) // divisor).to_i32
    end

    private def checked_count(*factors : Int32) : Int32
      result = 1_i64
      factors.each do |factor|
        unless factor > 0 && result <= Int32::MAX.to_i64 // factor
          raise ArgumentError.new("Wan S2V causal audio tensor exceeds native addressable storage")
        end
        result *= factor
      end
      result.to_i32
    end

    private def validate_count!(count : Int64) : Nil
      unless count > 0 && count <= Int32::MAX
        raise ArgumentError.new("Wan S2V causal audio tensor exceeds native addressable storage")
      end
    end
  end

  record WanS2VCausalAudioConvWeights,
    weight : ML::MetalBuffer, bias : ML::MetalBuffer

  record WanS2VCausalAudioLinearWeights,
    weight : ML::MetalBuffer, bias : ML::MetalBuffer

  record WanS2VCausalAudioWeights,
    layer_weights : ML::MetalBuffer,
    conv1_local : WanS2VCausalAudioConvWeights,
    conv1_global : WanS2VCausalAudioConvWeights,
    conv2 : WanS2VCausalAudioConvWeights,
    conv3 : WanS2VCausalAudioConvWeights,
    final_linear : WanS2VCausalAudioLinearWeights,
    padding_tokens : ML::MetalBuffer,
    config_signature : UInt64

  record WanS2VCausalAudioScratch,
    weighted_features : ML::MetalBuffer,
    activation_a : ML::MetalBuffer,
    activation_b : ML::MetalBuffer,
    config_signature : UInt64

  module WanS2VCausalAudioEncoder
    SOURCE                         = {{ read_file("#{__DIR__}/kernels/wan_s2v_audio.metal") }}
    WAN_SOURCE_COMMIT              = "1ea34ff48f87168174e12956e200b1d908b1c5ff"
    WAN_AUDIO_UTILS_SHA256         = "76931d7291f69a6b6ed3a7a1a9a106ad8b5f5d494a451db67718b4fbcc4dbcfd"
    WAN_AUXI_BLOCKS_SHA256         = "6ff7b5560036e5f0eaaa34460a328afcdbd7bd62baf93b699d5f6d2d556f95d6"
    MPS_LAYER_NORM_SHA256          = "1a289179208970a07e5da56d72c8533adb2549cea49427541c0178ded525d597"
    MPS_LAYER_NORM_DISPATCH_SHA256 = "4acdf9b945e74359254383077c10a0e0142cd824a6410d6ad0f4c8c9f32dffce"
    NORM_EPS                       = 1e-6_f32
    SOURCE_PREFIX                  = "s2v_audio."

    def self.pipeline(name : String) : ML::Metal::ComputePipeline
      ML::Metal::PipelineCache.get(name) { ML::Metal::ComputePipeline.new(name, SOURCE) }
    end

    def self.prepare! : Nil
      MetalOps.prepare!
      {"wan_s2v_audio_weighted", "wan_s2v_audio_causal_conv1d", "wan_s2v_audio_silu",
       "wan_s2v_audio_pack_local", "wan_s2v_audio_crop_global",
       "wan_s2v_audio_layer_norm_mps"}.each { |name| pipeline(name) }
    end

    def self.checkpoint_bindings(cfg : WanS2VCausalAudioConfig,
                                 dtype : ML::GGUF::SafeTensorDType? = nil,
                                 prefix : String = SOURCE_PREFIX) : Array(WeightBinding)
      cfg.validate!
      result = [] of WeightBinding
      add_binding(result, prefix + "layer_weights", "casual_audio_encoder.weights",
        [1_i64, cfg.feature_layers.to_i64, 1_i64, 1_i64], dtype)
      add_conv_bindings(result, prefix + "conv1_local", "casual_audio_encoder.encoder.conv1_local.conv",
        cfg.local_tokens * cfg.quarter_dim, cfg.input_dim, dtype)
      add_conv_bindings(result, prefix + "conv1_global", "casual_audio_encoder.encoder.conv1_global.conv",
        cfg.quarter_dim, cfg.input_dim, dtype)
      add_conv_bindings(result, prefix + "conv2", "casual_audio_encoder.encoder.conv2.conv",
        cfg.half_dim, cfg.quarter_dim, dtype)
      add_conv_bindings(result, prefix + "conv3", "casual_audio_encoder.encoder.conv3.conv",
        cfg.hidden_dim, cfg.half_dim, dtype)
      add_binding(result, prefix + "final_linear.weight", "casual_audio_encoder.encoder.final_linear.weight",
        [cfg.hidden_dim.to_i64, cfg.hidden_dim.to_i64], dtype)
      add_binding(result, prefix + "final_linear.bias", "casual_audio_encoder.encoder.final_linear.bias",
        [cfg.hidden_dim.to_i64], dtype)
      add_binding(result, prefix + "padding_tokens", "casual_audio_encoder.encoder.padding_tokens",
        [1_i64, 1_i64, 1_i64, cfg.hidden_dim.to_i64], dtype)
      result
    end

    def self.bind_weights(arena : Arena, cfg : WanS2VCausalAudioConfig,
                          prefix : String = SOURCE_PREFIX) : WanS2VCausalAudioWeights
      cfg.validate!
      conv1_local = conv_weights(arena, prefix + "conv1_local", cfg.local_tokens.to_i64 * cfg.quarter_dim * cfg.input_dim * 3,
        cfg.local_tokens.to_i64 * cfg.quarter_dim)
      conv1_global = conv_weights(arena, prefix + "conv1_global", cfg.quarter_dim.to_i64 * cfg.input_dim * 3,
        cfg.quarter_dim.to_i64)
      conv2 = conv_weights(arena, prefix + "conv2", cfg.half_dim.to_i64 * cfg.quarter_dim * 3,
        cfg.half_dim.to_i64)
      conv3 = conv_weights(arena, prefix + "conv3", cfg.hidden_dim.to_i64 * cfg.half_dim * 3,
        cfg.hidden_dim.to_i64)
      WanS2VCausalAudioWeights.new(
        resident_f32(arena, prefix + "layer_weights", cfg.feature_layers.to_i64),
        conv1_local, conv1_global, conv2, conv3,
        WanS2VCausalAudioLinearWeights.new(
          resident_f32(arena, prefix + "final_linear.weight", cfg.hidden_dim.to_i64 * cfg.hidden_dim),
          resident_f32(arena, prefix + "final_linear.bias", cfg.hidden_dim.to_i64)),
        resident_f32(arena, prefix + "padding_tokens", cfg.hidden_dim.to_i64),
        cfg.signature)
    end

    def self.prepare_scratch!(arena : Arena, cfg : WanS2VCausalAudioConfig,
                              prefix : String = "s2v_audio.s.") : WanS2VCausalAudioScratch
      cfg.validate!
      prepare!
      weighted = arena.alloc(prefix + "weighted", cfg.input_frames * cfg.input_dim)
      activation_a = arena.alloc(prefix + "activation_a", cfg.max_activation_values)
      activation_b = arena.alloc(prefix + "activation_b", cfg.max_activation_values)
      WanS2VCausalAudioScratch.new(weighted, activation_a, activation_b, cfg.signature)
    end

    # Consume the resident XLSR bucket states and emit Wan's local five-token
    # and global AdaIN features into a caller-owned serial command encoder.
    # No GPU allocations, command submission, CPU readback, or waits occur.
    # The trace map, when supplied, must contain only setup-allocated buffers
    # named by `cfg.trace_plan`.
    def self.encode(e : ML::Metal::ComputeEncoder, cfg : WanS2VCausalAudioConfig,
                    features : ML::MetalBuffer, local_output : ML::MetalBuffer,
                    global_output : ML::MetalBuffer, weights : WanS2VCausalAudioWeights,
                    scratch : WanS2VCausalAudioScratch,
                    trace : Hash(String, ML::MetalBuffer)? = nil) : Nil
      signature = cfg.signature
      unless signature == weights.config_signature && signature == scratch.config_signature
        raise ArgumentError.new("Wan S2V causal audio config changed after weights or scratch setup")
      end
      cfg.validate!
      MetalOps.check(features, cfg.feature_values.to_i64)
      MetalOps.check(local_output, cfg.local_output_values.to_i64)
      MetalOps.check(global_output, cfg.global_output_values.to_i64)
      MetalOps.check(scratch.weighted_features, (cfg.input_frames.to_i64 * cfg.input_dim))
      MetalOps.check(scratch.activation_a, cfg.max_activation_values.to_i64)
      MetalOps.check(scratch.activation_b, cfg.max_activation_values.to_i64)
      MetalOps.distinct(local_output, features, global_output, scratch.weighted_features,
        scratch.activation_a, scratch.activation_b, weights.layer_weights, weights.padding_tokens,
        weights.conv1_local.weight, weights.conv1_local.bias, weights.conv1_global.weight,
        weights.conv1_global.bias, weights.conv2.weight, weights.conv2.bias,
        weights.conv3.weight, weights.conv3.bias, weights.final_linear.weight,
        weights.final_linear.bias)
      MetalOps.distinct(global_output, features, scratch.weighted_features,
        local_output, scratch.activation_a, scratch.activation_b, weights.layer_weights,
        weights.padding_tokens, weights.conv1_local.weight, weights.conv1_local.bias,
        weights.conv1_global.weight, weights.conv1_global.bias, weights.conv2.weight,
        weights.conv2.bias, weights.conv3.weight, weights.conv3.bias,
        weights.final_linear.weight, weights.final_linear.bias)
      MetalOps.distinct(scratch.weighted_features, features, local_output, global_output,
        scratch.activation_a, scratch.activation_b, weights.layer_weights,
        weights.conv1_local.weight, weights.conv1_local.bias, weights.conv1_global.weight,
        weights.conv1_global.bias, weights.conv2.weight, weights.conv2.bias,
        weights.conv3.weight, weights.conv3.bias, weights.final_linear.weight,
        weights.final_linear.bias, weights.padding_tokens)
      MetalOps.distinct(scratch.activation_a, features, local_output, global_output,
        scratch.weighted_features, scratch.activation_b, weights.layer_weights,
        weights.conv1_local.weight, weights.conv1_local.bias, weights.conv1_global.weight,
        weights.conv1_global.bias, weights.conv2.weight, weights.conv2.bias,
        weights.conv3.weight, weights.conv3.bias, weights.final_linear.weight,
        weights.final_linear.bias, weights.padding_tokens)
      MetalOps.distinct(scratch.activation_b, features, local_output, global_output,
        scratch.weighted_features, weights.layer_weights, weights.conv1_local.weight,
        weights.conv1_local.bias, weights.conv1_global.weight, weights.conv1_global.bias,
        weights.conv2.weight, weights.conv2.bias, weights.conv3.weight, weights.conv3.bias,
        weights.final_linear.weight, weights.final_linear.bias, weights.padding_tokens)
      validate_trace!(trace, cfg, features, local_output, global_output, weights, scratch)

      weighted(e, cfg, features, weights.layer_weights, scratch.weighted_features)
      capture(e, trace, "weighted", scratch.weighted_features, cfg.input_frames * cfg.input_dim)

      causal_conv(e, scratch.weighted_features, weights.conv1_local, scratch.activation_a,
        input_frames: cfg.conv1_frames, output_frames: cfg.conv1_frames,
        input_channels: cfg.input_dim, output_channels: cfg.quarter_dim,
        sequences: cfg.local_tokens, weight_groups: cfg.local_tokens, stride: 1)
      capture(e, trace, "local_conv1", scratch.activation_a, cfg.local_conv1_values)
      non_affine_norm(e, scratch.activation_a, cfg.local_tokens * cfg.conv1_frames,
        cfg.quarter_dim, cfg.local_conv1_values)
      capture(e, trace, "local_norm1", scratch.activation_a, cfg.local_conv1_values)
      silu(e, scratch.activation_a, cfg.local_conv1_values)
      capture(e, trace, "local_act1", scratch.activation_a, cfg.local_conv1_values)

      causal_conv(e, scratch.activation_a, weights.conv2, scratch.activation_b,
        input_frames: cfg.conv1_frames, output_frames: cfg.conv2_frames,
        input_channels: cfg.quarter_dim, output_channels: cfg.half_dim,
        sequences: cfg.local_tokens, weight_groups: 1, stride: 2)
      capture(e, trace, "local_conv2", scratch.activation_b, cfg.local_conv2_values)
      non_affine_norm(e, scratch.activation_b, cfg.local_tokens * cfg.conv2_frames,
        cfg.half_dim, cfg.local_conv2_values)
      capture(e, trace, "local_norm2", scratch.activation_b, cfg.local_conv2_values)
      silu(e, scratch.activation_b, cfg.local_conv2_values)
      capture(e, trace, "local_act2", scratch.activation_b, cfg.local_conv2_values)

      causal_conv(e, scratch.activation_b, weights.conv3, scratch.activation_a,
        input_frames: cfg.conv2_frames, output_frames: cfg.conv3_frames,
        input_channels: cfg.half_dim, output_channels: cfg.hidden_dim,
        sequences: cfg.local_tokens, weight_groups: 1, stride: 2)
      capture(e, trace, "local_conv3", scratch.activation_a, cfg.local_conv3_values)
      non_affine_norm(e, scratch.activation_a, cfg.local_tokens * cfg.conv3_frames,
        cfg.hidden_dim, cfg.local_conv3_values)
      capture(e, trace, "local_norm3", scratch.activation_a, cfg.local_conv3_values)
      silu(e, scratch.activation_a, cfg.local_conv3_values)
      capture(e, trace, "local_act3", scratch.activation_a, cfg.local_conv3_values)
      pack_local(e, scratch.activation_a, weights.padding_tokens, local_output, cfg)
      capture(e, trace, "local_output", local_output, cfg.local_output_values)

      causal_conv(e, scratch.weighted_features, weights.conv1_global, scratch.activation_b,
        input_frames: cfg.conv1_frames, output_frames: cfg.conv1_frames,
        input_channels: cfg.input_dim, output_channels: cfg.quarter_dim,
        sequences: 1, weight_groups: 1, stride: 1)
      capture(e, trace, "global_conv1", scratch.activation_b, cfg.global_conv1_values)
      non_affine_norm(e, scratch.activation_b, cfg.conv1_frames,
        cfg.quarter_dim, cfg.global_conv1_values)
      capture(e, trace, "global_norm1", scratch.activation_b, cfg.global_conv1_values)
      silu(e, scratch.activation_b, cfg.global_conv1_values)
      capture(e, trace, "global_act1", scratch.activation_b, cfg.global_conv1_values)

      causal_conv(e, scratch.activation_b, weights.conv2, scratch.activation_a,
        input_frames: cfg.conv1_frames, output_frames: cfg.conv2_frames,
        input_channels: cfg.quarter_dim, output_channels: cfg.half_dim,
        sequences: 1, weight_groups: 1, stride: 2)
      capture(e, trace, "global_conv2", scratch.activation_a, cfg.global_conv2_values)
      non_affine_norm(e, scratch.activation_a, cfg.conv2_frames,
        cfg.half_dim, cfg.global_conv2_values)
      capture(e, trace, "global_norm2", scratch.activation_a, cfg.global_conv2_values)
      silu(e, scratch.activation_a, cfg.global_conv2_values)
      capture(e, trace, "global_act2", scratch.activation_a, cfg.global_conv2_values)

      causal_conv(e, scratch.activation_a, weights.conv3, scratch.activation_b,
        input_frames: cfg.conv2_frames, output_frames: cfg.conv3_frames,
        input_channels: cfg.half_dim, output_channels: cfg.hidden_dim,
        sequences: 1, weight_groups: 1, stride: 2)
      capture(e, trace, "global_conv3", scratch.activation_b, cfg.global_conv3_values)
      non_affine_norm(e, scratch.activation_b, cfg.conv3_frames,
        cfg.hidden_dim, cfg.global_conv3_values)
      capture(e, trace, "global_norm3", scratch.activation_b, cfg.global_conv3_values)
      silu(e, scratch.activation_b, cfg.global_conv3_values)
      capture(e, trace, "global_act3", scratch.activation_b, cfg.global_conv3_values)
      MetalOps.linear(e, scratch.activation_b, weights.final_linear.weight,
        weights.final_linear.bias, scratch.activation_a, cfg.conv3_frames,
        cfg.hidden_dim, cfg.hidden_dim)
      capture(e, trace, "global_linear", scratch.activation_a, cfg.global_conv3_values)
      crop_global(e, scratch.activation_a, global_output, cfg)
      capture(e, trace, "global_output", global_output, cfg.global_output_values)
    end

    private def self.weighted(e : ML::Metal::ComputeEncoder, cfg : WanS2VCausalAudioConfig,
                              features : ML::MetalBuffer, layer_weights : ML::MetalBuffer,
                              output : ML::MetalBuffer) : Nil
      e.set_pipeline(pipeline("wan_s2v_audio_weighted"))
      e.set_buffer(features, 0).set_buffer(layer_weights, 1)
      e.set_buffer(output, 2, ML::Metal::BufferAccess::Write)
      e.set_value(cfg.feature_layers.to_u32, 3).set_value(cfg.input_dim.to_u32, 4)
      e.set_value(cfg.bucket_frames.to_u32, 5).set_value(cfg.prefix_frames.to_u32, 6)
      e.set_value(cfg.input_frames.to_u32, 7)
      e.dispatch_1d(cfg.input_frames * cfg.input_dim)
    end

    private def self.causal_conv(e : ML::Metal::ComputeEncoder, input : ML::MetalBuffer,
                                 weights : WanS2VCausalAudioConvWeights, output : ML::MetalBuffer,
                                 input_frames : Int32, output_frames : Int32,
                                 input_channels : Int32, output_channels : Int32,
                                 sequences : Int32, weight_groups : Int32, stride : Int32) : Nil
      count = sequences.to_i64 * output_frames * output_channels
      # The first local convolution has one shared input sequence and distinct
      # output-head weight groups. Later shared convolutions consume all heads.
      input_sequences = weight_groups == sequences && sequences > 1 ? 1 : sequences
      MetalOps.check(input, input_sequences.to_i64 * input_frames * input_channels)
      MetalOps.check(weights.weight, weight_groups.to_i64 * output_channels * input_channels * 3)
      MetalOps.check(weights.bias, weight_groups.to_i64 * output_channels)
      MetalOps.check(output, count)
      MetalOps.distinct(output, input, weights.weight, weights.bias)
      e.set_pipeline(pipeline("wan_s2v_audio_causal_conv1d"))
      e.set_buffer(input, 0).set_buffer(weights.weight, 1).set_buffer(weights.bias, 2)
      e.set_buffer(output, 3, ML::Metal::BufferAccess::Write)
      e.set_value(input_frames.to_u32, 4).set_value(output_frames.to_u32, 5)
      e.set_value(input_channels.to_u32, 6).set_value(output_channels.to_u32, 7)
      e.set_value(sequences.to_u32, 8).set_value(weight_groups.to_u32, 9)
      e.set_value(stride.to_u32, 10).dispatch_1d(count.to_i32)
    end

    private def self.non_affine_norm(e : ML::Metal::ComputeEncoder, values : ML::MetalBuffer,
                                     rows : Int32, dim : Int32, count : Int32) : Nil
      unless rows > 0 && dim > 0 && rows.to_i64 * dim == count
        raise ArgumentError.new("Invalid Wan S2V causal audio normalization dimensions")
      end
      MetalOps.check(values, count.to_i64)
      # Match PyTorch v2.8.0 MPS LayerNorm's four-values-per-thread reduction,
      # raw-moment variance clamp, and precise rsqrt. Keep this scoped here:
      # generic Video norm intentionally retains its centered-variance math.
      e.set_pipeline(pipeline("wan_s2v_audio_layer_norm_mps"))
      e.set_buffer(values, 0).set_buffer(values, 1, ML::Metal::BufferAccess::Write)
      e.set_value(dim.to_u32, 2).set_value(NORM_EPS, 3)
      threads = Math.min((dim.to_i64 + 3) // 4, 1_024_i64).to_i32
      e.dispatch_threadgroups({rows, 1, 1}, {threads, 1, 1})
    end

    private def self.silu(e : ML::Metal::ComputeEncoder, values : ML::MetalBuffer,
                          count : Int32) : Nil
      e.set_pipeline(pipeline("wan_s2v_audio_silu"))
      e.set_buffer(values, 0).set_buffer(values, 1, ML::Metal::BufferAccess::Write)
      e.set_value(count.to_u32, 2).dispatch_1d(count)
    end

    private def self.pack_local(e : ML::Metal::ComputeEncoder, motion : ML::MetalBuffer,
                                pad : ML::MetalBuffer, output : ML::MetalBuffer,
                                cfg : WanS2VCausalAudioConfig) : Nil
      MetalOps.check(motion, cfg.local_conv3_values.to_i64)
      MetalOps.check(pad, cfg.hidden_dim.to_i64)
      MetalOps.check(output, cfg.local_output_values.to_i64)
      MetalOps.distinct(output, motion, pad)
      e.set_pipeline(pipeline("wan_s2v_audio_pack_local"))
      e.set_buffer(motion, 0).set_buffer(pad, 1)
      e.set_buffer(output, 2, ML::Metal::BufferAccess::Write)
      e.set_value(cfg.conv3_frames.to_u32, 3).set_value(cfg.video_frames.to_u32, 4)
      e.set_value(cfg.drop_frames.to_u32, 5).set_value(cfg.local_tokens.to_u32, 6)
      e.set_value(cfg.hidden_dim.to_u32, 7)
      e.dispatch_1d(cfg.local_output_values)
    end

    private def self.crop_global(e : ML::Metal::ComputeEncoder, motion : ML::MetalBuffer,
                                 output : ML::MetalBuffer, cfg : WanS2VCausalAudioConfig) : Nil
      MetalOps.check(motion, cfg.global_conv3_values.to_i64)
      MetalOps.check(output, cfg.global_output_values.to_i64)
      MetalOps.distinct(output, motion)
      e.set_pipeline(pipeline("wan_s2v_audio_crop_global"))
      e.set_buffer(motion, 0).set_buffer(output, 1, ML::Metal::BufferAccess::Write)
      e.set_value(cfg.conv3_frames.to_u32, 2).set_value(cfg.video_frames.to_u32, 3)
      e.set_value(cfg.drop_frames.to_u32, 4).set_value(cfg.hidden_dim.to_u32, 5)
      e.dispatch_1d(cfg.global_output_values)
    end

    private def self.capture(e : ML::Metal::ComputeEncoder,
                             trace : Hash(String, ML::MetalBuffer)?, name : String,
                             source : ML::MetalBuffer, count : Int32) : Nil
      return unless trace
      if destination = trace[name]?
        MetalOps.copy(e, source, destination, count)
      end
    end

    private def self.validate_trace!(trace : Hash(String, ML::MetalBuffer)?,
                                     cfg : WanS2VCausalAudioConfig,
                                     features : ML::MetalBuffer,
                                     local_output : ML::MetalBuffer,
                                     global_output : ML::MetalBuffer,
                                     weights : WanS2VCausalAudioWeights,
                                     scratch : WanS2VCausalAudioScratch) : Nil
      return unless trace
      trace.each do |name, buffer|
        count = trace_count(name, cfg)
        raise ArgumentError.new("Unknown Wan S2V causal audio trace #{name}") unless count > 0
        MetalOps.check(buffer, count.to_i64)
        MetalOps.distinct(buffer, features, local_output, global_output,
          scratch.weighted_features, scratch.activation_a, scratch.activation_b,
          weights.layer_weights, weights.padding_tokens,
          weights.conv1_local.weight, weights.conv1_local.bias,
          weights.conv1_global.weight, weights.conv1_global.bias,
          weights.conv2.weight, weights.conv2.bias, weights.conv3.weight,
          weights.conv3.bias, weights.final_linear.weight, weights.final_linear.bias)
        trace.each do |other_name, other|
          next if name == other_name
          MetalOps.distinct(buffer, other)
        end
      end
    end

    private def self.trace_count(name : String, cfg : WanS2VCausalAudioConfig) : Int32
      case name
      when "weighted"                                                     then cfg.input_frames * cfg.input_dim
      when "local_conv1", "local_norm1", "local_act1"                     then cfg.local_conv1_values
      when "local_conv2", "local_norm2", "local_act2"                     then cfg.local_conv2_values
      when "local_conv3", "local_norm3", "local_act3"                     then cfg.local_conv3_values
      when "local_output"                                                 then cfg.local_output_values
      when "global_conv1", "global_norm1", "global_act1"                  then cfg.global_conv1_values
      when "global_conv2", "global_norm2", "global_act2"                  then cfg.global_conv2_values
      when "global_conv3", "global_norm3", "global_act3", "global_linear" then cfg.global_conv3_values
      when "global_output"                                                then cfg.global_output_values
      else                                                                     0
      end
    end

    private def self.add_conv_bindings(result : Array(WeightBinding), target : String,
                                       source : String, out_dim : Int32, in_dim : Int32,
                                       dtype : ML::GGUF::SafeTensorDType?) : Nil
      add_binding(result, target + ".weight", source + ".weight",
        [out_dim.to_i64, in_dim.to_i64, 3_i64], dtype)
      add_binding(result, target + ".bias", source + ".bias", [out_dim.to_i64], dtype)
    end

    private def self.add_binding(result : Array(WeightBinding), target : String,
                                 source : String, shape : Array(Int64),
                                 dtype : ML::GGUF::SafeTensorDType?) : Nil
      result << WeightBinding.new(target, source, shape, dtype)
    end

    private def self.conv_weights(arena : Arena, prefix : String,
                                  weight_count : Int64, bias_count : Int64) : WanS2VCausalAudioConvWeights
      WanS2VCausalAudioConvWeights.new(
        resident_f32(arena, prefix + ".weight", weight_count),
        resident_f32(arena, prefix + ".bias", bias_count))
    end

    private def self.resident_f32(arena : Arena, name : String, count : Int64) : ML::MetalBuffer
      buffer = arena[name]
      raise ArgumentError.new("Wan S2V causal audio requires F32 resident weights") unless arena.dtype(name).f32?
      MetalOps.check(buffer, count)
      buffer
    end
  end
end
