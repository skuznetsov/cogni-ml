# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Apache-2.0. Native F32 Wav2Vec2 feature/encoder path for Wan S2V.
require "json"
require "./checkpoint"

{% if flag?(:darwin) && !flag?(:cpu_only) %}
  @[Link(ldflags: "-framework Metal -framework Foundation -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph")]
  lib WanAudioMPSFFI
    fun plan_create = gs_wan_audio_mps_conv1d_plan_create(device : Pointer(Void),
                                                          input : Pointer(Void), weight : Pointer(Void), bias : Pointer(Void),
                                                          output : Pointer(Void), input_frames : Int32, input_channels : Int32,
                                                          output_channels : Int32, kernel_width : Int32, stride : Int32) : Pointer(Void)
    fun encode = gs_wan_audio_mps_conv1d_encode(plan : Pointer(Void), command : Pointer(Void)) : Int32
    fun plan_release = gs_wan_audio_mps_conv1d_plan_release(plan : Pointer(Void)) : Void
    fun normalize_plan_create = gs_wan_audio_mps_normalize_plan_create(device : Pointer(Void),
                                                                       input : Pointer(Void), output : Pointer(Void),
                                                                       input_samples : Int32) : Pointer(Void)
    fun normalize_encode = gs_wan_audio_mps_normalize_encode(plan : Pointer(Void), command : Pointer(Void)) : Int32
    fun normalize_plan_release = gs_wan_audio_mps_normalize_plan_release(plan : Pointer(Void)) : Void
    fun normalize_boundary_plan_create = gs_wan_audio_mps_source_boundary_normalize_plan_create(device : Pointer(Void),
                                                                                                input : Pointer(Void), output : Pointer(Void),
                                                                                                scalar_a : Pointer(Void), scalar_b : Pointer(Void),
                                                                                                centered : Pointer(Void), input_samples : Int32) : Pointer(Void)
    fun normalize_boundary_encode = gs_wan_audio_mps_source_boundary_normalize_encode(plan : Pointer(Void), command : Pointer(Void)) : Int32
    fun normalize_boundary_plan_release = gs_wan_audio_mps_source_boundary_normalize_plan_release(plan : Pointer(Void)) : Void
    fun gelu_plan_create = gs_wan_audio_mps_gelu_plan_create(device : Pointer(Void),
                                                             input : Pointer(Void), output : Pointer(Void),
                                                             elements : Int32) : Pointer(Void)
    fun gelu_encode = gs_wan_audio_mps_gelu_encode(plan : Pointer(Void), command : Pointer(Void)) : Int32
    fun gelu_plan_release = gs_wan_audio_mps_gelu_plan_release(plan : Pointer(Void)) : Void
    fun position_plan_create = gs_wan_audio_mps_position_plan_create(device : Pointer(Void),
                                                                     input : Pointer(Void), weight_v : Pointer(Void),
                                                                     scale : Pointer(Void), bias : Pointer(Void),
                                                                     raw_output : Pointer(Void), output : Pointer(Void),
                                                                     frames : Int32, hidden : Int32, groups : Int32,
                                                                     kernel_width : Int32) : Pointer(Void)
    fun position_encode = gs_wan_audio_mps_position_encode(plan : Pointer(Void), command : Pointer(Void)) : Int32
    fun position_plan_release = gs_wan_audio_mps_position_plan_release(plan : Pointer(Void)) : Void
  end
{% end %}

module ML::Video
  # Inference-only subset of Transformers Wav2Vec2Config used by Wan S2V.
  # `target_frames` is Wan's bucket batch size (80 for ordinary inference,
  # 84 when three source motion frames are later dropped and four padded audio
  # buckets must remain available to the native pipeline).
  record WanAudioEncoderConfig,
    sample_rate : Int32 = 16_000,
    input_samples : Int32 = 80_000,
    conv_dims : Array(Int32) = [512, 512, 512, 512, 512, 512, 512],
    conv_kernels : Array(Int32) = [10, 3, 3, 3, 3, 2, 2],
    conv_strides : Array(Int32) = [5, 2, 2, 2, 2, 2, 2],
    hidden_size : Int32 = 1_024,
    intermediate_size : Int32 = 4_096,
    heads : Int32 = 16,
    layers : Int32 = 24,
    position_kernel : Int32 = 128,
    position_groups : Int32 = 16,
    feature_norm : String = "layer",
    stable_layer_norm : Bool = true,
    conv_bias : Bool = true,
    feature_activation : String = "gelu",
    hidden_activation : String = "gelu",
    layer_norm_eps : Float32 = 1e-5_f32,
    video_fps : Int32 = 30,
    target_fps : Int32 = 16,
    target_frames : Int32 = 84 do
    def validate! : Nil
      unless sample_rate == 16_000 && input_samples > 0 && target_frames > 0 &&
             video_fps > 0 && target_fps > 0 && conv_dims.size > 0 &&
             conv_dims.size == conv_kernels.size && conv_dims.size == conv_strides.size
        raise ArgumentError.new("Invalid Wan Wav2Vec2 geometry")
      end
      unless feature_norm == "layer" && stable_layer_norm && conv_bias &&
             feature_activation == "gelu" && hidden_activation == "gelu"
        raise ArgumentError.new("Unsupported Wan Wav2Vec2 architecture")
      end
      unless hidden_size > 0 && intermediate_size > 0 && layers > 0 && heads > 0 &&
             hidden_size % heads == 0 && hidden_size // heads <= 256 &&
             position_kernel > 0 && position_groups > 0 &&
             hidden_size % position_groups == 0 &&
             layer_norm_eps.finite? && layer_norm_eps > 0
        raise ArgumentError.new("Unsupported Wan Wav2Vec2 dimensions")
      end
      conv_dims.each_with_index do |dim, i|
        unless dim > 0 && conv_kernels[i] > 0 && conv_strides[i] > 0
          raise ArgumentError.new("Invalid Wav2Vec2 feature convolution")
        end
      end
      lengths = convolution_output_lengths
      feature_frames = lengths.last
      raise ArgumentError.new("Wav2Vec2 audio is too short") unless feature_frames > 1
      unless sample_rate % total_conv_stride == 0
        raise ArgumentError.new("Wav2Vec2 feature rate must be integral")
      end
      raise ArgumentError.new("Wan interpolation requires at least two frames") unless interpolation_frames > 1
      # All one-dimensional dispatch ranges and output tensors are Int32
      # addressable in the current native Metal wrapper.
      sequence_count = feature_frames.to_i64 * hidden_size
      ffn_count = feature_frames.to_i64 * intermediate_size
      output_count = hidden_state_count.to_i64 * hidden_size * bucket_frames
      unless conv_dims[0].to_i64 * lengths[0] <= Int32::MAX &&
             sequence_count <= Int32::MAX && ffn_count <= Int32::MAX && output_count <= Int32::MAX
        raise ArgumentError.new("Wav2Vec2 tensor exceeds native addressable storage")
      end
    end

    # Stable, allocation-free fingerprint used to ensure resident weights and
    # scratch were prepared for this exact scalar and mutable-array geometry.
    # This protects setup-time tensor layouts from later caller mutation.
    def signature : UInt64
      hash = 14_695_981_039_346_656_037_u64
      hash = signature_word(hash, sample_rate)
      hash = signature_word(hash, input_samples)
      hash = signature_array(hash, conv_dims)
      hash = signature_array(hash, conv_kernels)
      hash = signature_array(hash, conv_strides)
      hash = signature_word(hash, hidden_size)
      hash = signature_word(hash, intermediate_size)
      hash = signature_word(hash, heads)
      hash = signature_word(hash, layers)
      hash = signature_word(hash, position_kernel)
      hash = signature_word(hash, position_groups)
      hash = signature_string(hash, feature_norm)
      hash = signature_word(hash, stable_layer_norm ? 1 : 0)
      hash = signature_word(hash, conv_bias ? 1 : 0)
      hash = signature_string(hash, feature_activation)
      hash = signature_string(hash, hidden_activation)
      hash = signature_word(hash, layer_norm_eps.unsafe_as(UInt32))
      hash = signature_word(hash, video_fps)
      hash = signature_word(hash, target_fps)
      signature_word(hash, target_frames)
    end

    private def signature_word(hash : UInt64, value : Int32) : UInt64
      (hash ^ value.to_u32.to_u64) &* 1_099_511_628_211_u64
    end

    private def signature_word(hash : UInt64, value : UInt32) : UInt64
      (hash ^ value.to_u64) &* 1_099_511_628_211_u64
    end

    private def signature_array(hash : UInt64, values : Array(Int32)) : UInt64
      state = signature_word(hash, values.size.to_i32)
      values.each { |value| state = signature_word(state, value) }
      state
    end

    private def signature_string(hash : UInt64, value : String) : UInt64
      state = signature_word(hash, value.bytesize.to_i32)
      value.each_byte { |byte| state = (state ^ byte.to_u64) &* 1_099_511_628_211_u64 }
      state
    end

    def convolution_output_lengths : Array(Int32)
      values = [] of Int32
      current = input_samples
      conv_kernels.each_with_index do |kernel, i|
        current = (current - kernel) // conv_strides[i] + 1
        raise ArgumentError.new("Wav2Vec2 feature convolution has empty output") unless current > 0
        values << current
      end
      values
    end

    def total_conv_stride : Int32
      stride = 1
      conv_strides.each { |value| stride *= value }
      stride
    end

    def feature_rate : Int32
      sample_rate // total_conv_stride
    end

    def feature_frames : Int32
      current = input_samples
      conv_kernels.each_with_index do |kernel, i|
        current = (current - kernel) // conv_strides[i] + 1
      end
      current
    end

    def max_conv_values : Int32
      first_length = (input_samples - conv_kernels[0]) // conv_strides[0] + 1
      first_length * conv_dims[0]
    end

    def interpolation_frames : Int32
      (feature_frames.to_i64 * video_fps // feature_rate).to_i32
    end

    # Mirrors upstream `min_batch_num = int(T / (batch_frames * fps/30)) + 1`.
    def bucket_repeats : Int32
      (interpolation_frames.to_i64 * target_fps // (target_frames.to_i64 * video_fps) + 1).to_i32
    end

    def bucket_frames : Int32
      (bucket_repeats.to_i64 * target_frames).to_i32
    end

    # Number of synthetic 30-fps frames appended before upstream sampling.
    def padded_interpolation_frames : Int32
      needed = bucket_frames.to_i64 * video_fps
      ceil_div(needed, target_fps).to_i32 - interpolation_frames
    end

    # Exact `np.round` tie-to-even sample index for fixed_start=0.
    def bucket_source_index(bucket : Int32) : Int32
      raise ArgumentError.new("Invalid audio bucket index") unless bucket >= 0 && bucket < bucket_frames
      numerator = bucket.to_i64 * video_fps
      quotient = numerator // target_fps
      remainder = numerator % target_fps
      twice = remainder * 2
      quotient += 1 if twice > target_fps || (twice == target_fps && quotient.odd?)
      quotient.to_i32
    end

    def bucket_source_valid?(bucket : Int32) : Bool
      bucket_source_index(bucket) < interpolation_frames
    end

    def sequence_values : Int32
      feature_frames * hidden_size
    end

    def ffn_values : Int32
      feature_frames * intermediate_size
    end

    def hidden_state_count : Int32
      layers + 1
    end

    def output_values : Int32
      hidden_state_count * hidden_size * bucket_frames
    end

    # Scratch is setup-time persistent storage. The output buffer is supplied
    # by the caller and therefore excluded from this value.
    def scratch_bytes : Int64
      lengths = convolution_output_lengths
      max_conv = lengths[0].to_i64 * conv_dims[0]
      frame = feature_frames.to_i64 * hidden_size
      ffn = feature_frames.to_i64 * intermediate_size
      # Includes a centered-waveform temporary and two scalar slots used by
      # the optional eager-operation-boundary MPSGraph normalizer.
      (2_i64 * input_samples + 2_i64 * max_conv + 9_i64 * frame + 2_i64 * ffn + position_kernel + 2) * 4
    end

    # Read the pinned Hugging Face JSON config while fixing only audio/video
    # request geometry. No weights or Python objects are loaded here.
    def self.from_hf_json(path : String, input_samples : Int32 = 80_000,
                          target_frames : Int32 = 84) : self
      json = JSON.parse(File.read(path))
      get_i = ->(key : String) { json[key].as_i.to_i32 }
      get_s = ->(key : String) { json[key].as_s }
      get_b = ->(key : String) { json[key].as_bool }
      cfg = new(
        input_samples: input_samples,
        conv_dims: json["conv_dim"].as_a.map(&.as_i.to_i32),
        conv_kernels: json["conv_kernel"].as_a.map(&.as_i.to_i32),
        conv_strides: json["conv_stride"].as_a.map(&.as_i.to_i32),
        hidden_size: get_i.call("hidden_size"),
        intermediate_size: get_i.call("intermediate_size"),
        heads: get_i.call("num_attention_heads"),
        layers: get_i.call("num_hidden_layers"),
        position_kernel: get_i.call("num_conv_pos_embeddings"),
        position_groups: get_i.call("num_conv_pos_embedding_groups"),
        feature_norm: get_s.call("feat_extract_norm"),
        stable_layer_norm: get_b.call("do_stable_layer_norm"),
        conv_bias: get_b.call("conv_bias"),
        feature_activation: get_s.call("feat_extract_activation"),
        hidden_activation: get_s.call("hidden_act"),
        layer_norm_eps: json["layer_norm_eps"].as_f.to_f32,
        target_frames: target_frames
      )
      cfg.validate!
      cfg
    end

    private def ceil_div(n : Int64, d : Int32) : Int64
      (n + d - 1) // d
    end
  end

  record WanAudioLinearWeights, weight : ML::MetalBuffer, bias : ML::MetalBuffer
  record WanAudioConvWeights, weight : ML::MetalBuffer, bias : ML::MetalBuffer,
    norm_weight : ML::MetalBuffer, norm_bias : ML::MetalBuffer
  record WanAudioLayerWeights,
    attention_norm_weight : ML::MetalBuffer, attention_norm_bias : ML::MetalBuffer,
    query : WanAudioLinearWeights, key : WanAudioLinearWeights,
    value : WanAudioLinearWeights, attention_output : WanAudioLinearWeights,
    ffn_norm_weight : ML::MetalBuffer, ffn_norm_bias : ML::MetalBuffer,
    ffn_in : WanAudioLinearWeights, ffn_out : WanAudioLinearWeights
  record WanAudioEncoderWeights,
    conv_layers : Array(WanAudioConvWeights),
    feature_norm_weight : ML::MetalBuffer, feature_norm_bias : ML::MetalBuffer,
    feature_projection : WanAudioLinearWeights,
    position_weight_g : ML::MetalBuffer, position_weight_v : ML::MetalBuffer,
    position_bias : ML::MetalBuffer,
    layers : Array(WanAudioLayerWeights),
    final_norm_weight : ML::MetalBuffer, final_norm_bias : ML::MetalBuffer,
    config_signature : UInt64

  record WanAudioEncoderScratch,
    normalized_waveform : ML::MetalBuffer,
    normalization_scalar_a : ML::MetalBuffer, normalization_scalar_b : ML::MetalBuffer,
    normalization_centered : ML::MetalBuffer,
    conv_a : ML::MetalBuffer, conv_b : ML::MetalBuffer,
    current : ML::MetalBuffer, residual : ML::MetalBuffer,
    position : ML::MetalBuffer, position_pre_gelu : ML::MetalBuffer,
    query : ML::MetalBuffer, key : ML::MetalBuffer, value : ML::MetalBuffer,
    attention : ML::MetalBuffer, projection : ML::MetalBuffer,
    ffn : ML::MetalBuffer, ffn_gelu : ML::MetalBuffer, position_scale : ML::MetalBuffer,
    config_signature : UInt64

  # Retains seven shape-specialized MPSGraph Conv1d executables and their
  # caller-owned resident buffers. Prepare once after weights and scratch;
  # release only after every command buffer using these plans has completed.
  # Plans are bound to one fixed scratch set, so calls using one instance must
  # be serialized and must not overlap an earlier in-flight encode.
  class WanAudioMPSPlans
    getter config_signature : UInt64
    getter? released : Bool
    @handles : Array(Pointer(Void))
    @inputs : Array(ML::MetalBuffer)
    @weights : Array(ML::MetalBuffer)
    @biases : Array(ML::MetalBuffer)
    @outputs : Array(ML::MetalBuffer)
    getter? graph_gelu : Bool

    def initialize(cfg : WanAudioEncoderConfig, weights : WanAudioEncoderWeights,
                   scratch : WanAudioEncoderScratch, graph_gelu : Bool = false)
      cfg.validate!
      signature = cfg.signature
      unless signature == weights.config_signature && signature == scratch.config_signature
        raise ArgumentError.new("Wan Wav2Vec2 config changed after weights or scratch setup")
      end
      unless cfg.conv_dims.size == weights.conv_layers.size
        raise ArgumentError.new("Wan Wav2Vec2 convolution plans do not match bound weights")
      end

      @config_signature = signature
      @released = false
      @graph_gelu = graph_gelu
      @handles = [] of Pointer(Void)
      @inputs = [] of ML::MetalBuffer
      @weights = [] of ML::MetalBuffer
      @biases = [] of ML::MetalBuffer
      @outputs = [] of ML::MetalBuffer

      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        device = ML::Metal::Device.instance.device_handle
        raise "Metal device is unavailable for Wan audio MPS plans" if device.null?
        lengths = cfg.convolution_output_lengths
        input_len = cfg.input_samples
        input_channels = 1
        cfg.conv_dims.each_with_index do |output_channels, i|
          input = if i == 0
                    scratch.normalized_waveform
                  elsif graph_gelu
                    scratch.conv_b
                  elsif i.odd?
                    scratch.conv_a
                  else
                    scratch.conv_b
                  end
          output = if graph_gelu || i.even?
                     scratch.conv_a
                   else
                     scratch.conv_b
                   end
          layer = weights.conv_layers[i]
          handle = WanAudioMPSFFI.plan_create(device, input.handle, layer.weight.handle,
            layer.bias.handle, output.handle, input_len, input_channels,
            output_channels, cfg.conv_kernels[i], cfg.conv_strides[i])
          if handle.null?
            release
            raise "Failed to prepare Wan audio MPS Conv1d plan #{i}"
          end
          @handles << handle
          @inputs << input
          @weights << layer.weight
          @biases << layer.bias
          @outputs << output
          input_len = lengths[i]
          input_channels = output_channels
        end
      {% else %}
        raise ArgumentError.new("Wan audio MPS plans require macOS Metal Performance Shaders")
      {% end %}
    end

    def size : Int32
      @handles.size
    end

    def validate!(cfg : WanAudioEncoderConfig, weights : WanAudioEncoderWeights,
                  scratch : WanAudioEncoderScratch) : Nil
      raise ArgumentError.new("Wan audio MPS plans were released") if @released
      unless cfg.signature == @config_signature && weights.config_signature == @config_signature &&
             scratch.config_signature == @config_signature && @handles.size == cfg.conv_dims.size
        raise ArgumentError.new("Wan audio MPS plans do not match encoder config")
      end
      cfg.conv_dims.each_index do |i|
        expected_input = if i == 0
                           scratch.normalized_waveform
                         elsif @graph_gelu
                           scratch.conv_b
                         elsif i.odd?
                           scratch.conv_a
                         else
                           scratch.conv_b
                         end
        expected_output = if @graph_gelu || i.even?
                            scratch.conv_a
                          else
                            scratch.conv_b
                          end
        layer = weights.conv_layers[i]
        unless @inputs[i].handle == expected_input.handle && @outputs[i].handle == expected_output.handle &&
               @weights[i].handle == layer.weight.handle && @biases[i].handle == layer.bias.handle
          raise ArgumentError.new("Wan audio MPS plan buffers do not match encoder workspace")
        end
        handle(i)
      end
    end

    def handle(index : Int32) : Pointer(Void)
      raise ArgumentError.new("Wan audio MPS plans were released") if @released
      plan = @handles[index]?
      raise ArgumentError.new("Invalid Wan audio MPS Conv1d plan index") unless plan && !plan.null?
      plan
    end

    def encode(index : Int32, command_handle : Pointer(Void)) : Nil
      raise ArgumentError.new("Wan audio MPS plans were released") if @released
      plan = handle(index)
      raise ArgumentError.new("Wan audio MPS command handle is null") if command_handle.null?
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        status = WanAudioMPSFFI.encode(plan, command_handle)
        raise "Wan audio MPS Conv1d encode rejected (#{status})" unless status == 0
      {% else %}
        raise ArgumentError.new("Wan audio MPS plans require macOS Metal Performance Shaders")
      {% end %}
    end

    # Call only after the command chain's terminal completion; the MPS graph
    # retains TensorData references while queued work is in flight.
    def release : Nil
      return if @released
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        @handles.each { |handle| WanAudioMPSFFI.plan_release(handle) unless handle.null? }
      {% end %}
      @handles.clear
      @inputs.clear
      @weights.clear
      @biases.clear
      @outputs.clear
      @released = true
    end

    def finalize
      release
    end
  end

  # Optional source-shaped MPSGraph waveform normalization. The plan is bound
  # to one caller-owned waveform and one scratch output; create it before
  # encode, serialize uses, and release after the chain's terminal wait.
  class WanAudioMPSNormalizationPlan
    getter config_signature : UInt64
    getter? released : Bool
    @handle : Pointer(Void)
    @input : ML::MetalBuffer
    @output : ML::MetalBuffer
    @source_boundaries : Bool
    @scalar_a : ML::MetalBuffer?
    @scalar_b : ML::MetalBuffer?
    @centered : ML::MetalBuffer?

    def initialize(cfg : WanAudioEncoderConfig, waveform : ML::MetalBuffer,
                   scratch : WanAudioEncoderScratch, source_boundaries : Bool = false)
      cfg.validate!
      signature = cfg.signature
      unless signature == scratch.config_signature
        raise ArgumentError.new("Wan waveform normalization config changed after scratch setup")
      end
      MetalOps.check(waveform, cfg.input_samples.to_i64)
      MetalOps.check(scratch.normalized_waveform, cfg.input_samples.to_i64)
      MetalOps.distinct(waveform, scratch.normalized_waveform,
        scratch.normalization_scalar_a, scratch.normalization_scalar_b,
        scratch.normalization_centered)
      if source_boundaries
        MetalOps.check(scratch.normalization_scalar_a, 1_i64)
        MetalOps.check(scratch.normalization_scalar_b, 1_i64)
        MetalOps.check(scratch.normalization_centered, cfg.input_samples.to_i64)
      end

      @config_signature = signature
      @released = false
      @handle = Pointer(Void).null
      @input = waveform
      @output = scratch.normalized_waveform
      @source_boundaries = source_boundaries
      @scalar_a = source_boundaries ? scratch.normalization_scalar_a : nil
      @scalar_b = source_boundaries ? scratch.normalization_scalar_b : nil
      @centered = source_boundaries ? scratch.normalization_centered : nil
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        device = ML::Metal::Device.instance.device_handle
        raise "Metal device is unavailable for Wan waveform normalization" if device.null?
        @handle = if source_boundaries
                    WanAudioMPSFFI.normalize_boundary_plan_create(device, waveform.handle,
                      scratch.normalized_waveform.handle, scratch.normalization_scalar_a.handle,
                      scratch.normalization_scalar_b.handle, scratch.normalization_centered.handle,
                      cfg.input_samples)
                  else
                    WanAudioMPSFFI.normalize_plan_create(device, waveform.handle,
                      scratch.normalized_waveform.handle, cfg.input_samples)
                  end
        if @handle.null?
          raise "Failed to prepare Wan waveform normalization MPSGraph plan"
        end
      {% else %}
        raise ArgumentError.new("Wan waveform normalization MPS plan requires macOS Metal Performance Shaders")
      {% end %}
    end

    def validate!(cfg : WanAudioEncoderConfig, waveform : ML::MetalBuffer,
                  scratch : WanAudioEncoderScratch) : Nil
      raise ArgumentError.new("Wan waveform normalization plan was released") if @released
      unless cfg.signature == @config_signature && scratch.config_signature == @config_signature &&
             waveform.handle == @input.handle && scratch.normalized_waveform.handle == @output.handle &&
             (!@source_boundaries || (@scalar_a.try(&.handle) == scratch.normalization_scalar_a.handle &&
             @scalar_b.try(&.handle) == scratch.normalization_scalar_b.handle &&
             @centered.try(&.handle) == scratch.normalization_centered.handle))
        raise ArgumentError.new("Wan waveform normalization plan does not match encoder workspace")
      end
      raise ArgumentError.new("Wan waveform normalization plan handle is null") if @handle.null?
    end

    # Caller must pause the MPSCommandChain first. This only queues the
    # precompiled graph; it never allocates buffers, submits, waits, or reads.
    def encode(command_handle : Pointer(Void)) : Nil
      raise ArgumentError.new("Wan waveform normalization plan was released") if @released
      raise ArgumentError.new("Wan waveform normalization command handle is null") if command_handle.null?
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        status = @source_boundaries ? WanAudioMPSFFI.normalize_boundary_encode(@handle, command_handle) : WanAudioMPSFFI.normalize_encode(@handle, command_handle)
        raise "Wan waveform normalization MPSGraph encode rejected (#{status})" unless status == 0
      {% else %}
        raise ArgumentError.new("Wan waveform normalization MPS plan requires macOS Metal Performance Shaders")
      {% end %}
    end

    # Call only after the command chain's terminal completion.
    def release : Nil
      return if @released
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        unless @handle.null?
          if @source_boundaries
            WanAudioMPSFFI.normalize_boundary_plan_release(@handle)
          else
            WanAudioMPSFFI.normalize_plan_release(@handle)
          end
        end
      {% end %}
      @handle = Pointer(Void).null
      @released = true
    end

    def finalize
      release
    end
  end

  # Optional source-shaped exact GELU plans. Seven plans cover the feature
  # convolution shapes; one covers the encoder FFN shape. They are bound to
  # the fixed resident scratch workspace and enqueue into the caller's chain.
  class WanAudioMPSGeluPlans
    getter config_signature : UInt64
    getter? released : Bool
    @handles : Array(Pointer(Void))
    @inputs : Array(ML::MetalBuffer)
    @outputs : Array(ML::MetalBuffer)

    def initialize(cfg : WanAudioEncoderConfig, scratch : WanAudioEncoderScratch)
      cfg.validate!
      unless cfg.signature == scratch.config_signature
        raise ArgumentError.new("Wan GELU config changed after scratch setup")
      end
      MetalOps.check(scratch.conv_a, cfg.max_conv_values.to_i64)
      MetalOps.check(scratch.conv_b, cfg.max_conv_values.to_i64)
      MetalOps.check(scratch.ffn, cfg.ffn_values.to_i64)
      MetalOps.check(scratch.ffn_gelu, cfg.ffn_values.to_i64)
      MetalOps.distinct(scratch.conv_a, scratch.conv_b)
      MetalOps.distinct(scratch.ffn, scratch.ffn_gelu)

      @config_signature = cfg.signature
      @released = false
      @handles = [] of Pointer(Void)
      @inputs = [] of ML::MetalBuffer
      @outputs = [] of ML::MetalBuffer
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        device = ML::Metal::Device.instance.device_handle
        raise "Metal device is unavailable for Wan audio GELU plans" if device.null?
        cfg.convolution_output_lengths.each_with_index do |length, i|
          count = length.to_i64 * cfg.conv_dims[i]
          raise ArgumentError.new("Wan feature GELU shape exceeds native addressable storage") if count > Int32::MAX
          handle = WanAudioMPSFFI.gelu_plan_create(device, scratch.conv_a.handle,
            scratch.conv_b.handle, count.to_i32)
          if handle.null?
            release
            raise "Failed to prepare Wan feature MPSGraph GELU plan #{i}"
          end
          @handles << handle
          @inputs << scratch.conv_a
          @outputs << scratch.conv_b
        end
        ffn = WanAudioMPSFFI.gelu_plan_create(device, scratch.ffn.handle,
          scratch.ffn_gelu.handle, cfg.ffn_values)
        if ffn.null?
          release
          raise "Failed to prepare Wan encoder FFN MPSGraph GELU plan"
        end
        @handles << ffn
        @inputs << scratch.ffn
        @outputs << scratch.ffn_gelu
      {% else %}
        raise ArgumentError.new("Wan audio MPSGraph GELU plans require macOS Metal Performance Shaders")
      {% end %}
    end

    def feature_count : Int32
      @handles.size - 1
    end

    def validate!(cfg : WanAudioEncoderConfig, scratch : WanAudioEncoderScratch) : Nil
      raise ArgumentError.new("Wan audio GELU plans were released") if @released
      unless cfg.signature == @config_signature && scratch.config_signature == @config_signature &&
             @handles.size == cfg.conv_dims.size + 1
        raise ArgumentError.new("Wan audio GELU plans do not match encoder config")
      end
      cfg.conv_dims.each_index do |i|
        unless @inputs[i].handle == scratch.conv_a.handle && @outputs[i].handle == scratch.conv_b.handle
          raise ArgumentError.new("Wan feature GELU plan buffers do not match encoder workspace")
        end
        handle(i)
      end
      unless @inputs.last.handle == scratch.ffn.handle && @outputs.last.handle == scratch.ffn_gelu.handle
        raise ArgumentError.new("Wan encoder GELU plan buffers do not match encoder workspace")
      end
      handle(@handles.size - 1)
    end

    def encode_feature(index : Int32, command_handle : Pointer(Void)) : Nil
      encode(index, command_handle, "feature")
    end

    def encode_ffn(command_handle : Pointer(Void)) : Nil
      encode(@handles.size - 1, command_handle, "encoder FFN")
    end

    private def encode(index : Int32, command_handle : Pointer(Void), label : String) : Nil
      raise ArgumentError.new("Wan audio GELU plans were released") if @released
      raise ArgumentError.new("Wan audio GELU command handle is null") if command_handle.null?
      plan = handle(index)
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        status = WanAudioMPSFFI.gelu_encode(plan, command_handle)
        raise "Wan audio MPSGraph #{label} GELU encode rejected (#{status})" unless status == 0
      {% else %}
        raise ArgumentError.new("Wan audio MPSGraph GELU plans require macOS Metal Performance Shaders")
      {% end %}
    end

    private def handle(index : Int32) : Pointer(Void)
      raise ArgumentError.new("Wan audio GELU plans were released") if @released
      plan = @handles[index]?
      raise ArgumentError.new("Invalid Wan audio MPSGraph GELU plan index") unless plan && !plan.null?
      plan
    end

    # Call only after the chain's terminal completion.
    def release : Nil
      return if @released
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        @handles.each { |handle| WanAudioMPSFFI.gelu_plan_release(handle) unless handle.null? }
      {% end %}
      @handles.clear
      @inputs.clear
      @outputs.clear
      @released = true
    end

    def finalize
      release
    end
  end

  # Prepared grouped positional Conv1d plus exact GELU, bound to one native
  # Wav2Vec2 scratch set. The scale is computed by the existing native
  # weight-normalization kernel and supplied as a graph input, isolating this
  # plan's convolution/activation arithmetic from that reduction.
  class WanAudioMPSPositionPlan
    getter config_signature : UInt64
    getter? released : Bool
    @handle : Pointer(Void)
    @input : ML::MetalBuffer
    @weight_v : ML::MetalBuffer
    @scale : ML::MetalBuffer
    @bias : ML::MetalBuffer
    @raw_output : ML::MetalBuffer
    @output : ML::MetalBuffer

    def initialize(cfg : WanAudioEncoderConfig, weights : WanAudioEncoderWeights,
                   scratch : WanAudioEncoderScratch)
      cfg.validate!
      unless cfg.signature == weights.config_signature && cfg.signature == scratch.config_signature
        raise ArgumentError.new("Wan position plan config changed after weights or scratch setup")
      end
      unless cfg.feature_frames > 0 && cfg.hidden_size > 0 && cfg.position_groups > 0 &&
             cfg.position_kernel > 0
        raise ArgumentError.new("Invalid Wan positional convolution geometry")
      end
      sequence = cfg.sequence_values.to_i64
      MetalOps.check(scratch.current, sequence)
      MetalOps.check(weights.position_weight_v,
        cfg.hidden_size.to_i64 * (cfg.hidden_size // cfg.position_groups) * cfg.position_kernel)
      MetalOps.check(scratch.position_scale, cfg.position_kernel.to_i64)
      MetalOps.check(weights.position_bias, cfg.hidden_size.to_i64)
      MetalOps.check(scratch.position_pre_gelu, sequence)
      MetalOps.check(scratch.position, sequence)
      MetalOps.distinct(scratch.current, weights.position_weight_v, scratch.position_scale,
        weights.position_bias, scratch.position_pre_gelu, scratch.position)

      @config_signature = cfg.signature
      @released = false
      @handle = Pointer(Void).null
      @input = scratch.current
      @weight_v = weights.position_weight_v
      @scale = scratch.position_scale
      @bias = weights.position_bias
      @raw_output = scratch.position_pre_gelu
      @output = scratch.position
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        device = ML::Metal::Device.instance.device_handle
        raise "Metal device is unavailable for Wan positional MPS plan" if device.null?
        @handle = WanAudioMPSFFI.position_plan_create(device, @input.handle,
          @weight_v.handle, @scale.handle, @bias.handle, @raw_output.handle,
          @output.handle, cfg.feature_frames, cfg.hidden_size, cfg.position_groups,
          cfg.position_kernel)
        if @handle.null?
          raise "Failed to prepare Wan positional Conv1d MPSGraph plan"
        end
      {% else %}
        raise ArgumentError.new("Wan positional MPS plan requires macOS Metal Performance Shaders")
      {% end %}
    end

    def validate!(cfg : WanAudioEncoderConfig, weights : WanAudioEncoderWeights,
                  scratch : WanAudioEncoderScratch) : Nil
      raise ArgumentError.new("Wan positional MPS plan was released") if @released
      unless cfg.signature == @config_signature && weights.config_signature == @config_signature &&
             scratch.config_signature == @config_signature &&
             @input.handle == scratch.current.handle &&
             @weight_v.handle == weights.position_weight_v.handle &&
             @scale.handle == scratch.position_scale.handle &&
             @bias.handle == weights.position_bias.handle &&
             @raw_output.handle == scratch.position_pre_gelu.handle &&
             @output.handle == scratch.position.handle
        raise ArgumentError.new("Wan positional MPS plan does not match encoder workspace")
      end
      raise ArgumentError.new("Wan positional MPS plan handle is null") if @handle.null?
    end

    # Caller pauses the MPSCommandChain first. This only queues the prepared
    # graph; it performs no host/GPU allocation, submit, wait, or readback.
    def encode(command_handle : Pointer(Void)) : Nil
      raise ArgumentError.new("Wan positional MPS plan was released") if @released
      raise ArgumentError.new("Wan positional MPS command handle is null") if command_handle.null?
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        status = WanAudioMPSFFI.position_encode(@handle, command_handle)
        raise "Wan positional MPSGraph encode rejected (#{status})" unless status == 0
      {% else %}
        raise ArgumentError.new("Wan positional MPS plan requires macOS Metal Performance Shaders")
      {% end %}
    end

    # Call only after the chain's terminal completion.
    def release : Nil
      return if @released
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        WanAudioMPSFFI.position_plan_release(@handle) unless @handle.null?
      {% end %}
      @handle = Pointer(Void).null
      @released = true
    end

    def finalize
      release
    end
  end

  # Optional pre-attention and first-block diagnostics. Buffers are allocated
  # by prepare_trace! and remain caller-owned/resident; ordinary production
  # encode passes nil.
  record WanAudioEncoderTrace,
    normalized_waveform : ML::MetalBuffer,
    first_conv_raw : ML::MetalBuffer,
    first_conv_norm : ML::MetalBuffer,
    second_conv_raw : ML::MetalBuffer,
    feature_convolutions : Array(ML::MetalBuffer),
    feature_norm : ML::MetalBuffer,
    feature_projection : ML::MetalBuffer,
    position_scale : ML::MetalBuffer,
    position_conv_raw : ML::MetalBuffer,
    position_conv : ML::MetalBuffer,
    position_add : ML::MetalBuffer,
    pre_attention_norm : ML::MetalBuffer,
    query : ML::MetalBuffer, key : ML::MetalBuffer, value : ML::MetalBuffer,
    attention : ML::MetalBuffer, attention_projection : ML::MetalBuffer,
    attention_residual : ML::MetalBuffer, ffn_norm : ML::MetalBuffer,
    ffn_pre_gelu : ML::MetalBuffer, ffn_post_gelu : ML::MetalBuffer,
    ffn_output : ML::MetalBuffer, layer_output : ML::MetalBuffer,
    config_signature : UInt64

  module WanAudioEncoder
    SOURCE                    = {{ read_file("#{__DIR__}/kernels/wan_audio_encoder.metal") }}
    HF_TRANSFORMERS_VERSION   = "4.57.3"
    HF_MODEL_REVISION         = "dab4e9c55bbe4c8c4d03db1c2c98c7f0ac9c454b"
    HF_MODELING_SOURCE_SHA256 = "1c19d14f99c623c71aa8a058235978a06a8af4defb5e1f54321f7a6b7984e572"
    SOURCE_PREFIX             = "wav2vec2."

    def self.pipeline(name : String) : ML::Metal::ComputePipeline
      ML::Metal::PipelineCache.get(name) { ML::Metal::ComputePipeline.new(name, SOURCE) }
    end

    def self.prepare! : Nil
      MetalOps.prepare!
      {"wan_audio_layer_norm_mps", "wan_audio_normalize", "wan_audio_conv1d", "wan_audio_gelu_mps", "wan_audio_position_scale",
       "wan_audio_position_conv", "wan_audio_bucket"}.each { |name| pipeline(name) }
    end

    # State-dict bindings follow Transformers 4.57.3 Wav2Vec2ForCTC names.
    # The CTC lm_head is unused because S2V consumes hidden states. The
    # masked_spec_embed is also unused on the inference path (eval, no supplied
    # mask_time_indices), even if the checkpoint config enables SpecAugment.
    def self.checkpoint_bindings(cfg : WanAudioEncoderConfig,
                                 dtype : ML::GGUF::SafeTensorDType? = nil,
                                 prefix : String = "audio.") : Array(WeightBinding)
      cfg.validate!
      result = [] of WeightBinding
      cfg.conv_dims.each_with_index do |out_dim, i|
        in_dim = i == 0 ? 1_i64 : cfg.conv_dims[i - 1].to_i64
        conv = "feature_extractor.conv_layers.#{i}."
        add_binding(result, prefix + "conv_#{i}.weight", conv + "conv.weight",
          [out_dim.to_i64, in_dim, cfg.conv_kernels[i].to_i64], dtype)
        add_binding(result, prefix + "conv_#{i}.bias", conv + "conv.bias", [out_dim.to_i64], dtype)
        add_binding(result, prefix + "conv_#{i}.norm.weight", conv + "layer_norm.weight", [out_dim.to_i64], dtype)
        add_binding(result, prefix + "conv_#{i}.norm.bias", conv + "layer_norm.bias", [out_dim.to_i64], dtype)
      end
      add_binding(result, prefix + "feature_norm.weight", "feature_projection.layer_norm.weight",
        [cfg.conv_dims.last.to_i64], dtype)
      add_binding(result, prefix + "feature_norm.bias", "feature_projection.layer_norm.bias",
        [cfg.conv_dims.last.to_i64], dtype)
      add_binding(result, prefix + "feature_projection.weight", "feature_projection.projection.weight",
        [cfg.hidden_size.to_i64, cfg.conv_dims.last.to_i64], dtype)
      add_binding(result, prefix + "feature_projection.bias", "feature_projection.projection.bias",
        [cfg.hidden_size.to_i64], dtype)
      add_binding(result, prefix + "position.weight_g", "encoder.pos_conv_embed.conv.weight_g",
        [1_i64, 1_i64, cfg.position_kernel.to_i64], dtype)
      add_binding(result, prefix + "position.weight_v", "encoder.pos_conv_embed.conv.weight_v",
        [cfg.hidden_size.to_i64, (cfg.hidden_size // cfg.position_groups).to_i64, cfg.position_kernel.to_i64], dtype)
      add_binding(result, prefix + "position.bias", "encoder.pos_conv_embed.conv.bias", [cfg.hidden_size.to_i64], dtype)
      cfg.layers.times do |i|
        base = "encoder.layers.#{i}."
        add_binding(result, prefix + "layer_#{i}.attention_norm.weight", base + "layer_norm.weight", [cfg.hidden_size.to_i64], dtype)
        add_binding(result, prefix + "layer_#{i}.attention_norm.bias", base + "layer_norm.bias", [cfg.hidden_size.to_i64], dtype)
        add_linear_bindings(result, prefix + "layer_#{i}.query", base + "attention.q_proj", cfg.hidden_size, cfg.hidden_size, dtype)
        add_linear_bindings(result, prefix + "layer_#{i}.key", base + "attention.k_proj", cfg.hidden_size, cfg.hidden_size, dtype)
        add_linear_bindings(result, prefix + "layer_#{i}.value", base + "attention.v_proj", cfg.hidden_size, cfg.hidden_size, dtype)
        add_linear_bindings(result, prefix + "layer_#{i}.attention_output", base + "attention.out_proj", cfg.hidden_size, cfg.hidden_size, dtype)
        add_binding(result, prefix + "layer_#{i}.ffn_norm.weight", base + "final_layer_norm.weight", [cfg.hidden_size.to_i64], dtype)
        add_binding(result, prefix + "layer_#{i}.ffn_norm.bias", base + "final_layer_norm.bias", [cfg.hidden_size.to_i64], dtype)
        add_linear_bindings(result, prefix + "layer_#{i}.ffn_in", base + "feed_forward.intermediate_dense", cfg.intermediate_size, cfg.hidden_size, dtype)
        add_linear_bindings(result, prefix + "layer_#{i}.ffn_out", base + "feed_forward.output_dense", cfg.hidden_size, cfg.intermediate_size, dtype)
      end
      add_binding(result, prefix + "final_norm.weight", "encoder.layer_norm.weight", [cfg.hidden_size.to_i64], dtype)
      add_binding(result, prefix + "final_norm.bias", "encoder.layer_norm.bias", [cfg.hidden_size.to_i64], dtype)
      result
    end

    # Resolve and validate all references once during setup. Encode receives
    # only immutable handles; it constructs no names or intermediate arrays.
    def self.bind_weights(arena : Arena, cfg : WanAudioEncoderConfig,
                          prefix : String = "audio.") : WanAudioEncoderWeights
      cfg.validate!
      convs = [] of WanAudioConvWeights
      cfg.conv_dims.each_with_index do |dim, i|
        p = prefix + "conv_#{i}."
        convs << WanAudioConvWeights.new(
          resident_f32(arena, p + "weight", dim.to_i64 * (i == 0 ? 1_i64 : cfg.conv_dims[i - 1]) * cfg.conv_kernels[i]),
          resident_f32(arena, p + "bias", dim.to_i64),
          resident_f32(arena, p + "norm.weight", dim.to_i64),
          resident_f32(arena, p + "norm.bias", dim.to_i64))
      end
      last_conv = cfg.conv_dims.last
      projection = linear_weights(arena, prefix + "feature_projection", cfg.hidden_size.to_i64 * last_conv,
        cfg.hidden_size.to_i64)
      layers = [] of WanAudioLayerWeights
      cfg.layers.times do |i|
        p = prefix + "layer_#{i}."
        layers << WanAudioLayerWeights.new(
          resident_f32(arena, p + "attention_norm.weight", cfg.hidden_size.to_i64),
          resident_f32(arena, p + "attention_norm.bias", cfg.hidden_size.to_i64),
          linear_weights(arena, p + "query", cfg.hidden_size.to_i64 * cfg.hidden_size, cfg.hidden_size.to_i64),
          linear_weights(arena, p + "key", cfg.hidden_size.to_i64 * cfg.hidden_size, cfg.hidden_size.to_i64),
          linear_weights(arena, p + "value", cfg.hidden_size.to_i64 * cfg.hidden_size, cfg.hidden_size.to_i64),
          linear_weights(arena, p + "attention_output", cfg.hidden_size.to_i64 * cfg.hidden_size, cfg.hidden_size.to_i64),
          resident_f32(arena, p + "ffn_norm.weight", cfg.hidden_size.to_i64),
          resident_f32(arena, p + "ffn_norm.bias", cfg.hidden_size.to_i64),
          linear_weights(arena, p + "ffn_in", cfg.intermediate_size.to_i64 * cfg.hidden_size, cfg.intermediate_size.to_i64),
          linear_weights(arena, p + "ffn_out", cfg.hidden_size.to_i64 * cfg.intermediate_size, cfg.hidden_size.to_i64))
      end
      WanAudioEncoderWeights.new(
        convs,
        resident_f32(arena, prefix + "feature_norm.weight", last_conv.to_i64),
        resident_f32(arena, prefix + "feature_norm.bias", last_conv.to_i64),
        projection,
        resident_f32(arena, prefix + "position.weight_g", cfg.position_kernel.to_i64),
        resident_f32(arena, prefix + "position.weight_v",
          cfg.hidden_size.to_i64 * (cfg.hidden_size // cfg.position_groups) * cfg.position_kernel),
        resident_f32(arena, prefix + "position.bias", cfg.hidden_size.to_i64),
        layers,
        resident_f32(arena, prefix + "final_norm.weight", cfg.hidden_size.to_i64),
        resident_f32(arena, prefix + "final_norm.bias", cfg.hidden_size.to_i64),
        cfg.signature)
    end

    def self.prepare_scratch!(arena : Arena, cfg : WanAudioEncoderConfig,
                              prefix : String = "audio.s.") : WanAudioEncoderScratch
      cfg.validate!
      prepare!
      lengths = cfg.convolution_output_lengths
      conv_max = lengths[0] * cfg.conv_dims[0]
      frame = cfg.sequence_values
      arena.alloc(prefix + "normalized_waveform", cfg.input_samples)
      arena.alloc(prefix + "normalization_scalar_a", 1)
      arena.alloc(prefix + "normalization_scalar_b", 1)
      arena.alloc(prefix + "normalization_centered", cfg.input_samples)
      conv_a = arena.alloc(prefix + "conv_a", conv_max)
      conv_b = arena.alloc(prefix + "conv_b", conv_max)
      current = arena.alloc(prefix + "current", frame)
      residual = arena.alloc(prefix + "residual", frame)
      position = arena.alloc(prefix + "position", frame)
      position_pre_gelu = arena.alloc(prefix + "position_pre_gelu", frame)
      query = arena.alloc(prefix + "query", frame)
      key = arena.alloc(prefix + "key", frame)
      value = arena.alloc(prefix + "value", frame)
      attention = arena.alloc(prefix + "attention", frame)
      projection = arena.alloc(prefix + "projection", frame)
      ffn = arena.alloc(prefix + "ffn", cfg.ffn_values)
      ffn_gelu = arena.alloc(prefix + "ffn_gelu", cfg.ffn_values)
      position_scale = arena.alloc(prefix + "position_scale", cfg.position_kernel)
      WanAudioEncoderScratch.new(arena[prefix + "normalized_waveform"],
        arena[prefix + "normalization_scalar_a"], arena[prefix + "normalization_scalar_b"],
        arena[prefix + "normalization_centered"], conv_a, conv_b,
        current, residual, position, position_pre_gelu, query, key, value, attention, projection,
        ffn, ffn_gelu, position_scale, cfg.signature)
    end

    # Test-only resident trace storage. No diagnostic buffers are allocated by
    # encode itself, and all captures are copied on the caller's command buffer.
    def self.prepare_trace!(arena : Arena, cfg : WanAudioEncoderConfig,
                            prefix : String = "audio.trace.") : WanAudioEncoderTrace
      cfg.validate!
      sequence = cfg.sequence_values
      ffn = cfg.ffn_values
      lengths = cfg.convolution_output_lengths
      conv_traces = [] of ML::MetalBuffer
      cfg.conv_dims.each_with_index do |dim, i|
        conv_traces << arena.alloc(prefix + "feature_conv_#{i}", lengths[i] * dim)
      end
      first_conv_values = lengths[0] * cfg.conv_dims[0]
      WanAudioEncoderTrace.new(
        arena.alloc(prefix + "normalized_waveform", cfg.input_samples),
        arena.alloc(prefix + "feature_conv_0_raw", first_conv_values),
        arena.alloc(prefix + "feature_conv_0_norm", first_conv_values),
        arena.alloc(prefix + "feature_conv_1_raw", lengths[1] * cfg.conv_dims[1]),
        conv_traces,
        arena.alloc(prefix + "feature_norm", cfg.feature_frames * cfg.conv_dims.last),
        arena.alloc(prefix + "feature_projection", sequence),
        arena.alloc(prefix + "position_scale", cfg.position_kernel),
        arena.alloc(prefix + "position_conv_raw", sequence),
        arena.alloc(prefix + "position_conv", sequence),
        arena.alloc(prefix + "position_add", sequence),
        arena.alloc(prefix + "pre_attention_norm", sequence),
        arena.alloc(prefix + "query", sequence),
        arena.alloc(prefix + "key", sequence),
        arena.alloc(prefix + "value", sequence),
        arena.alloc(prefix + "attention", sequence),
        arena.alloc(prefix + "attention_projection", sequence),
        arena.alloc(prefix + "attention_residual", sequence),
        arena.alloc(prefix + "ffn_norm", sequence),
        arena.alloc(prefix + "ffn_pre_gelu", ffn),
        arena.alloc(prefix + "ffn_post_gelu", ffn),
        arena.alloc(prefix + "ffn_output", sequence),
        arena.alloc(prefix + "layer_output", sequence),
        cfg.signature)
    end

    # Emit all feature extraction, transformer, interpolation, and bucket work
    # into the caller's existing serial command encoder. This method performs
    # no allocation, readback, submission, or wait.
    def self.encode(e : ML::Metal::ComputeEncoder, cfg : WanAudioEncoderConfig,
                    waveform : ML::MetalBuffer, output : ML::MetalBuffer,
                    weights : WanAudioEncoderWeights, scratch : WanAudioEncoderScratch,
                    trace : WanAudioEncoderTrace? = nil) : Nil
      encode_impl(e, cfg, waveform, output, weights, scratch, trace, nil, nil, nil, nil, nil)
    end

    # MPSGraph Conv1d path. The caller owns the chain and its eventual terminal
    # completion; each graph step pauses the current Metal encoder, queues the
    # prepared executable, and resumes with the chain's refreshed root.
    def self.encode(chain, cfg : WanAudioEncoderConfig, waveform : ML::MetalBuffer,
                    output : ML::MetalBuffer, weights : WanAudioEncoderWeights,
                    scratch : WanAudioEncoderScratch, plans : WanAudioMPSPlans,
                    trace : WanAudioEncoderTrace? = nil,
                    normalizer : WanAudioMPSNormalizationPlan? = nil,
                    gelu_plans : WanAudioMPSGeluPlans? = nil,
                    position_plan : WanAudioMPSPositionPlan? = nil) : Nil
      encode_impl(chain.encoder, cfg, waveform, output, weights, scratch, trace,
        plans, chain, normalizer, gelu_plans, position_plan)
    end

    private def self.encode_impl(e : ML::Metal::ComputeEncoder, cfg : WanAudioEncoderConfig,
                                 waveform : ML::MetalBuffer, output : ML::MetalBuffer,
                                 weights : WanAudioEncoderWeights, scratch : WanAudioEncoderScratch,
                                 trace : WanAudioEncoderTrace?, plans, chain,
                                 normalizer : WanAudioMPSNormalizationPlan?,
                                 gelu_plans : WanAudioMPSGeluPlans?,
                                 position_plan : WanAudioMPSPositionPlan?) : Nil
      signature = cfg.signature
      unless signature == weights.config_signature && signature == scratch.config_signature
        raise ArgumentError.new("Wan Wav2Vec2 config changed after weights or scratch setup")
      end
      cfg.validate!
      plans.validate!(cfg, weights, scratch) if plans
      if conv_plans = plans
        if conv_plans.graph_gelu? != !gelu_plans.nil?
          raise ArgumentError.new("Wan audio Conv1d and GELU MPS plan modes must match")
        end
      elsif gelu_plans
        raise ArgumentError.new("Wan audio MPSGraph GELU plans require MPS Conv1d plans")
      end
      gelu_plans.try(&.validate!(cfg, scratch))
      position_plan.try(&.validate!(cfg, weights, scratch))
      if gelu_plans && !chain
        raise ArgumentError.new("Wan audio MPSGraph GELU plans require an MPSCommandChain")
      end
      if position_plan && !chain
        raise ArgumentError.new("Wan positional MPSGraph plan requires an MPSCommandChain")
      end
      if graph_normalizer = normalizer
        graph_normalizer.validate!(cfg, waveform, scratch)
        unless chain
          raise ArgumentError.new("Wan waveform normalization MPS plan requires an MPSCommandChain")
        end
      end
      seq = cfg.sequence_values
      if diagnostic = trace
        unless diagnostic.config_signature == signature
          raise ArgumentError.new("Wan Wav2Vec2 trace was prepared for a different config")
        end
        MetalOps.check(diagnostic.normalized_waveform, cfg.input_samples.to_i64)
        first_conv_values = cfg.convolution_output_lengths[0].to_i64 * cfg.conv_dims[0]
        MetalOps.check(diagnostic.first_conv_raw, first_conv_values)
        MetalOps.check(diagnostic.first_conv_norm, first_conv_values)
        second_conv_values = cfg.convolution_output_lengths[1].to_i64 * cfg.conv_dims[1]
        MetalOps.check(diagnostic.second_conv_raw, second_conv_values)
        unless diagnostic.feature_convolutions.size == cfg.conv_dims.size
          raise ArgumentError.new("Wan Wav2Vec2 feature trace was prepared for a different config")
        end
        trace_input_len = cfg.input_samples
        cfg.conv_dims.each_with_index do |dim, i|
          output_len = (trace_input_len - cfg.conv_kernels[i]) // cfg.conv_strides[i] + 1
          MetalOps.check(diagnostic.feature_convolutions[i], output_len.to_i64 * dim)
          trace_input_len = output_len
        end
        feature_values = cfg.feature_frames.to_i64 * cfg.conv_dims.last
        MetalOps.check(diagnostic.feature_norm, feature_values)
        MetalOps.check(diagnostic.feature_projection, seq.to_i64)
        MetalOps.check(diagnostic.position_scale, cfg.position_kernel.to_i64)
        MetalOps.check(diagnostic.position_conv_raw, seq.to_i64)
        MetalOps.check(diagnostic.position_conv, seq.to_i64)
        MetalOps.check(diagnostic.position_add, seq.to_i64)
        MetalOps.check(diagnostic.pre_attention_norm, seq.to_i64)
        MetalOps.check(diagnostic.query, seq.to_i64)
        MetalOps.check(diagnostic.key, seq.to_i64)
        MetalOps.check(diagnostic.value, seq.to_i64)
        MetalOps.check(diagnostic.attention, seq.to_i64)
        MetalOps.check(diagnostic.attention_projection, seq.to_i64)
        MetalOps.check(diagnostic.attention_residual, seq.to_i64)
        MetalOps.check(diagnostic.ffn_norm, seq.to_i64)
        MetalOps.check(diagnostic.ffn_pre_gelu, cfg.ffn_values.to_i64)
        MetalOps.check(diagnostic.ffn_post_gelu, cfg.ffn_values.to_i64)
        MetalOps.check(diagnostic.ffn_output, seq.to_i64)
        MetalOps.check(diagnostic.layer_output, seq.to_i64)
      end
      unless cfg.input_samples > 0 && cfg.target_frames > 0 && cfg.layers == weights.layers.size &&
             cfg.conv_dims.size == weights.conv_layers.size && cfg.sequence_values > 0 && cfg.output_values > 0
        raise ArgumentError.new("Wan Wav2Vec2 encode geometry does not match bound weights")
      end
      frames = cfg.feature_frames
      MetalOps.check(waveform, cfg.input_samples.to_i64)
      MetalOps.check(output, cfg.output_values.to_i64)
      MetalOps.check(scratch.normalized_waveform, cfg.input_samples.to_i64)
      MetalOps.check(scratch.normalization_scalar_a, 1_i64)
      MetalOps.check(scratch.normalization_scalar_b, 1_i64)
      MetalOps.check(scratch.normalization_centered, cfg.input_samples.to_i64)
      MetalOps.check(scratch.conv_a, cfg.max_conv_values.to_i64)
      MetalOps.check(scratch.conv_b, cfg.max_conv_values.to_i64)
      MetalOps.check(scratch.current, seq.to_i64)
      MetalOps.check(scratch.residual, seq.to_i64)
      MetalOps.check(scratch.position, seq.to_i64)
      MetalOps.check(scratch.position_pre_gelu, seq.to_i64)
      MetalOps.check(scratch.query, seq.to_i64)
      MetalOps.check(scratch.key, seq.to_i64)
      MetalOps.check(scratch.value, seq.to_i64)
      MetalOps.check(scratch.attention, seq.to_i64)
      MetalOps.check(scratch.projection, seq.to_i64)
      MetalOps.check(scratch.ffn, cfg.ffn_values.to_i64)
      MetalOps.check(scratch.ffn_gelu, cfg.ffn_values.to_i64)
      MetalOps.check(scratch.position_scale, cfg.position_kernel.to_i64)
      MetalOps.distinct(output, waveform, scratch.normalized_waveform, scratch.normalization_scalar_a,
        scratch.normalization_scalar_b, scratch.normalization_centered, scratch.conv_a, scratch.conv_b,
        scratch.current, scratch.residual, scratch.position, scratch.position_pre_gelu, scratch.query, scratch.key,
        scratch.value, scratch.attention, scratch.projection, scratch.ffn, scratch.ffn_gelu, scratch.position_scale)
      MetalOps.distinct(waveform, scratch.normalized_waveform, scratch.normalization_scalar_a,
        scratch.normalization_scalar_b, scratch.normalization_centered, scratch.conv_a, scratch.conv_b,
        scratch.current, scratch.residual, scratch.position, scratch.position_pre_gelu, scratch.query, scratch.key,
        scratch.value, scratch.attention, scratch.projection, scratch.ffn, scratch.ffn_gelu, scratch.position_scale)

      if graph_normalizer = normalizer
        unless mps_chain = chain
          raise ArgumentError.new("Wan waveform normalization MPS plan requires an MPSCommandChain")
        end
        mps_chain.pause!
        graph_normalizer.encode(mps_chain.handle)
        mps_chain.resume!
        e = mps_chain.encoder
      else
        e.set_pipeline(pipeline("wan_audio_normalize"))
        e.set_buffer(waveform, 0).set_buffer(scratch.normalized_waveform, 1, ML::Metal::BufferAccess::Write)
        e.set_value(cfg.input_samples.to_u32, 2).set_value(1e-7_f32, 3)
        e.dispatch_threadgroups({1, 1, 1}, {256, 1, 1})
      end
      if diagnostic = trace
        MetalOps.copy(e, scratch.normalized_waveform, diagnostic.normalized_waveform,
          cfg.input_samples)
      end

      input = scratch.normalized_waveform
      input_len = cfg.input_samples
      input_channels = 1
      use_a = true
      graph_gelu = !gelu_plans.nil?
      cfg.conv_kernels.each_with_index do |kernel_size, i|
        output_len = (input_len - kernel_size) // cfg.conv_strides[i] + 1
        conv_output = graph_gelu ? scratch.conv_a : (use_a ? scratch.conv_a : scratch.conv_b)
        conv_layer = weights.conv_layers[i]
        if conv_plans = plans
          unless mps_chain = chain
            raise ArgumentError.new("Wan audio MPS plans require an MPSCommandChain")
          end
          mps_chain.pause!
          conv_plans.encode(i, mps_chain.handle)
          mps_chain.resume!
          e = mps_chain.encoder
        else
          conv1d(e, input, conv_layer.weight, conv_layer.bias, conv_output,
            input_len, output_len, input_channels, cfg.conv_dims[i], kernel_size, cfg.conv_strides[i])
        end
        if i == 0
          if diagnostic = trace
            MetalOps.copy(e, conv_output, diagnostic.first_conv_raw, output_len * cfg.conv_dims[i])
          end
        elsif i == 1
          if diagnostic = trace
            MetalOps.copy(e, conv_output, diagnostic.second_conv_raw, output_len * cfg.conv_dims[i])
          end
        end
        layer_norm_mps(e, conv_output, conv_layer.norm_weight, conv_layer.norm_bias, conv_output,
          output_len, cfg.conv_dims[i], cfg.layer_norm_eps)
        if i == 0
          if diagnostic = trace
            MetalOps.copy(e, conv_output, diagnostic.first_conv_norm, output_len * cfg.conv_dims[i])
          end
        end
        if feature_gelu = gelu_plans
          unless mps_chain = chain
            raise ArgumentError.new("Wan audio MPSGraph GELU plans require an MPSCommandChain")
          end
          mps_chain.pause!
          feature_gelu.encode_feature(i, mps_chain.handle)
          mps_chain.resume!
          e = mps_chain.encoder
        else
          gelu_mps(e, conv_output, output_len * cfg.conv_dims[i])
        end
        feature_output = graph_gelu ? scratch.conv_b : conv_output
        if diagnostic = trace
          MetalOps.copy(e, feature_output, diagnostic.feature_convolutions[i],
            output_len * cfg.conv_dims[i])
        end
        input = feature_output
        input_len = output_len
        input_channels = cfg.conv_dims[i]
        use_a = !use_a
      end

      # The last feature-convolution output is normalized in place before the
      # 512 -> 1024 projection, matching Wav2Vec2FeatureProjection.
      layer_norm_mps(e, input, weights.feature_norm_weight, weights.feature_norm_bias, input,
        frames, input_channels, cfg.layer_norm_eps)
      if diagnostic = trace
        MetalOps.copy(e, input, diagnostic.feature_norm, frames * input_channels)
      end
      MetalOps.linear(e, input, weights.feature_projection.weight, weights.feature_projection.bias,
        scratch.current, frames, input_channels, cfg.hidden_size)
      if diagnostic = trace
        MetalOps.copy(e, scratch.current, diagnostic.feature_projection, seq)
      end

      e.set_pipeline(pipeline("wan_audio_position_scale"))
      e.set_buffer(weights.position_weight_v, 0).set_buffer(weights.position_weight_g, 1)
      e.set_buffer(scratch.position_scale, 2, ML::Metal::BufferAccess::Write)
      e.set_value(cfg.hidden_size.to_u32, 3)
      e.set_value((cfg.hidden_size // cfg.position_groups).to_u32, 4)
      e.set_value(cfg.position_kernel.to_u32, 5)
      e.dispatch_threadgroups({cfg.position_kernel, 1, 1}, {256, 1, 1})
      if diagnostic = trace
        MetalOps.copy(e, scratch.position_scale, diagnostic.position_scale, cfg.position_kernel)
      end

      if prepared_position = position_plan
        unless mps_chain = chain
          raise ArgumentError.new("Wan positional MPSGraph plan requires an MPSCommandChain")
        end
        mps_chain.pause!
        prepared_position.encode(mps_chain.handle)
        mps_chain.resume!
        e = mps_chain.encoder
      else
        e.set_pipeline(pipeline("wan_audio_position_conv"))
        e.set_buffer(scratch.current, 0).set_buffer(weights.position_weight_v, 1)
        e.set_buffer(scratch.position_scale, 2).set_buffer(weights.position_bias, 3)
        e.set_buffer(scratch.position_pre_gelu, 4, ML::Metal::BufferAccess::Write)
        e.set_buffer(scratch.position, 9, ML::Metal::BufferAccess::Write)
        e.set_value(frames.to_u32, 5).set_value(cfg.hidden_size.to_u32, 6)
        e.set_value((cfg.hidden_size // cfg.position_groups).to_u32, 7)
        e.set_value(cfg.position_kernel.to_u32, 8)
        e.dispatch_1d(seq)
      end
      if diagnostic = trace
        MetalOps.copy(e, scratch.position_pre_gelu, diagnostic.position_conv_raw, seq)
        MetalOps.copy(e, scratch.position, diagnostic.position_conv, seq)
      end
      MetalOps.add(e, scratch.current, scratch.position, scratch.current, seq)
      if diagnostic = trace
        MetalOps.copy(e, scratch.current, diagnostic.position_add, seq)
      end
      bucket(e, cfg, scratch.current, output, 0)

      cfg.layers.times do |layer_index|
        layer = weights.layers[layer_index]
        MetalOps.copy(e, scratch.current, scratch.residual, seq)
        layer_norm_mps(e, scratch.current, layer.attention_norm_weight, layer.attention_norm_bias,
          scratch.current, frames, cfg.hidden_size, cfg.layer_norm_eps)
        if diagnostic = trace
          if layer_index == 0
            MetalOps.copy(e, scratch.current, diagnostic.pre_attention_norm, seq)
          end
        end
        MetalOps.linear(e, scratch.current, layer.query.weight, layer.query.bias, scratch.query,
          frames, cfg.hidden_size, cfg.hidden_size)
        MetalOps.linear(e, scratch.current, layer.key.weight, layer.key.bias, scratch.key,
          frames, cfg.hidden_size, cfg.hidden_size)
        MetalOps.linear(e, scratch.current, layer.value.weight, layer.value.bias, scratch.value,
          frames, cfg.hidden_size, cfg.hidden_size)
        if diagnostic = trace
          if layer_index == 0
            MetalOps.copy(e, scratch.query, diagnostic.query, seq)
            MetalOps.copy(e, scratch.key, diagnostic.key, seq)
            MetalOps.copy(e, scratch.value, diagnostic.value, seq)
          end
        end
        MetalOps.attention(e, scratch.query, scratch.key, scratch.value, scratch.attention,
          frames, frames, cfg.heads, cfg.hidden_size // cfg.heads)
        if diagnostic = trace
          if layer_index == 0
            MetalOps.copy(e, scratch.attention, diagnostic.attention, seq)
          end
        end
        MetalOps.linear(e, scratch.attention, layer.attention_output.weight,
          layer.attention_output.bias, scratch.projection, frames, cfg.hidden_size, cfg.hidden_size)
        if diagnostic = trace
          if layer_index == 0
            MetalOps.copy(e, scratch.projection, diagnostic.attention_projection, seq)
          end
        end
        MetalOps.add(e, scratch.residual, scratch.projection, scratch.current, seq)
        if diagnostic = trace
          if layer_index == 0
            MetalOps.copy(e, scratch.current, diagnostic.attention_residual, seq)
          end
        end

        # Keep the post-attention hidden state intact for the FFN residual.
        # StableLayerNorm normalizes a temporary input to the feed-forward
        # network; overwriting `current` here would make the final residual add
        # use the normalized tensor instead of the pre-norm hidden state.
        layer_norm_mps(e, scratch.current, layer.ffn_norm_weight, layer.ffn_norm_bias,
          scratch.position, frames, cfg.hidden_size, cfg.layer_norm_eps)
        if diagnostic = trace
          if layer_index == 0
            MetalOps.copy(e, scratch.position, diagnostic.ffn_norm, seq)
          end
        end
        MetalOps.linear(e, scratch.position, layer.ffn_in.weight, layer.ffn_in.bias,
          scratch.ffn, frames, cfg.hidden_size, cfg.intermediate_size)
        if diagnostic = trace
          if layer_index == 0
            MetalOps.copy(e, scratch.ffn, diagnostic.ffn_pre_gelu, cfg.ffn_values)
          end
        end
        ffn_hidden = scratch.ffn
        if ffn_gelu = gelu_plans
          unless mps_chain = chain
            raise ArgumentError.new("Wan audio MPSGraph GELU plans require an MPSCommandChain")
          end
          mps_chain.pause!
          ffn_gelu.encode_ffn(mps_chain.handle)
          mps_chain.resume!
          e = mps_chain.encoder
          ffn_hidden = scratch.ffn_gelu
        else
          MetalOps.gelu(e, scratch.ffn, scratch.ffn, cfg.ffn_values, approximate: false)
        end
        if diagnostic = trace
          if layer_index == 0
            MetalOps.copy(e, ffn_hidden, diagnostic.ffn_post_gelu, cfg.ffn_values)
          end
        end
        MetalOps.linear(e, ffn_hidden, layer.ffn_out.weight, layer.ffn_out.bias,
          scratch.projection, frames, cfg.intermediate_size, cfg.hidden_size)
        if diagnostic = trace
          if layer_index == 0
            MetalOps.copy(e, scratch.projection, diagnostic.ffn_output, seq)
          end
        end
        MetalOps.add(e, scratch.current, scratch.projection, scratch.current, seq)
        if diagnostic = trace
          if layer_index == 0
            MetalOps.copy(e, scratch.current, diagnostic.layer_output, seq)
          end
        end

        if layer_index < cfg.layers - 1
          bucket(e, cfg, scratch.current, output, layer_index + 1)
        else
          layer_norm_mps(e, scratch.current, weights.final_norm_weight, weights.final_norm_bias,
            scratch.current, frames, cfg.hidden_size, cfg.layer_norm_eps)
          bucket(e, cfg, scratch.current, output, cfg.layers)
        end
      end
    end

    private def self.bucket(e : ML::Metal::ComputeEncoder, cfg : WanAudioEncoderConfig,
                            hidden : ML::MetalBuffer, output : ML::MetalBuffer,
                            layer_index : Int32) : Nil
      e.set_pipeline(pipeline("wan_audio_bucket"))
      e.set_buffer(hidden, 0).set_buffer(output, 1, ML::Metal::BufferAccess::Write)
      e.set_value(cfg.feature_frames.to_u32, 2).set_value(cfg.interpolation_frames.to_u32, 3)
      e.set_value(cfg.hidden_size.to_u32, 4).set_value(cfg.bucket_frames.to_u32, 5)
      e.set_value(cfg.video_fps.to_u32, 6).set_value(cfg.target_fps.to_u32, 7)
      e.set_value(layer_index.to_u32, 8)
      e.dispatch_1d(cfg.hidden_size * cfg.bucket_frames)
    end

    private def self.layer_norm_mps(e : ML::Metal::ComputeEncoder, input : ML::MetalBuffer,
                                    weight : ML::MetalBuffer, bias : ML::MetalBuffer,
                                    output : ML::MetalBuffer, rows : Int32, dim : Int32,
                                    epsilon : Float32) : Nil
      unless rows > 0 && dim > 0 && epsilon.finite? && epsilon > 0
        raise ArgumentError.new("Invalid Wan Wav2Vec2 LayerNorm dimensions")
      end
      count = rows.to_i64 * dim
      MetalOps.check(input, count)
      MetalOps.check(weight, dim.to_i64)
      MetalOps.check(bias, dim.to_i64)
      MetalOps.check(output, count)
      MetalOps.inplace_or_distinct(input, output)
      MetalOps.distinct(output, weight, bias)

      e.set_pipeline(pipeline("wan_audio_layer_norm_mps"))
      e.set_buffer(input, 0).set_buffer(output, 1, ML::Metal::BufferAccess::Write)
      e.set_buffer(weight, 2).set_buffer(bias, 3)
      e.set_value(dim.to_u32, 4).set_value(epsilon, 5)
      threads = Math.min((dim.to_i64 + 3) // 4, 1_024_i64).to_i32
      e.dispatch_threadgroups({rows, 1, 1}, {threads, 1, 1})
    end

    private def self.conv1d(e : ML::Metal::ComputeEncoder, input : ML::MetalBuffer,
                            weight : ML::MetalBuffer, bias : ML::MetalBuffer,
                            output : ML::MetalBuffer, input_len : Int32, output_len : Int32,
                            input_channels : Int32, output_channels : Int32,
                            kernel_size : Int32, stride : Int32) : Nil
      count = output_len.to_i64 * output_channels
      MetalOps.check(input, input_len.to_i64 * input_channels)
      MetalOps.check(weight, output_channels.to_i64 * input_channels * kernel_size)
      MetalOps.check(bias, output_channels.to_i64)
      MetalOps.check(output, count)
      MetalOps.distinct(output, input, weight, bias)
      e.set_pipeline(pipeline("wan_audio_conv1d"))
      e.set_buffer(input, 0).set_buffer(weight, 1).set_buffer(bias, 2)
      e.set_buffer(output, 3, ML::Metal::BufferAccess::Write)
      e.set_value(input_len.to_u32, 4).set_value(output_len.to_u32, 5)
      e.set_value(input_channels.to_u32, 6).set_value(output_channels.to_u32, 7)
      e.set_value(kernel_size.to_u32, 8).set_value(stride.to_u32, 9)
      e.dispatch_1d(count.to_i32)
    end

    private def self.gelu_mps(e : ML::Metal::ComputeEncoder, values : ML::MetalBuffer,
                              count : Int32) : Nil
      MetalOps.dims(count)
      MetalOps.check(values, count.to_i64)
      e.set_pipeline(pipeline("wan_audio_gelu_mps"))
      e.set_buffer(values, 0).set_buffer(values, 1, ML::Metal::BufferAccess::Write)
      e.set_value(count.to_u32, 2).dispatch_1d(count)
    end

    private def self.add_binding(bindings : Array(WeightBinding), target : String,
                                 source : String, shape : Array(Int64),
                                 dtype : ML::GGUF::SafeTensorDType?) : Nil
      bindings << WeightBinding.new(target, SOURCE_PREFIX + source, shape, dtype)
    end

    private def self.add_linear_bindings(bindings : Array(WeightBinding), prefix : String,
                                         source : String, out_dim : Int32, in_dim : Int32,
                                         dtype : ML::GGUF::SafeTensorDType?) : Nil
      add_binding(bindings, prefix + ".weight", source + ".weight", [out_dim.to_i64, in_dim.to_i64], dtype)
      add_binding(bindings, prefix + ".bias", source + ".bias", [out_dim.to_i64], dtype)
    end

    private def self.resident_f32(arena : Arena, name : String, count : Int64) : ML::MetalBuffer
      buffer = arena[name]
      raise ArgumentError.new("Wan Wav2Vec2 requires F32 resident weights") unless arena.dtype(name).f32?
      MetalOps.check(buffer, count)
      buffer
    end

    private def self.linear_weights(arena : Arena, prefix : String,
                                    weight_count : Int64, bias_count : Int64) : WanAudioLinearWeights
      WanAudioLinearWeights.new(resident_f32(arena, prefix + ".weight", weight_count),
        resident_f32(arena, prefix + ".bias", bias_count))
    end
  end
end
