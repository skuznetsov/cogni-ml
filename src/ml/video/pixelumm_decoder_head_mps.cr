# Opt-in final head for the PixelUMM v1 decoder. The caller owns the incoming
# block35 activation, all four checkpoint weights, and the MPS command chain.
require "../core/buffer"
require "./metal_mps_command_chain"
require "./metal_ops"
require "./pixelumm_decoder_block_mps"

{% if flag?(:darwin) && !flag?(:cpu_only) %}
  @[Link(ldflags: "-framework Foundation -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph")]
{% end %}

{% unless flag?(:cpu_only) %}
  lib PixelUMMDecoderHeadMPSFFI
    fun plan_create = gs_pixelumm_decoder_head_plan_create(
      device : Void*, temporal : Int32, height : Int32, width : Int32,
      tile : Int32, dim : Int32, input : Void*, weights : Void**,
      owned : Void**, error : UInt8*, error_capacity : LibC::SizeT,
    ) : Void*
    fun encode = gs_pixelumm_decoder_head_encode(
      plan : Void*, chain : Void*, error : UInt8*, error_capacity : LibC::SizeT,
    ) : Int32
    fun plan_release = gs_pixelumm_decoder_head_plan_release(plan : Void*) : Void
  end
{% end %}

module ML::Video::PixelUMMDecoderHeadMPS
  MODEL_DIM               =    4_096
  OUTPUT_DIM              =    3_072
  OUTPUT_CHANNELS         =        3
  OUTPUT_FRAMES_PER_TOKEN =        4
  PATCH_SIZE              =       16
  TILE_TOKENS             =        8
  RMS_NORM_EPS            = 1e-6_f32

  WEIGHT_NAMES = {
    "final_norm_weight", "head_norm_weight",
    "head_linear_weight", "head_linear_bias",
  }

  # Names and shapes match cogni_ml.pixelumm_head_reference/v1. Output is
  # always resident; other named stages are exposed only when requested.
  CAPTURE_NAMES = {
    "final_norm_tile_major", "restored_thw", "head_norm_thw",
    "linear_thw", "output_thw",
  }

  # The order is part of the native bridge binding contract.
  OWNED_BUFFER_NAMES = {
    "final_norm_pre_gain", "final_norm_tile_major", "restored_thw",
    "head_norm_pre_gain", "head_norm_thw", "linear_thw", "output_thw",
    "final_norm_statistics", "head_norm_statistics", "rms_power_exponent",
    "restore_indices",
  }

  # The repeated byte pair is BF16 0x7fc0 on little-endian Apple Silicon and
  # the repeated word is also an F32 NaN for the RMS statistics planes.
  module PoisonPattern
    SLAB_BYTES = 1 << 20

    def self.nan_slab : Bytes
      slab = Bytes.new(SLAB_BYTES, 0xc0_u8)
      index = 1
      while index < slab.size
        slab[index] = 0x7f_u8
        index += 2
      end
      slab
    end

    # Copy a bounded slab across a destination without restarting at offset 0.
    # This is CPU-testable independently of MetalBuffer construction.
    def self.fill!(destination : Slice(UInt8), slab : Slice(UInt8)) : Nil
      raise ArgumentError.new("PixelUMM poison slab must not be empty") if slab.empty?
      offset = 0
      while offset < destination.size
        count = Math.min(slab.size, destination.size - offset)
        (destination.to_unsafe + offset).copy_from(slab.to_unsafe, count)
        offset += count
      end
    end
  end

  struct Config
    getter temporal : Int32
    getter height : Int32
    getter width : Int32
    getter tile_tokens : Int32
    getter model_dim : Int32 = MODEL_DIM
    getter output_dim : Int32 = OUTPUT_DIM

    def initialize(@temporal : Int32, @height : Int32, @width : Int32,
                   @tile_tokens : Int32 = TILE_TOKENS)
      validate!
    end

    def validate! : Nil
      unless @temporal > 0 && @height > 0 && @width > 0
        raise ArgumentError.new("PixelUMM head temporal, height, and width must be positive")
      end
      unless @tile_tokens == TILE_TOKENS
        raise ArgumentError.new("PixelUMM head requires the decoder's 8-token tile layout")
      end
      output_t = @temporal.to_i64 * OUTPUT_FRAMES_PER_TOKEN
      output_h = @height.to_i64 * PATCH_SIZE
      output_w = @width.to_i64 * PATCH_SIZE
      unless output_t <= Int32::MAX && output_h <= Int32::MAX && output_w <= Int32::MAX
        raise ArgumentError.new("PixelUMM head output geometry exceeds MPS limits")
      end
      checked_token_count
      # Reuse the decoder's exact tile-order definition and checked token
      # bound, so block35 and head cannot disagree about their row identity.
      block_config
      unless output_elements <= Int32::MAX
        raise ArgumentError.new("PixelUMM head output exceeds the native Metal dispatch limit")
      end
    end

    def tokens : Int32
      (checked_token_count).to_i32
    end

    def output_shape : Array(Int32)
      [1_i32, OUTPUT_CHANNELS, (@temporal * OUTPUT_FRAMES_PER_TOKEN),
       (@height * PATCH_SIZE), (@width * PATCH_SIZE)]
    end

    def output_elements : Int64
      checked_token_count * OUTPUT_DIM
    end

    def output_bytes : Int64
      output_elements * 2
    end

    def layout : ML::Video::PixelUMMDecoderBlockMPS::AttentionLayout
      block_config.layout
    end

    # For each original THW row this returns its source tile-major row.
    def restoration_indices : Array(Int32)
      layout.inverse_permutation
    end

    private def checked_token_count : Int64
      t = @temporal.to_i64
      h = @height.to_i64
      w = @width.to_i64
      raise ArgumentError.new("PixelUMM head token count overflows") if t > Int64::MAX // h
      th = t * h
      raise ArgumentError.new("PixelUMM head token count overflows") if th > Int64::MAX // w
      th * w
    end

    private def block_config : ML::Video::PixelUMMDecoderBlockMPS::Config
      ML::Video::PixelUMMDecoderBlockMPS::Config.new(
        temporal: @temporal, height: @height, width: @width,
        tile_tokens: @tile_tokens)
    end
  end

  class Plan
    WEIGHT_NAMES       = PixelUMMDecoderHeadMPS::WEIGHT_NAMES
    CAPTURE_NAMES      = PixelUMMDecoderHeadMPS::CAPTURE_NAMES
    OWNED_BUFFER_NAMES = PixelUMMDecoderHeadMPS::OWNED_BUFFER_NAMES

    getter config : Config
    getter input : ML::MetalBuffer
    getter named_bytes : Int64 = 0_i64
    getter named_bytes_by_name = {} of String => Int64
    getter poison_count : Int32 = 0_i32
    getter last_poisoned_bytes : Int64 = 0_i64

    @weights : Hash(String, ML::MetalBuffer)
    @buffers = {} of String => ML::MetalBuffer
    @selected_captures : Array(String)
    @native = Pointer(Void).null
    @chain : ML::Metal::MPSCommandChain? = nil
    @released = false
    @error_text = Bytes.new(1024, 0_u8)

    @@retained_mutex = Mutex.new
    @@retained = [] of Plan

    def initialize(@config : Config, *, input : ML::MetalBuffer,
                   weights : Hash(String, ML::MetalBuffer),
                   max_named_bytes : Int64,
                   captures : Array(String) = [] of String)
      @input = input
      @weights = weights
      @selected_captures = captures.dup
      metadata = weights.map { |name, buffer| {name, buffer.size} }.to_h
      self.class.validate_admission!(@config,
        input_bytes: @input.size, weights_bytes: metadata,
        max_named_bytes: max_named_bytes, captures: @selected_captures)

      {% if flag?(:cpu_only) %}
        raise "Metal disabled (cpu_only)"
      {% else %}
        begin
          validate_bindings!
          allocate_owned_buffers!
          setup_native!
          @@retained_mutex.synchronize { @@retained << self }
        rescue ex
          release_native_plan
          release_owned_buffers
          raise ex
        end
      {% end %}
    end

    # This head owns a fixed full-J working set. Selected captures name the
    # corresponding resident stage buffers and never create copy buffers.
    def self.owned_bytes(config : Config, captures : Array(String) = [] of String) : Int64
      config.validate!
      validate_captures!(captures)
      j = config.tokens.to_i64
      d = config.model_dim.to_i64
      out = config.output_dim.to_i64
      # Five hidden BF16 boundaries: final RMS pre-gain/output, THW restore,
      # head RMS pre-gain/output. Two BF16 [J,3072] boundaries hold linear and
      # unpatchified output. Statistics, inverse row map, and exponent are
      # explicit named buffers as well.
      5_i64 * j * d * 2 + 2_i64 * j * out * 2 +
        2_i64 * j * 4 + j * 4 + 4
    end

    # Pure CPU metadata admission. The input and all weights are borrowed;
    # `max_named_bytes` covers only this plan's resident output and scratch.
    def self.validate_admission!(config : Config, *, input_bytes : Int64,
                                 weights_bytes : Hash(String, Int64),
                                 max_named_bytes : Int64,
                                 captures : Array(String) = [] of String) : Nil
      config.validate!
      expected_weights = {
        "final_norm_weight"  => config.model_dim.to_i64 * 2,
        "head_norm_weight"   => config.model_dim.to_i64 * 2,
        "head_linear_weight" => config.output_dim.to_i64 * config.model_dim * 2,
        "head_linear_bias"   => config.output_dim.to_i64 * 2,
      }
      unless weights_bytes.keys.sort == expected_weights.keys.sort
        raise ArgumentError.new("exactly four PixelUMM head weight metadata entries required")
      end
      expected_weights.each do |name, bytes|
        unless weights_bytes[name] == bytes
          raise ArgumentError.new("PixelUMM head #{name} must be exactly #{bytes} bytes")
        end
      end
      expected_input = config.tokens.to_i64 * config.model_dim * 2
      unless input_bytes == expected_input
        raise ArgumentError.new("PixelUMM head input extent must be exactly #{expected_input} bytes")
      end
      validate_captures!(captures)
      required = owned_bytes(config, captures)
      unless max_named_bytes > 0 && required <= max_named_bytes
        raise ArgumentError.new("PixelUMM head resident buffers exceed the admitted budget")
      end
    end

    def encode(chain : ML::Metal::MPSCommandChain) : Nil
      check_live!
      chain.encoder
      if previous = @chain
        unless previous.completed?
          raise ArgumentError.new("PixelUMM head plan cannot be reused while its command chain is in flight")
        end
      end
      @chain = chain
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        chain.pause!
        status = -1
        begin
          status = PixelUMMDecoderHeadMPSFFI.encode(
            @native, chain.handle, @error_text.to_unsafe, @error_text.size)
        ensure
          chain.resume!
        end
        raise ArgumentError.new("PixelUMM head encode failed (#{status}): #{error_message}") unless status == 0
      {% else %}
        raise ArgumentError.new("PixelUMM head requires macOS Metal Performance Shaders")
      {% end %}
    end

    # The buffer handle is available for same-chain GPU chaining before encode.
    # CPU reads remain the caller's responsibility and are valid only after
    # the command chain's terminal wait.
    def output : ML::MetalBuffer
      check_live!
      @buffers["output_thw"]
    end

    def stage(name : String) : ML::MetalBuffer
      check_live!
      if name == "output_thw"
        return @buffers[name]
      end
      unless @selected_captures.includes?(name)
        raise ArgumentError.new("PixelUMM head stage was not captured: #{name}")
      end
      @buffers[name]? || raise ArgumentError.new("unknown PixelUMM head stage: #{name}")
    end

    def stages : Hash(String, ML::MetalBuffer)
      check_live!
      result = {} of String => ML::MetalBuffer
      @selected_captures.each do |name|
        result[name] = @buffers[name] unless name == "output_thw"
      end
      result["output_thw"] = @buffers["output_thw"]
      result
    end

    # Poison every writable activation/statistics plane after a completed
    # request. The inverse permutation and exponent are immutable setup data.
    # This makes accidental scratch dependence visible on the next encode.
    def poison_transients! : Int64
      check_live!
      if chain = @chain
        raise ArgumentError.new("complete PixelUMM head command chain before poisoning scratch") unless chain.completed?
      end
      slab = PixelUMMDecoderHeadMPS::PoisonPattern.nan_slab
      names = {
        "final_norm_pre_gain", "final_norm_tile_major", "restored_thw",
        "head_norm_pre_gain", "head_norm_thw", "linear_thw", "output_thw",
        "final_norm_statistics", "head_norm_statistics",
      }
      poisoned = 0_i64
      names.each do |name|
        buffer = @buffers[name]
        destination = Slice.new(buffer.contents.as(Pointer(UInt8)), buffer.size.to_i)
        PixelUMMDecoderHeadMPS::PoisonPattern.fill!(destination, slab)
        poisoned += buffer.size
      end
      @poison_count += 1
      @last_poisoned_bytes = poisoned
      poisoned
    end

    def transient_bytes : Int64
      config.tokens.to_i64 * config.model_dim * 2 * 5 +
        config.tokens.to_i64 * config.output_dim * 2 * 2 +
        config.tokens.to_i64 * 4 * 2
    end

    def release : Nil
      return if @released
      if chain = @chain
        raise ArgumentError.new("complete PixelUMM head command chain before release") unless chain.completed?
      end
      release_native_plan
      release_owned_buffers
      @released = true
      @@retained_mutex.synchronize { @@retained.reject! { |plan| plan.same?(self) } }
    end

    private def self.validate_captures!(captures : Array(String)) : Nil
      if captures.uniq.size != captures.size
        raise ArgumentError.new("PixelUMM head capture names must be unique")
      end
      unknown = captures.reject { |name| CAPTURE_NAMES.includes?(name) }
      raise ArgumentError.new("unknown PixelUMM head capture stage(s): #{unknown.join(",")}") unless unknown.empty?
    end

    private def validate_bindings! : Nil
      raise ArgumentError.new("PixelUMM head input must be an exact-size shared buffer") unless @input.valid? && @input.storage_mode.shared? &&
                                                                                                @input.size == @config.tokens.to_i64 * @config.model_dim * 2
      expected = {
        "final_norm_weight"  => @config.model_dim.to_i64 * 2,
        "head_norm_weight"   => @config.model_dim.to_i64 * 2,
        "head_linear_weight" => @config.output_dim.to_i64 * @config.model_dim * 2,
        "head_linear_bias"   => @config.output_dim.to_i64 * 2,
      }
      unless @weights.keys.sort == WEIGHT_NAMES.to_a.sort
        raise ArgumentError.new("exactly four PixelUMM head weight buffers required")
      end
      expected.each do |name, bytes|
        buffer = @weights[name]
        unless buffer.valid? && buffer.storage_mode.shared? && buffer.size == bytes
          raise ArgumentError.new("PixelUMM head #{name} must be an exact-size shared BF16 buffer")
        end
      end
      borrowed = [@input] + WEIGHT_NAMES.map { |name| @weights[name] }.to_a
      validate_disjoint!(borrowed, "borrowed PixelUMM head buffers must not alias")
    end

    private def allocate_owned_buffers! : Nil
      j = @config.tokens.to_i64
      d = @config.model_dim.to_i64
      out = @config.output_dim.to_i64
      sizes = {
        "final_norm_pre_gain"   => j * d * 2,
        "final_norm_tile_major" => j * d * 2,
        "restored_thw"          => j * d * 2,
        "head_norm_pre_gain"    => j * d * 2,
        "head_norm_thw"         => j * d * 2,
        "linear_thw"            => j * out * 2,
        "output_thw"            => j * out * 2,
        "final_norm_statistics" => j * 4,
        "head_norm_statistics"  => j * 4,
        "rms_power_exponent"    => 4_i64,
        "restore_indices"       => j * 4,
      }
      sizes.each do |name, bytes|
        @buffers[name] = ML::MetalBuffer.new(bytes, ML::StorageMode::Shared)
        @named_bytes_by_name[name] = bytes
        @named_bytes += bytes
      end
      exponent = Slice(Float32).new(1, 2.0_f32)
      @buffers["rms_power_exponent"].write_bytes(
        exponent.to_unsafe.as(Pointer(UInt8)), sizeof(Float32))
      indices = @config.restoration_indices
      @buffers["restore_indices"].write_bytes(
        indices.to_unsafe.as(Pointer(UInt8)), indices.size * sizeof(Int32))
      expected = self.class.owned_bytes(@config, @selected_captures)
      raise ArgumentError.new("PixelUMM head named-byte inventory mismatch") unless @named_bytes == expected
      owned = OWNED_BUFFER_NAMES.map { |name| @buffers[name] }.to_a
      borrowed = [@input] + WEIGHT_NAMES.map { |name| @weights[name] }.to_a
      validate_disjoint!(borrowed + owned,
        "PixelUMM head writable buffers must be disjoint from borrowed buffers and each other")
    end

    private def setup_native! : Nil
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        weight_handles = WEIGHT_NAMES.map { |name| @weights[name].handle }.to_a
        owned_handles = OWNED_BUFFER_NAMES.map { |name| @buffers[name].handle }.to_a
        @native = PixelUMMDecoderHeadMPSFFI.plan_create(
          ML::Metal::Device.instance.device_handle,
          @config.temporal, @config.height, @config.width,
          @config.tile_tokens, @config.model_dim, @input.handle,
          weight_handles.to_unsafe, owned_handles.to_unsafe,
          @error_text.to_unsafe, @error_text.size)
        raise ArgumentError.new("PixelUMM head plan preparation failed: #{error_message}") if @native.null?
      {% else %}
        raise ArgumentError.new("PixelUMM head requires macOS Metal Performance Shaders")
      {% end %}
    end

    private def validate_disjoint!(buffers : Array(ML::MetalBuffer), message : String) : Nil
      buffers.each_with_index do |left, left_index|
        buffers.each_with_index do |right, right_index|
          next if right_index <= left_index
          raise ArgumentError.new(message) if ML::Video::MetalOps.overlap?(left, right)
        end
      end
    end

    private def release_native_plan : Nil
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        PixelUMMDecoderHeadMPSFFI.plan_release(@native) unless @native.null?
      {% end %}
      @native = Pointer(Void).null
    end

    private def release_owned_buffers : Nil
      @buffers.each_value(&.release)
      @buffers.clear
      @named_bytes_by_name.clear
      @named_bytes = 0_i64
    end

    private def error_message : String
      String.new(@error_text.to_unsafe)
    end

    private def check_live! : Nil
      raise ArgumentError.new("PixelUMM head plan has been released") if @released || @native.null?
    end
  end
end
