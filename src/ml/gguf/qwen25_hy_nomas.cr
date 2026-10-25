require "../core/buffer"
require "../video/metal_mps_command_chain"

{% unless flag?(:cpu_only) %}
  lib Qwen25MPSFFI
    fun create = gs_qwen25_mps_block_plan_create(device : Void*, batches : Int32,
                                                 tokens : Int32, dim : Int32, ffn : Int32, heads : Int32, kv_heads : Int32,
                                                 head_dim : Int32, rms_norm_eps : Float32, input : Void*, mask : Void*,
                                                 positions : Void*, inv_freq : Void*, weights : Void**, stages : Void**, scratch : Void**,
                                                 error : UInt8*, error_capacity : LibC::SizeT) : Void*
    fun encode = gs_qwen25_mps_block_encode(plan : Void*, chain : Void*,
                                            error : UInt8*, error_capacity : LibC::SizeT) : Int32
    fun destroy = gs_qwen25_mps_block_plan_destroy(plan : Void*) : Void
  end
{% end %}

module ML::GGUF
  # Text-only Qwen2.5-VL geometry. This is deliberately separate from the
  # Qwen3.5 hybrid/GDN configuration and Qwen3-VL's Q/K-normalized decoder.
  struct Qwen25VLConfig
    getter dim : Int32
    getter ffn_dim : Int32
    getter heads : Int32
    getter kv_heads : Int32
    getter head_dim : Int32
    getter layers : Int32
    getter rms_norm_eps : Float32
    getter rope_theta : Float32
    getter max_positions : Int32

    def initialize(@dim = 64, @ffn_dim = 256, @heads = 2, @kv_heads = 1,
                   @head_dim = 32, @layers = 2, @rms_norm_eps = 1e-6_f32,
                   @rope_theta = 1_000_000_f32, @max_positions = 32768)
      raise ArgumentError.new("positive Qwen2.5-VL dimensions required") unless {
                                                                                  @dim, @ffn_dim, @heads, @kv_heads, @head_dim, @layers, @max_positions,
                                                                                }.all? { |n| n > 0 }
      raise ArgumentError.new("dim must equal heads * head_dim") unless @dim.to_i64 == @heads.to_i64 * @head_dim
      raise ArgumentError.new("query heads must be divisible by KV heads") unless @heads % @kv_heads == 0
      raise ArgumentError.new("split-half rotary requires even head_dim") unless @head_dim.even?
      raise ArgumentError.new("finite positive RMS epsilon required") unless @rms_norm_eps.finite? && @rms_norm_eps > 0
      raise ArgumentError.new("finite rotary base > 1 required") unless @rope_theta.finite? && @rope_theta > 1
    end

    def self.hy_nomas_7b : self
      new(dim: 3584, ffn_dim: 18944, heads: 28, kv_heads: 4, head_dim: 128,
        layers: 28, max_positions: 128000)
    end

    # HF states are [embedding, block0, ..., final-normalized blockN-1].
    # Hy-Nomas asks for states[-(skip+1)], apply_final_norm=false. Skip=2
    # means 26 blocks. Skip=0 additionally requires the stack's final norm;
    # this count alone does not qualify that unsupported final-norm path.
    def selected_block_count(skip : Int32) : Int32
      raise ArgumentError.new("hidden-state skip must be in 0...layers") unless 0 <= skip < @layers
      @layers - skip
    end

    def kv_dim : Int32
      (@kv_heads.to_i64 * @head_dim).to_i32
    end
  end

  # Prepared, explicitly released native graph. All bound tensors use BF16
  # except positions and inverse frequencies (F32). Only serial host calls
  # are supported. Borrowed bindings must not be overwritten or released in
  # a way that races scheduled reads before the terminal wait. An ordered
  # earlier GPU producer on the same chain may populate a block's input.
  class Qwen25VLBlockPlan
    WEIGHT_NAMES = {"input_norm", "q_w", "q_bias", "k_w", "k_bias", "v_w", "v_bias",
                    "o_w", "post_norm", "gate_w", "up_w", "down_w"}
    STAGE_NAMES = {"input_norm", "q_proj", "k_proj", "v_proj", "q_rope", "k_rope",
                   "attended", "o_proj", "residual", "post_norm", "gate_proj",
                   "up_proj", "silu_product", "down_proj", "hidden_out", "rotary_cos", "rotary_sin"}
    SCRATCH_NAMES = {"input_normalized", "q_cos_product", "q_sin_product", "k_cos_product",
                     "k_sin_product", "post_normalized", "silu"}

    getter config : Qwen25VLConfig
    getter tokens : Int32
    getter input : MetalBuffer
    getter layer_index : Int32
    getter workspace_bytes : Int64 = 0_i64
    @mask : MetalBuffer
    @positions : MetalBuffer
    @inv_freq : MetalBuffer
    @weights : Array(MetalBuffer)
    @stages = {} of String => MetalBuffer
    @scratch = {} of String => MetalBuffer
    @native = Pointer(Void).null
    @chain : Metal::MPSCommandChain? = nil
    @released = false
    @error_text = Bytes.new(1024, 0_u8)
    # Finalizers must not free a graph still referenced by an asynchronous
    # command chain. Explicit release removes this root after terminal drain.
    @@retained = {} of UInt64 => Qwen25VLBlockPlan

    def self.retained_count : Int32
      @@retained.size
    end

    def self.weight_shapes(c : Qwen25VLConfig) : Hash(String, Array(Int32))
      {"input_norm" => [c.dim], "q_w" => [c.dim, c.dim], "q_bias" => [c.dim],
       "k_w" => [c.kv_dim, c.dim], "k_bias" => [c.kv_dim],
       "v_w" => [c.kv_dim, c.dim], "v_bias" => [c.kv_dim],
       "o_w" => [c.dim, c.dim], "post_norm" => [c.dim],
       "gate_w" => [c.ffn_dim, c.dim], "up_w" => [c.ffn_dim, c.dim],
       "down_w" => [c.dim, c.ffn_dim]}
    end

    def self.stage_shapes(c : Qwen25VLConfig, tokens : Int32) : Hash(String, Array(Int32))
      d = [1, tokens, c.dim]
      kv = [1, tokens, c.kv_dim]
      f = [1, tokens, c.ffn_dim]
      {"input_norm" => d, "q_proj" => d, "k_proj" => kv, "v_proj" => kv,
       "q_rope" => [1, c.heads, tokens, c.head_dim],
       "k_rope" => [1, c.kv_heads, tokens, c.head_dim],
       "attended" => d, "o_proj" => d, "residual" => d, "post_norm" => d,
       "gate_proj" => f, "up_proj" => f, "silu_product" => f,
       "down_proj" => d, "hidden_out" => d,
       "rotary_cos" => [1, 1, tokens, c.head_dim],
       "rotary_sin" => [1, 1, tokens, c.head_dim]}
    end

    def self.elements(shape : Array(Int32)) : Int64
      shape.reduce(1_i64) { |n, v| n * v }
    end

    def self.scratch_shapes(c : Qwen25VLConfig, tokens : Int32) : Hash(String, Array(Int32))
      d = [1, tokens, c.dim]
      q = [1, c.heads, tokens, c.head_dim]
      k = [1, c.kv_heads, tokens, c.head_dim]
      {"input_normalized" => d, "q_cos_product" => q, "q_sin_product" => q,
       "k_cos_product" => k, "k_sin_product" => k, "post_normalized" => d,
       "silu" => [1, tokens, c.ffn_dim]}
    end

    def initialize(@config : Qwen25VLConfig, @tokens : Int32, @input : MetalBuffer,
                   @mask : MetalBuffer, @positions : MetalBuffer, @inv_freq : MetalBuffer,
                   weights : Hash(String, MetalBuffer), @layer_index = 0,
                   max_workspace_bytes : Int64 = 512_i64 * 1024 * 1024)
      @weights = [] of MetalBuffer
      raise ArgumentError.new("tokens outside configured context") unless 0 < @tokens <= @config.max_positions
      raise ArgumentError.new("layer index outside decoder") unless 0 <= @layer_index < @config.layers
      raise ArgumentError.new("exact Qwen2.5-VL weight bindings required") unless weights.keys.sort == WEIGHT_NAMES.to_a.sort
      shapes = self.class.weight_shapes(@config)
      WEIGHT_NAMES.each do |name|
        buffer = weights[name]
        check_buffer!(buffer, self.class.elements(shapes[name]) * 2, name)
        @weights << buffer
      end
      check_buffer!(@input, @tokens.to_i64 * @config.dim * 2, "input")
      check_buffer!(@mask, @tokens.to_i64 * @tokens * 2, "mask")
      check_buffer!(@positions, @tokens.to_i64 * 4, "positions")
      check_buffer!(@inv_freq, @config.head_dim.to_i64 // 2 * 4, "inverse frequencies")
      stage_shapes = self.class.stage_shapes(@config, @tokens)
      scratch_shapes = self.class.scratch_shapes(@config, @tokens)
      @workspace_bytes = (stage_shapes.values + scratch_shapes.values).sum { |shape| self.class.elements(shape) * 2 }
      raise ArgumentError.new("workspace exceeds admitted byte budget") unless 0 < @workspace_bytes <= max_workspace_bytes
      {% if flag?(:cpu_only) %}
        raise "Metal disabled (cpu_only)"
      {% else %}
        begin
          stage_shapes.each do |name, shape|
            @stages[name] = MetalBuffer.new(self.class.elements(shape) * 2)
          end
          scratch_shapes.each do |name, shape|
            @scratch[name] = MetalBuffer.new(self.class.elements(shape) * 2)
          end
          weight_handles = @weights.map(&.handle)
          stage_handles = STAGE_NAMES.map { |name| @stages[name].handle }.to_a
          scratch_handles = SCRATCH_NAMES.map { |name| @scratch[name].handle }.to_a
          @native = Qwen25MPSFFI.create(Metal::Device.instance.device_handle,
            1, @tokens, @config.dim, @config.ffn_dim, @config.heads, @config.kv_heads,
            @config.head_dim, @config.rms_norm_eps, @input.handle, @mask.handle,
            @positions.handle, @inv_freq.handle, weight_handles.to_unsafe,
            stage_handles.to_unsafe, scratch_handles.to_unsafe, @error_text.to_unsafe, @error_text.size)
          raise "Qwen2.5-VL graph preparation failed: #{error_message}" if @native.null?
          @@retained[object_id] = self
        rescue ex
          @stages.each_value(&.release)
          @stages.clear
          @scratch.each_value(&.release)
          @scratch.clear
          raise ex
        end
      {% end %}
    end

    def stage(name : String) : MetalBuffer
      check_live!
      @stages[name]
    end

    def output : MetalBuffer
      stage("hidden_out")
    end

    # No neural host computation, GPU-buffer allocation, readback or wait.
    # Prepared executable bindings include every intermediate for qualification.
    def encode(chain : Metal::MPSCommandChain) : Nil
      check_live!
      raise ArgumentError.new("cannot encode into a completed chain") if chain.completed?
      if previous = @chain
        unless previous.same?(chain) || previous.completed?
          raise ArgumentError.new("Qwen2.5-VL plan already belongs to an in-flight chain")
        end
      end
      check_bound_buffers!
      # Retain the chain before entering MPS: a failed encode may have submitted
      # partial work and still requires the caller's normal terminal drain.
      @chain = chain
      chain.pause!
      {% unless flag?(:cpu_only) %}
        status = Qwen25MPSFFI.encode(@native, chain.handle, @error_text.to_unsafe, @error_text.size)
        chain.resume!
        raise "Qwen2.5-VL encode failed (#{status}): #{error_message}" unless status == 0
      {% end %}
    end

    def release : Nil
      return if @released
      if chain = @chain
        raise ArgumentError.new("terminal wait required before Qwen2.5-VL release") unless chain.completed?
      end
      {% unless flag?(:cpu_only) %}
        Qwen25MPSFFI.destroy(@native) unless @native.null?
      {% end %}
      @native = Pointer(Void).null
      @stages.each_value(&.release)
      @stages.clear
      @scratch.each_value(&.release)
      @scratch.clear
      @released = true
      @chain = nil
      @@retained.delete(object_id)
    end

    private def check_buffer!(buffer : MetalBuffer, bytes : Int64, name : String) : Nil
      raise ArgumentError.new("#{name}: valid shared buffer required") unless buffer.valid? && buffer.storage_mode.shared?
      raise ArgumentError.new("#{name}: buffer bytes must equal #{bytes}, got #{buffer.size}") unless buffer.size == bytes
    end

    private def check_bound_buffers! : Nil
      raise ArgumentError.new("released Qwen2.5-VL input binding") unless @input.valid? && @mask.valid? && @positions.valid? && @inv_freq.valid?
      raise ArgumentError.new("released Qwen2.5-VL weight binding") unless @weights.all?(&.valid?)
      @stages.each_value do |buffer|
        raise ArgumentError.new("released Qwen2.5-VL stage binding") unless buffer.valid?
      end
      @scratch.each_value do |buffer|
        raise ArgumentError.new("released Qwen2.5-VL scratch binding") unless buffer.valid?
      end
    end

    private def check_live! : Nil
      raise ArgumentError.new("Qwen2.5-VL plan released") if @released || @native.null?
    end

    private def error_message : String
      String.new(@error_text.to_unsafe)
    end
  end
end
