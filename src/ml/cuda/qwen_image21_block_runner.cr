require "./driver"
require "./qwen_image21_projection"
require "../gguf/qwen_image21_block"

module ML::CUDA
  # One Qwen-Image 2.1 single-stream DiT block. Hidden states, projections,
  # attention scratch, and every named checkpoint stay device-resident between
  # enqueue and an explicit diagnostic read_checkpoint call.
  class QwenImage21BlockRunner
    PTX = {{ read_file("src/ml/cuda/kernels/qwen_image21_block.ptx") }}

    CHECKPOINT_NAMES = [
      "norm1", "q_proj", "k_proj", "v", "q_norm", "k_norm", "q_rope",
      "k_rope", "attended", "attn_out", "state", "norm2", "gate_up",
      "activated", "mlp_out", "output",
    ] of String

    alias KernelParam = UInt64 | UInt32 | Float32

    @closed : Bool
    @input_ready : Bool
    @hidden_uploaded : Bool
    @tokens : Int32
    @config : ML::GGUF::QwenImage21BlockConfig
    @buffers : Array(DeviceBuffer)
    @parameter_keepalive : Array(Void*)
    @kernel_params : Hash(String, Pointer(Void*))
    @checkpoints : Hash(String, DeviceBuffer)
    @projections : Array(QwenImage21Projection)
    @functions : Hash(String, KernelFunction)
    @module : CUDAModule
    @hidden_input : DeviceBuffer
    @input_device_ptr : DevicePtr
    @modulation_device : DeviceBuffer
    @image_ids_device : DeviceBuffer
    @key_valid_device : DeviceBuffer
    @rope_cos_device : DeviceBuffer
    @rope_sin_device : DeviceBuffer
    @norm_q_device : DeviceBuffer
    @norm_k_device : DeviceBuffer
    @gate1 : DeviceBuffer
    @gate2 : DeviceBuffer

    getter tokens : Int32
    getter config : ML::GGUF::QwenImage21BlockConfig

    def initialize(
      weights : ML::GGUF::QwenImage21BlockWeights,
      config : ML::GGUF::QwenImage21BlockConfig,
      tokens : Int32,
      modulation : Array(Float32),
      positions : Array(StaticArray(Int32, 3)),
      image_ids : Array(Int32),
      key_valid : Array(Bool)? = nil,
    )
      @config = config
      @tokens = tokens
      validate_inputs!(weights, modulation, positions, image_ids, key_valid)
      validate_weights!(weights)
      validate_attention_rows!(image_ids, key_valid)
      raise ArgumentError.new("token count exceeds supported CUDA grid") if @tokens > 65_535

      @closed = false
      @input_ready = false
      @hidden_uploaded = false
      @buffers = [] of DeviceBuffer
      @parameter_keepalive = [] of Void*
      @kernel_params = {} of String => Pointer(Void*)
      @checkpoints = {} of String => DeviceBuffer
      @projections = [] of QwenImage21Projection
      # This first allocation precedes the cleanup scope, so it is itself
      # the earliest owned resource and the first non-null typed input ivar.
      @hidden_input = allocate_f32(@tokens * @config.hidden_dim)
      @input_device_ptr = @hidden_input.ptr
      # Typed placeholders let Crystal keep these ivars non-nullable while the
      # construction scope owns all subsequent allocations until commit.
      @modulation_device = @hidden_input
      @image_ids_device = @hidden_input
      @key_valid_device = @hidden_input
      @rope_cos_device = @hidden_input
      @rope_sin_device = @hidden_input
      @norm_q_device = @hidden_input
      @norm_k_device = @hidden_input
      @gate1 = @hidden_input
      @gate2 = @hidden_input
      @functions = {} of String => KernelFunction
      module_handle : CUDAModule? = nil
      functions = {} of String => KernelFunction
      initialized = false
      begin
        @modulation_device = upload_f32(modulation, "qwen_image21_modulation")
        @image_ids_device = upload_i32(image_ids, "qwen_image21_image_ids")
        valid_bytes = key_valid ? key_valid.not_nil!.map { |valid| valid ? 1_u8 : 0_u8 } : Array(UInt8).new(@tokens, 1_u8)
        @key_valid_device = upload_u8(valid_bytes, "qwen_image21_key_valid")

        cos_values, sin_values = rope_tables(positions)
        @rope_cos_device = upload_f32(cos_values, "qwen_image21_rope_cos")
        @rope_sin_device = upload_f32(sin_values, "qwen_image21_rope_sin")
        @norm_q_device = upload_f32(weights.norm_q, "qwen_image21_norm_q")
        @norm_k_device = upload_f32(weights.norm_k, "qwen_image21_norm_k")

        CHECKPOINT_NAMES.each do |name|
          @checkpoints[name] = allocate_f32(checkpoint_element_count(name))
        end
        @gate1 = allocate_f32(@tokens * @config.hidden_dim)
        @gate2 = allocate_f32(@tokens * @config.hidden_dim)

        @projections << QwenImage21Projection.new(weights.to_q)
        @projections << QwenImage21Projection.new(weights.to_k)
        @projections << QwenImage21Projection.new(weights.to_v)
        @projections << QwenImage21Projection.new(weights.to_out)
        @projections << QwenImage21Projection.new(weights.gate_up)
        @projections << QwenImage21Projection.new(weights.mlp_out)

        mod = CUDAModule.load(PTX, "qwen_image21_block")
        module_handle = mod
        functions = {
          "layernorm"     => mod.function("qi21_layernorm_modulate_gate"),
          "rmsnorm"       => mod.function("qi21_rmsnorm"),
          "rope"          => mod.function("qi21_rope"),
          "attention"     => mod.function("qi21_block_causal_attention"),
          "residual_norm" => mod.function("qi21_residual_norm_modulate_gate"),
          "swiglu"        => mod.function("qi21_swiglu"),
          "residual_add"  => mod.function("qi21_residual_gate_add"),
        }
        setup_kernel_params
        initialized = true
      ensure
        unless initialized
          module_handle.try(&.close)
          @projections.each(&.close)
          @buffers.each(&.close)
        end
      end
      @module = module_handle.not_nil!
      @functions = functions
    end

    # Copies the initial hidden state before dispatch. This is an input upload
    # boundary, not part of enqueue.
    def upload_hidden(values : Array(Float32)) : Nil
      raise "Qwen-Image block runner is closed" if @closed
      expected = @tokens * @config.hidden_dim
      raise ArgumentError.new("hidden size mismatch") unless values.size == expected

      ML::CUDA.copy_htod!(@hidden_input.ptr, values.to_unsafe.as(Void*), bytesize_f32(expected), "qwen_image21_hidden")
      @input_device_ptr = @hidden_input.ptr
      @hidden_uploaded = true
      @input_ready = true
    end

    # Uses another resident [tokens, hidden_dim] input. The caller owns its
    # lifetime and must keep it valid through the terminal completion boundary.
    def use_device_input(ptr : DevicePtr) : Nil
      raise "Qwen-Image block runner is closed" if @closed
      raise ArgumentError.new("device input pointer must be non-zero") if ptr == 0_u64
      if ptr == @hidden_input.ptr && !@hidden_uploaded
        raise ArgumentError.new("runner upload buffer has not been initialized")
      end
      validate_input_aliases!(ptr)

      @input_device_ptr = ptr
      @input_ready = true
    end

    # Enqueues the whole block on the current CUDA stream. It performs no host
    # activation math, allocation, upload, readback, or synchronization.
    def enqueue : Nil
      raise "Qwen-Image block runner is closed" if @closed
      raise ArgumentError.new("hidden input has not been uploaded or selected") unless @input_ready

      set_param_ptr("layernorm", 0, @input_device_ptr)
      set_param_ptr("residual_norm", 0, @input_device_ptr)

      norm1 = checkpoint("norm1")
      q_proj = checkpoint("q_proj")
      k_proj = checkpoint("k_proj")
      v = checkpoint("v")
      q_norm = checkpoint("q_norm")
      k_norm = checkpoint("k_norm")
      q_rope = checkpoint("q_rope")
      k_rope = checkpoint("k_rope")
      attended = checkpoint("attended")
      attn_out = checkpoint("attn_out")
      state = checkpoint("state")
      norm2 = checkpoint("norm2")
      gate_up = checkpoint("gate_up")
      activated = checkpoint("activated")
      mlp_out = checkpoint("mlp_out")
      output = checkpoint("output")

      launch_1d("layernorm", @tokens, 1_u32, "Qwen-Image norm1")
      @projections[0].enqueue(norm1.ptr, q_proj.ptr, @tokens)
      @projections[1].enqueue(norm1.ptr, k_proj.ptr, @tokens)
      @projections[2].enqueue(norm1.ptr, v.ptr, @tokens)
      launch_2d("rmsnorm", @tokens, @config.heads, 1_u32, "Qwen-Image Q RMSNorm", q_proj.ptr,
        q_norm.ptr, @norm_q_device.ptr)
      launch_2d("rmsnorm", @tokens, @config.heads, 1_u32, "Qwen-Image K RMSNorm", k_proj.ptr,
        k_norm.ptr, @norm_k_device.ptr)
      launch_2d("rope", @tokens, @config.heads, 1_u32, "Qwen-Image Q RoPE", q_norm.ptr,
        q_rope.ptr, @rope_cos_device.ptr, @rope_sin_device.ptr)
      launch_2d("rope", @tokens, @config.heads, 1_u32, "Qwen-Image K RoPE", k_norm.ptr,
        k_rope.ptr, @rope_cos_device.ptr, @rope_sin_device.ptr)
      launch_2d("attention", @tokens, @config.heads, 1_u32, "Qwen-Image block attention",
        q_rope.ptr, k_rope.ptr, v.ptr, attended.ptr)
      @projections[3].enqueue(attended.ptr, attn_out.ptr, @tokens)
      set_param_ptr("residual_norm", 1, attn_out.ptr)
      launch_1d("residual_norm", @tokens, 1_u32, "Qwen-Image attention residual and norm2")
      @projections[4].enqueue(norm2.ptr, gate_up.ptr, @tokens)
      launch_swiglu
      @projections[5].enqueue(activated.ptr, mlp_out.ptr, @tokens)
      launch_residual_add
    end

    def output_device_ptr : DevicePtr
      checkpoint("output").ptr
    end

    def checkpoint_names : Array(String)
      CHECKPOINT_NAMES.dup
    end

    def checkpoint_shapes : Hash(String, Tuple(Int32, Int32))
      dim = @config.hidden_dim
      intermediate = @config.intermediate_dim
      shapes = {} of String => Tuple(Int32, Int32)
      CHECKPOINT_NAMES.each do |name|
        width = case name
                when "gate_up"   then 2 * intermediate
                when "activated" then intermediate
                else                  dim
                end
        shapes[name] = {@tokens, width}
      end
      shapes
    end

    # Blocking DtoH diagnostic readback for parity and debugging tools.
    # Callers must complete the selected stream before reading checkpoints.
    # The block enqueue path never waits for device completion.
    def read_checkpoint(name : String) : Array(Float32)
      raise "Qwen-Image block runner is closed" if @closed
      buffer = checkpoint(name)
      values = Array(Float32).new(checkpoint_element_count(name), 0.0_f32)
      ML::CUDA.copy_dtoh!(values.to_unsafe.as(Void*), buffer.ptr, bytesize_f32(values.size), "qwen_image21_#{name}")
      values
    end

    # Batch blocking diagnostic reads after caller-controlled terminal sync.
    # This helper adds no explicit synchronize call and remains outside enqueue.
    def read_checkpoints(names : Array(String) = CHECKPOINT_NAMES) : Hash(String, Array(Float32))
      values = {} of String => Array(Float32)
      names.each { |name| values[name] = read_checkpoint(name) }
      values
    end

    def close : Nil
      return if @closed

      @projections.each(&.close)
      @module.close
      @buffers.each(&.close)
      @closed = true
    end

    private def validate_inputs!(
      weights : ML::GGUF::QwenImage21BlockWeights,
      modulation : Array(Float32),
      positions : Array(StaticArray(Int32, 3)),
      image_ids : Array(Int32),
      key_valid : Array(Bool)?,
    ) : Nil
      raise ArgumentError.new("tokens must be positive") unless @tokens > 0
      raise ArgumentError.new("hidden dimension must be positive") unless @config.hidden_dim > 0
      raise ArgumentError.new("modulation size mismatch") unless modulation.size == @tokens * 4 * @config.hidden_dim
      raise ArgumentError.new("positions size mismatch") unless positions.size == @tokens
      raise ArgumentError.new("image_ids size mismatch") unless image_ids.size == @tokens
      raise ArgumentError.new("key_valid size mismatch") if key_valid && key_valid.not_nil!.size != @tokens
    end

    private def validate_weights!(weights : ML::GGUF::QwenImage21BlockWeights) : Nil
      dim = @config.hidden_dim
      {weights.to_q, weights.to_k, weights.to_v, weights.to_out}.each do |weight|
        raise ArgumentError.new("attention weight shape mismatch") unless weight.in_dim == dim && weight.out_dim == dim
      end
      raise ArgumentError.new("norm_q size mismatch") unless weights.norm_q.size == @config.head_dim
      raise ArgumentError.new("norm_k size mismatch") unless weights.norm_k.size == @config.head_dim
      unless weights.gate_up.in_dim == dim && weights.gate_up.out_dim == 2 * @config.intermediate_dim
        raise ArgumentError.new("gate_up weight shape mismatch")
      end
      unless weights.mlp_out.in_dim == @config.intermediate_dim && weights.mlp_out.out_dim == dim
        raise ArgumentError.new("MLP output weight shape mismatch")
      end
    end

    private def validate_attention_rows!(image_ids : Array(Int32), key_valid : Array(Bool)?) : Nil
      @tokens.times do |query|
        allowed = false
        @tokens.times do |key|
          next if key_valid && !key_valid.not_nil![key]
          same_image = image_ids[query] >= 0 && image_ids[query] == image_ids[key]
          if query >= key || same_image
            allowed = true
            break
          end
        end
        raise ArgumentError.new("attention row has no valid keys") unless allowed
      end
    end

    private def validate_input_aliases!(ptr : DevicePtr) : Nil
      input_bytes = (@tokens.to_i64 * @config.hidden_dim * sizeof(Float32)).to_u64
      @buffers.each do |buffer|
        # The dedicated upload buffer is a legal input source. Every other
        # runner allocation is output, scratch, metadata, or checkpoint state.
        next if buffer.ptr == @hidden_input.ptr
        next unless ptr < buffer.ptr + buffer.bytesize && buffer.ptr < ptr + input_bytes

        raise ArgumentError.new("device input must not alias runner-owned output or scratch")
      end
    end

    private def rope_tables(positions : Array(StaticArray(Int32, 3))) : Tuple(Array(Float32), Array(Float32))
      pair_count = @config.head_dim // 2
      cos_values = Array(Float32).new(@tokens * pair_count, 0.0_f32)
      sin_values = Array(Float32).new(@tokens * pair_count, 0.0_f32)
      @tokens.times do |token|
        pair_offset = 0
        3.times do |axis|
          axis_dim = @config.axes_dims[axis]
          (axis_dim // 2).times do |pair|
            frequency = 1.0_f64 / (@config.rope_theta.to_f64 ** ((pair * 2).to_f64 / axis_dim))
            angle = positions[token][axis].to_f64 * frequency
            index = token * pair_count + pair_offset + pair
            cos_values[index] = Math.cos(angle).to_f32
            sin_values[index] = Math.sin(angle).to_f32
          end
          pair_offset += axis_dim // 2
        end
      end
      {cos_values, sin_values}
    end

    private def setup_kernel_params : Nil
      dim = @config.hidden_dim.to_u32
      heads = @config.heads.to_u32
      head_dim = @config.head_dim.to_u32
      tokens = @tokens.to_u32
      eps = @config.eps

      add_kernel_params("layernorm", [@input_device_ptr, @modulation_device.ptr,
                                      checkpoint("norm1").ptr, @gate1.ptr, tokens, dim, eps, 0_u32] of KernelParam)
      add_kernel_params("rmsnorm", [0_u64, 0_u64, 0_u64, tokens, heads, head_dim, eps] of KernelParam)
      add_kernel_params("rope", [0_u64, 0_u64, @rope_cos_device.ptr, @rope_sin_device.ptr,
                                 tokens, heads, head_dim] of KernelParam)
      add_kernel_params("attention", [0_u64, 0_u64, 0_u64, @image_ids_device.ptr,
                                      @key_valid_device.ptr, 0_u64, tokens, heads, head_dim,
                                      (1.0_f64 / Math.sqrt(@config.head_dim)).to_f32] of KernelParam)
      add_kernel_params("residual_norm", [@input_device_ptr, 0_u64, @gate1.ptr,
                                          @modulation_device.ptr, checkpoint("state").ptr, checkpoint("norm2").ptr,
                                          @gate2.ptr, tokens, dim, eps] of KernelParam)
      add_kernel_params("swiglu", [checkpoint("gate_up").ptr, checkpoint("activated").ptr,
                                   tokens, @config.intermediate_dim.to_u32] of KernelParam)
      add_kernel_params("residual_add", [checkpoint("state").ptr, @gate2.ptr,
                                         checkpoint("mlp_out").ptr, checkpoint("output").ptr,
                                         (@tokens * dim.to_i32).to_u32] of KernelParam)
    end

    private def launch_1d(name : String, grid_x : Int32, block_x : UInt32, label : String) : Nil
      ML::CUDA.launch!(@functions[name], grid_x.to_u32, 1_u32, 1_u32,
        block_x, 1_u32, 1_u32, @kernel_params[name].as(Void**), label)
    end

    private def launch_2d(name : String, grid_x : Int32, grid_y : Int32,
                          block_x : UInt32, label : String,
                          ptr0 : DevicePtr, ptr1 : DevicePtr, ptr2 : DevicePtr,
                          ptr3 : DevicePtr = 0_u64) : Nil
      case name
      when "rmsnorm"
        set_param_ptr(name, 0, ptr0)
        set_param_ptr(name, 1, ptr1)
        set_param_ptr(name, 2, ptr2)
      when "rope"
        set_param_ptr(name, 0, ptr0)
        set_param_ptr(name, 1, ptr1)
      when "attention"
        set_param_ptr(name, 0, ptr0)
        set_param_ptr(name, 1, ptr1)
        set_param_ptr(name, 2, ptr2)
        set_param_ptr(name, 5, ptr3)
      else
        raise ArgumentError.new("unsupported 2-D Qwen-Image kernel #{name}")
      end
      ML::CUDA.launch!(@functions[name], grid_x.to_u32, grid_y.to_u32, 1_u32,
        block_x, 1_u32, 1_u32, @kernel_params[name].as(Void**), label)
    end

    private def launch_swiglu : Nil
      grid_y = ((@config.intermediate_dim + 127) // 128).to_u32
      ML::CUDA.launch!(@functions["swiglu"], @tokens.to_u32, grid_y, 1_u32,
        128_u32, 1_u32, 1_u32, @kernel_params["swiglu"].as(Void**), "Qwen-Image SwiGLU")
    end

    private def launch_residual_add : Nil
      count = @tokens * @config.hidden_dim
      grid_x = ((count + 255) // 256).to_u32
      ML::CUDA.launch!(@functions["residual_add"], grid_x, 1_u32, 1_u32,
        256_u32, 1_u32, 1_u32, @kernel_params["residual_add"].as(Void**), "Qwen-Image MLP residual")
    end

    private def add_kernel_params(name : String, values : Array(KernelParam)) : Nil
      params = Pointer(Void*).malloc(values.size)
      values.each_with_index do |value, index|
        params[index] = case value
                        when UInt64  then box_u64(value).as(Void*)
                        when UInt32  then box_u32(value).as(Void*)
                        when Float32 then box_f32(value).as(Void*)
                        else              raise "unsupported Qwen-Image PTX parameter"
                        end
      end
      @parameter_keepalive << params.as(Void*)
      @kernel_params[name] = params
    end

    private def set_param_ptr(name : String, index : Int32, value : DevicePtr) : Nil
      # Each pointer argument was boxed once during setup. Mutating its value
      # keeps cuLaunchKernel's by-value parameter storage stable without
      # allocating on the enqueue path.
      @kernel_params[name][index].as(Pointer(DevicePtr)).value = value
    end

    private def box_u64(value : UInt64) : Pointer(UInt64)
      box = Pointer(UInt64).malloc(1)
      box.value = value
      @parameter_keepalive << box.as(Void*)
      box
    end

    private def box_u32(value : UInt32) : Pointer(UInt32)
      box = Pointer(UInt32).malloc(1)
      box.value = value
      @parameter_keepalive << box.as(Void*)
      box
    end

    private def box_f32(value : Float32) : Pointer(Float32)
      box = Pointer(Float32).malloc(1)
      box.value = value
      @parameter_keepalive << box.as(Void*)
      box
    end

    private def checkpoint(name : String) : DeviceBuffer
      @checkpoints[name]? || raise ArgumentError.new("unknown checkpoint #{name}")
    end

    private def checkpoint_element_count(name : String) : Int32
      width = case name
              when "gate_up"   then 2 * @config.intermediate_dim
              when "activated" then @config.intermediate_dim
              else                  @config.hidden_dim
              end
      @tokens * width
    end

    private def allocate_f32(elements : Int32) : DeviceBuffer
      allocate((elements.to_i64 * sizeof(Float32)).to_u64)
    end

    private def allocate(bytes : UInt64) : DeviceBuffer
      buffer = DeviceBuffer.new(bytes)
      @buffers << buffer
      buffer
    end

    private def upload_f32(values : Array(Float32), label : String) : DeviceBuffer
      buffer = allocate_f32(values.size.to_i32)
      ML::CUDA.copy_htod!(buffer.ptr, values.to_unsafe.as(Void*), bytesize_f32(values.size.to_i32), label)
      buffer
    end

    private def upload_i32(values : Array(Int32), label : String) : DeviceBuffer
      buffer = allocate((values.size * sizeof(Int32)).to_u64)
      ML::CUDA.copy_htod!(buffer.ptr, values.to_unsafe.as(Void*), (values.size * sizeof(Int32)).to_u64, label)
      buffer
    end

    private def upload_u8(values : Array(UInt8), label : String) : DeviceBuffer
      buffer = allocate(values.size.to_u64)
      ML::CUDA.copy_htod!(buffer.ptr, values.to_unsafe.as(Void*), values.size.to_u64, label)
      buffer
    end

    private def bytesize_f32(elements : Int32) : LibC::SizeT
      (elements.to_i64 * sizeof(Float32)).to_u64
    end
  end
end
