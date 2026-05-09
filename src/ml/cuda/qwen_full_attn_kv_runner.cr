require "./driver"
require "../gguf/reader"

module ML::CUDA
  # Post-projection boundary for full-attention CUDA work:
  # split Q/gate, RMSNorm Q/K, apply precomputed RoPE tables, and append K/V
  # rows to the layer KV cache.
  class QwenFullAttnKVRunner
    FULL_ATTN_POST_PTX = {{ read_file("src/ml/cuda/kernels/fullattn_post_probe.ptx") }}

    class Weights
      getter q_norm : Array(Float32)
      getter k_norm : Array(Float32)

      def self.load(gguf : ML::GGUF::GGUFFile, layer : Int32) : self
        prefix = "blk.#{layer}"
        q_norm_info = gguf.tensor("#{prefix}.attn_q_norm.weight") || raise "missing #{prefix}.attn_q_norm.weight"
        k_norm_info = gguf.tensor("#{prefix}.attn_k_norm.weight") || raise "missing #{prefix}.attn_k_norm.weight"
        new(gguf.read_tensor_f32(q_norm_info), gguf.read_tensor_f32(k_norm_info))
      end

      def initialize(@q_norm : Array(Float32), @k_norm : Array(Float32))
      end
    end

    getter q_gpu_all : Array(Float32)
    getter gate_gpu_all : Array(Float32)
    getter k_gpu_all : Array(Float32)
    getter attn_gpu_all : Array(Float32)
    getter k_cache_gpu : Array(Float32)
    getter v_cache_gpu : Array(Float32)

    def initialize(@tokens : Int32,
                   @max_seq : Int32,
                   @start_pos : Int32,
                   @n_head : Int32,
                   @n_head_kv : Int32,
                   @head_dim : Int32,
                   @rope_dim : Int32,
                   @eps : Float32,
                   @q_full_device_ptr : DevicePtr,
                   @k_device_ptr : DevicePtr,
                   @v_device_ptr : DevicePtr,
                   weights : Weights,
                   cos_table : Array(Float32),
                   sin_table : Array(Float32))
      raise ArgumentError.new("tokens must be positive") unless @tokens > 0
      raise ArgumentError.new("max_seq too small") unless @max_seq >= @start_pos + @tokens
      raise ArgumentError.new("rope_dim must be even") unless @rope_dim.even?
      raise ArgumentError.new("q_norm size mismatch") unless weights.q_norm.size == @head_dim
      raise ArgumentError.new("k_norm size mismatch") unless weights.k_norm.size == @head_dim
      half = @rope_dim // 2
      raise ArgumentError.new("cos/sin table size mismatch") unless cos_table.size == @tokens * half && sin_table.size == @tokens * half

      @q_norm = weights.q_norm
      @k_norm = weights.k_norm
      @cos_table = cos_table
      @sin_table = sin_table
      @q_dim = @n_head * @head_dim
      @kv_dim = @n_head_kv * @head_dim
      @q_gpu_all = Array(Float32).new(@tokens * @q_dim, 0.0_f32)
      @gate_gpu_all = Array(Float32).new(@tokens * @q_dim, 0.0_f32)
      @k_gpu_all = Array(Float32).new(@tokens * @kv_dim, 0.0_f32)
      @attn_gpu_all = Array(Float32).new(@tokens * @q_dim, 0.0_f32)
      @k_cache_gpu = Array(Float32).new(@max_seq * @kv_dim, 0.0_f32)
      @v_cache_gpu = Array(Float32).new(@max_seq * @kv_dim, 0.0_f32)
      @modules = [] of CUDAModule
      @buffers = [] of DeviceBuffer
      @param_keepalive = [] of Void*
      @closed = false

      build_runner
    end

    def upload_constants : Nil
      runner.upload_weights
    end

    def run_sequence : Nil
      runner.reset_sequence
      runner.run_sequence
    end

    def read_outputs : Nil
      runner.read_outputs
    end

    def close : Nil
      return if @closed

      @buffers.each(&.close)
      @modules.each(&.close)
      @closed = true
    end

    private def build_runner : Nil
      mod = CUDAModule.load(FULL_ATTN_POST_PTX, "fullattn_post")
      @modules << mod
      q_fn = mod.function("full_attn_q_split_norm_rope_probe")
      k_fn = mod.function("full_attn_k_norm_rope_cache_probe")
      attn_fn = mod.function("full_attn_decode_cache_probe")

      sizes = [bytesize_f32(@q_norm.size), bytesize_f32(@k_norm.size),
               bytesize_f32(@cos_table.size), bytesize_f32(@sin_table.size),
               bytesize_f32(@tokens * @q_dim), bytesize_f32(@tokens * @q_dim),
               bytesize_f32(@tokens * @kv_dim), bytesize_f32(@max_seq * @kv_dim),
               bytesize_f32(@max_seq * @kv_dim), bytesize_f32(@tokens * @n_head * @max_seq),
               bytesize_f32(@tokens * @q_dim)]
      ptrs = sizes.map do |size_bytes|
        buffer = DeviceBuffer.new(size_bytes)
        @buffers << buffer
        buffer.ptr
      end
      d_q_norm, d_k_norm, d_cos, d_sin, d_q_out, d_gate_out, d_k_out, d_k_cache, d_v_cache, d_scores, d_attn_out = ptrs

      upload_constants = -> {
        ML::CUDA.copy_htod!(d_q_norm, @q_norm.to_unsafe.as(Void*), bytesize_f32(@q_norm.size), "q_norm")
        ML::CUDA.copy_htod!(d_k_norm, @k_norm.to_unsafe.as(Void*), bytesize_f32(@k_norm.size), "k_norm")
        ML::CUDA.copy_htod!(d_cos, @cos_table.to_unsafe.as(Void*), bytesize_f32(@cos_table.size), "cos")
        ML::CUDA.copy_htod!(d_sin, @sin_table.to_unsafe.as(Void*), bytesize_f32(@sin_table.size), "sin")
      }

      reset_sequence = -> {
        # Projection outputs are already device-resident inputs. KV cache rows
        # written by this probe are overwritten before comparison.
      }

      n_head_u32 = @n_head.to_u32
      n_head_kv_u32 = @n_head_kv.to_u32
      head_dim_u32 = @head_dim.to_u32
      rope_dim_u32 = @rope_dim.to_u32
      start_pos_u32 = @start_pos.to_u32
      max_seq_u32 = @max_seq.to_u32
      heads_per_group_u32 = (@n_head // @n_head_kv).to_u32
      scale = (1.0_f64 / Math.sqrt(@head_dim.to_f64)).to_f32

      q_params = Pointer(Void*).malloc(10)
      q_params[0] = box_ptr(@q_full_device_ptr).as(Void*)
      q_params[1] = box_ptr(d_q_norm).as(Void*)
      q_params[2] = box_ptr(d_q_out).as(Void*)
      q_params[3] = box_ptr(d_gate_out).as(Void*)
      q_params[4] = box_ptr(d_cos).as(Void*)
      q_params[5] = box_ptr(d_sin).as(Void*)
      q_params[6] = box_u32(n_head_u32).as(Void*)
      q_params[7] = box_u32(head_dim_u32).as(Void*)
      q_params[8] = box_u32(rope_dim_u32).as(Void*)
      q_params[9] = box_f32(@eps).as(Void*)

      k_params = Pointer(Void*).malloc(14)
      k_params[0] = box_ptr(@k_device_ptr).as(Void*)
      k_params[1] = box_ptr(@v_device_ptr).as(Void*)
      k_params[2] = box_ptr(d_k_norm).as(Void*)
      k_params[3] = box_ptr(d_k_out).as(Void*)
      k_params[4] = box_ptr(d_k_cache).as(Void*)
      k_params[5] = box_ptr(d_v_cache).as(Void*)
      k_params[6] = box_ptr(d_cos).as(Void*)
      k_params[7] = box_ptr(d_sin).as(Void*)
      k_params[8] = box_u32(n_head_kv_u32).as(Void*)
      k_params[9] = box_u32(head_dim_u32).as(Void*)
      k_params[10] = box_u32(rope_dim_u32).as(Void*)
      k_params[11] = box_u32(start_pos_u32).as(Void*)
      k_params[12] = box_u32(max_seq_u32).as(Void*)
      k_params[13] = box_f32(@eps).as(Void*)

      attn_params = Pointer(Void*).malloc(13)
      attn_params[0] = box_ptr(d_q_out).as(Void*)
      attn_params[1] = box_ptr(d_gate_out).as(Void*)
      attn_params[2] = box_ptr(d_k_cache).as(Void*)
      attn_params[3] = box_ptr(d_v_cache).as(Void*)
      attn_params[4] = box_ptr(d_scores).as(Void*)
      attn_params[5] = box_ptr(d_attn_out).as(Void*)
      attn_params[6] = box_u32(n_head_u32).as(Void*)
      attn_params[7] = box_u32(n_head_kv_u32).as(Void*)
      attn_params[8] = box_u32(head_dim_u32).as(Void*)
      attn_params[9] = box_u32(heads_per_group_u32).as(Void*)
      attn_params[10] = box_u32(start_pos_u32).as(Void*)
      attn_params[11] = box_u32(max_seq_u32).as(Void*)
      attn_params[12] = box_f32(scale).as(Void*)

      run_token = ->(tok : Int32) {
        # Kernels index token by block id; launch all token blocks once.
        if tok == 0
          ML::CUDA.launch!(q_fn, @tokens.to_u32, @n_head.to_u32, 1_u32, 1_u32, 1_u32, 1_u32, q_params, "q norm rope")
          ML::CUDA.launch!(k_fn, @tokens.to_u32, @n_head_kv.to_u32, 1_u32, 1_u32, 1_u32, 1_u32, k_params, "k norm rope cache")
          ML::CUDA.launch!(attn_fn, @tokens.to_u32, @n_head.to_u32, 1_u32, 1_u32, 1_u32, 1_u32, attn_params, "attn decode cache")
        end
      }

      read_outputs = -> {
        ML::CUDA.copy_dtoh!(@q_gpu_all.to_unsafe.as(Void*), d_q_out, bytesize_f32(@q_gpu_all.size), "q_out")
        ML::CUDA.copy_dtoh!(@gate_gpu_all.to_unsafe.as(Void*), d_gate_out, bytesize_f32(@gate_gpu_all.size), "gate_out")
        ML::CUDA.copy_dtoh!(@k_gpu_all.to_unsafe.as(Void*), d_k_out, bytesize_f32(@k_gpu_all.size), "k_out")
        ML::CUDA.copy_dtoh!(@attn_gpu_all.to_unsafe.as(Void*), d_attn_out, bytesize_f32(@attn_gpu_all.size), "attn_out")
        ML::CUDA.copy_dtoh!(@k_cache_gpu.to_unsafe.as(Void*), d_k_cache, bytesize_f32(@k_cache_gpu.size), "k_cache")
        ML::CUDA.copy_dtoh!(@v_cache_gpu.to_unsafe.as(Void*), d_v_cache, bytesize_f32(@v_cache_gpu.size), "v_cache")
      }
      @runner = ResidentSequenceRunner.new(@tokens, upload_constants, reset_sequence, run_token, read_outputs)
    end

    private def runner : ResidentSequenceRunner
      @runner.not_nil!
    end

    private def box_ptr(value : DevicePtr) : Pointer(DevicePtr)
      ptr = Pointer(DevicePtr).malloc(1)
      ptr.value = value
      @param_keepalive << ptr.as(Void*)
      ptr
    end

    private def box_u32(value : UInt32) : Pointer(UInt32)
      ptr = Pointer(UInt32).malloc(1)
      ptr.value = value
      @param_keepalive << ptr.as(Void*)
      ptr
    end

    private def box_f32(value : Float32) : Pointer(Float32)
      ptr = Pointer(Float32).malloc(1)
      ptr.value = value
      @param_keepalive << ptr.as(Void*)
      ptr
    end

    private def bytesize_f32(elements : Int32) : LibC::SizeT
      (elements * sizeof(Float32)).to_u64
    end
  end
end
