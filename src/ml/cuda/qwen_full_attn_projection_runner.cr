require "./driver"
require "../gguf/reader"

module ML::CUDA
  # Reusable CUDA runner for a Qwen full-attention layer's input projection
  # bundle: attn_q, attn_k, attn_v from the same hidden vector sequence.
  class QwenFullAttnProjectionRunner
    Q4K_PTX = {{ read_file("src/ml/cuda/kernels/q4k_gemv_probe.ptx") }}
    Q6K_PTX = {{ read_file("src/ml/cuda/kernels/q6k_gemv_probe.ptx") }}

    class Weights
      getter hidden : Int32
      getter q_dim : Int32
      getter k_dim : Int32
      getter v_dim : Int32
      getter q_raw : Bytes
      getter k_raw : Bytes
      getter v_raw : Bytes
      getter v_type : ML::GGUF::TensorType

      def self.load(gguf : ML::GGUF::GGUFFile, layer : Int32) : self
        prefix = "blk.#{layer}"
        q_info = gguf.tensor("#{prefix}.attn_q.weight") || raise "missing #{prefix}.attn_q.weight"
        k_info = gguf.tensor("#{prefix}.attn_k.weight") || raise "missing #{prefix}.attn_k.weight"
        v_info = gguf.tensor("#{prefix}.attn_v.weight") || raise "missing #{prefix}.attn_v.weight"
        raise "expected Q4_K full-attn q/k" unless q_info.type.q4_k? && k_info.type.q4_k?
        raise "expected Q4_K/Q6_K full-attn v" unless v_info.type.q4_k? || v_info.type.q6_k?

        hidden = q_info.dims[0].to_i32
        q_dim = q_info.dims[1].to_i32
        k_dim = k_info.dims[1].to_i32
        v_dim = v_info.dims[1].to_i32
        raise "full-attn input shape mismatch" unless k_info.dims[0].to_i32 == hidden && v_info.dims[0].to_i32 == hidden

        new(hidden, q_dim, k_dim, v_dim,
          gguf.read_tensor_raw(q_info), gguf.read_tensor_raw(k_info),
          gguf.read_tensor_raw(v_info), v_info.type)
      end

      def initialize(@hidden : Int32,
                     @q_dim : Int32,
                     @k_dim : Int32,
                     @v_dim : Int32,
                     @q_raw : Bytes,
                     @k_raw : Bytes,
                     @v_raw : Bytes,
                     @v_type : ML::GGUF::TensorType)
      end
    end

    getter tokens : Int32
    getter hidden : Int32
    getter q_gpu_all : Array(Float32)
    getter k_gpu_all : Array(Float32)
    getter v_gpu_all : Array(Float32)
    getter q_device_ptr : DevicePtr
    getter k_device_ptr : DevicePtr
    getter v_device_ptr : DevicePtr

    def self.from_weights(weights : Weights, tokens : Int32, xs : Array(Float32)) : self
      new(tokens, weights.hidden, weights.q_dim, weights.k_dim, weights.v_dim,
        xs, weights.q_raw, weights.k_raw, weights.v_raw, weights.v_type)
    end

    private def initialize(@tokens : Int32,
                           @hidden : Int32,
                           @q_dim : Int32,
                           @k_dim : Int32,
                           @v_dim : Int32,
                           @xs : Array(Float32),
                           @q_raw : Bytes,
                           @k_raw : Bytes,
                           @v_raw : Bytes,
                           @v_type : ML::GGUF::TensorType)
      raise ArgumentError.new("tokens must be positive") unless @tokens > 0
      raise ArgumentError.new("xs size mismatch") unless @xs.size == @tokens * @hidden

      @modules = [] of CUDAModule
      @buffers = [] of DeviceBuffer
      @param_keepalive = [] of Void*
      @q_gpu_all = Array(Float32).new(@tokens * @q_dim, 0.0_f32)
      @k_gpu_all = Array(Float32).new(@tokens * @k_dim, 0.0_f32)
      @v_gpu_all = Array(Float32).new(@tokens * @v_dim, 0.0_f32)
      @input_device_base = nil.as(DevicePtr?)
      @owned_input_device_ptr = nil.as(DevicePtr?)
      @q_device_ptr = 0_u64
      @k_device_ptr = 0_u64
      @v_device_ptr = 0_u64
      @closed = false

      build_runner
    end

    def upload_weights : Nil
      runner.upload_weights
    end

    def replace_sequence_input(xs : Array(Float32)) : Nil
      raise ArgumentError.new("xs size mismatch") unless xs.size == @tokens * @hidden

      @xs = xs
      @input_device_base = @owned_input_device_ptr
    end

    def use_device_sequence_input(ptr : DevicePtr) : Nil
      raise ArgumentError.new("device input pointer must be non-zero") if ptr == 0_u64

      @input_device_base = ptr
    end

    def reset_sequence : Nil
      runner.reset_sequence
    end

    def run_sequence : Nil
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
      q4_mod = CUDAModule.load(Q4K_PTX, "q4")
      q6_mod = CUDAModule.load(Q6K_PTX, "q6")
      @modules.concat([q4_mod, q6_mod])

      q4_fn = q4_mod.function("q4_k_gemv_warp4_f32")
      q6_fn = q6_mod.function("q6_k_gemv_warp4_f32")
      v_fn = @v_type.q4_k? ? q4_fn : q6_fn

      sizes = [bytesize_f32(@tokens * @hidden), @q_raw.size.to_u64, @k_raw.size.to_u64, @v_raw.size.to_u64,
               bytesize_f32(@tokens * @q_dim), bytesize_f32(@tokens * @k_dim), bytesize_f32(@tokens * @v_dim)]
      ptrs = sizes.map do |size_bytes|
        buffer = DeviceBuffer.new(size_bytes)
        @buffers << buffer
        buffer.ptr
      end
      d_xs, d_q_w, d_k_w, d_v_w, d_q_all, d_k_all, d_v_all = ptrs
      @owned_input_device_ptr = d_xs
      @input_device_base = d_xs
      @q_device_ptr = d_q_all
      @k_device_ptr = d_k_all
      @v_device_ptr = d_v_all

      upload_weights = -> {
        ML::CUDA.copy_htod!(d_q_w, @q_raw.to_unsafe.as(Void*), @q_raw.size.to_u64, "attn_q_w")
        ML::CUDA.copy_htod!(d_k_w, @k_raw.to_unsafe.as(Void*), @k_raw.size.to_u64, "attn_k_w")
        ML::CUDA.copy_htod!(d_v_w, @v_raw.to_unsafe.as(Void*), @v_raw.size.to_u64, "attn_v_w")
      }

      reset_sequence = -> {
        if @input_device_base == d_xs
          ML::CUDA.copy_htod!(d_xs, @xs.to_unsafe.as(Void*), bytesize_f32(@tokens * @hidden), "xs")
        end
      }

      hidden_u32 = @hidden.to_u32
      q_dim_u32 = @q_dim.to_u32
      k_dim_u32 = @k_dim.to_u32
      v_dim_u32 = @v_dim.to_u32
      q_grid = ((@q_dim + 3) // 4).to_u32
      k_grid = ((@k_dim + 3) // 4).to_u32
      v_grid = ((@v_dim + 3) // 4).to_u32

      d_x_cur_ptr = box_ptr(d_xs)
      d_q_cur_ptr = box_ptr(d_q_all)
      d_k_cur_ptr = box_ptr(d_k_all)
      d_v_cur_ptr = box_ptr(d_v_all)

      q_params = Pointer(Void*).malloc(5)
      q_params[0] = box_ptr(d_q_w).as(Void*)
      q_params[1] = d_x_cur_ptr.as(Void*)
      q_params[2] = d_q_cur_ptr.as(Void*)
      q_params[3] = box_u32(hidden_u32).as(Void*)
      q_params[4] = box_u32(q_dim_u32).as(Void*)

      k_params = Pointer(Void*).malloc(5)
      k_params[0] = box_ptr(d_k_w).as(Void*)
      k_params[1] = d_x_cur_ptr.as(Void*)
      k_params[2] = d_k_cur_ptr.as(Void*)
      k_params[3] = box_u32(hidden_u32).as(Void*)
      k_params[4] = box_u32(k_dim_u32).as(Void*)

      v_params = Pointer(Void*).malloc(5)
      v_params[0] = box_ptr(d_v_w).as(Void*)
      v_params[1] = d_x_cur_ptr.as(Void*)
      v_params[2] = d_v_cur_ptr.as(Void*)
      v_params[3] = box_u32(hidden_u32).as(Void*)
      v_params[4] = box_u32(v_dim_u32).as(Void*)

      run_token = ->(tok : Int32) {
        x_offset = bytesize_f32(tok * @hidden)
        d_x_cur_ptr.value = @input_device_base.not_nil! + x_offset
        d_q_cur_ptr.value = d_q_all + bytesize_f32(tok * @q_dim)
        d_k_cur_ptr.value = d_k_all + bytesize_f32(tok * @k_dim)
        d_v_cur_ptr.value = d_v_all + bytesize_f32(tok * @v_dim)
        ML::CUDA.launch!(q4_fn, q_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, q_params, "attn q")
        ML::CUDA.launch!(q4_fn, k_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, k_params, "attn k")
        ML::CUDA.launch!(v_fn, v_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, v_params, "attn v")
      }

      read_outputs = -> {
        ML::CUDA.copy_dtoh!(@q_gpu_all.to_unsafe.as(Void*), d_q_all, bytesize_f32(@q_gpu_all.size), "q_all")
        ML::CUDA.copy_dtoh!(@k_gpu_all.to_unsafe.as(Void*), d_k_all, bytesize_f32(@k_gpu_all.size), "k_all")
        ML::CUDA.copy_dtoh!(@v_gpu_all.to_unsafe.as(Void*), d_v_all, bytesize_f32(@v_gpu_all.size), "v_all")
      }
      @runner = ResidentSequenceRunner.new(@tokens, upload_weights, reset_sequence, run_token, read_outputs)
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

    private def bytesize_f32(elements : Int32) : LibC::SizeT
      (elements * sizeof(Float32)).to_u64
    end
  end
end
