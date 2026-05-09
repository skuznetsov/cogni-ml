require "./driver"
require "../gguf/reader"

module ML::CUDA
  # CUDA output-head boundary: output RMSNorm followed by quantized lm_head
  # projection. This reads logits back for correctness; a future fast decode
  # path should replace readback with a resident top1/topK kernel.
  class QwenOutputHeadRunner
    DN_PTX  = {{ read_file("src/ml/cuda/kernels/deltanet_step_probe.ptx") }}
    Q4K_PTX = {{ read_file("src/ml/cuda/kernels/q4k_gemv_probe.ptx") }}
    Q6K_PTX = {{ read_file("src/ml/cuda/kernels/q6k_gemv_probe.ptx") }}

    class Weights
      getter norm : Array(Float32)
      getter output_raw : Bytes
      getter output_type : ML::GGUF::TensorType
      getter hidden : Int32
      getter vocab : Int32

      def self.load(gguf : ML::GGUF::GGUFFile) : self
        norm_info = gguf.tensor("output_norm.weight") || raise "missing output_norm.weight"
        output_info = gguf.tensor("output.weight") || gguf.tensor("token_embd.weight") || raise "missing output/token_embd weight"
        raise "expected Q4_K/Q6_K output weight" unless output_info.type.q4_k? || output_info.type.q6_k?
        new(gguf.read_tensor_f32(norm_info),
          gguf.read_tensor_raw(output_info), output_info.type,
          output_info.dims[0].to_i32, output_info.dims[1].to_i32)
      end

      def initialize(@norm : Array(Float32),
                     @output_raw : Bytes,
                     @output_type : ML::GGUF::TensorType,
                     @hidden : Int32,
                     @vocab : Int32)
      end
    end

    getter logits_gpu_all : Array(Float32)
    getter logits_device_ptr : DevicePtr

    def self.from_weights(weights : Weights,
                          tokens : Int32,
                          xs : Array(Float32),
                          eps : Float32) : self
      new(tokens, weights.hidden, weights.vocab, xs, weights.norm,
        weights.output_raw, weights.output_type, eps)
    end

    private def initialize(@tokens : Int32,
                           @hidden : Int32,
                           @vocab : Int32,
                           @xs : Array(Float32),
                           @norm : Array(Float32),
                           @output_raw : Bytes,
                           @output_type : ML::GGUF::TensorType,
                           @eps : Float32)
      raise ArgumentError.new("tokens must be positive") unless @tokens > 0
      raise ArgumentError.new("xs size mismatch") unless @xs.size == @tokens * @hidden
      raise ArgumentError.new("norm size mismatch") unless @norm.size == @hidden

      @modules = [] of CUDAModule
      @buffers = [] of DeviceBuffer
      @param_keepalive = [] of Void*
      @input_device_base = nil.as(DevicePtr?)
      @owned_input_device_ptr = nil.as(DevicePtr?)
      @logits_device_ptr = 0_u64
      @logits_gpu_all = Array(Float32).new(@tokens * @vocab, 0.0_f32)
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

    def top1_ids : Array(Int32)
      ids = Array(Int32).new(@tokens)
      @tokens.times do |tok|
        base = tok * @vocab
        best_id = 0
        best = @logits_gpu_all[base]
        (1...@vocab).each do |i|
          value = @logits_gpu_all[base + i]
          if value > best
            best = value
            best_id = i
          end
        end
        ids << best_id
      end
      ids
    end

    def close : Nil
      return if @closed

      @buffers.each(&.close)
      @modules.each(&.close)
      @closed = true
    end

    private def build_runner : Nil
      dn_mod = CUDAModule.load(DN_PTX, "delta_head")
      q4_mod = CUDAModule.load(Q4K_PTX, "q4_head")
      q6_mod = CUDAModule.load(Q6K_PTX, "q6_head")
      @modules.concat([dn_mod, q4_mod, q6_mod])

      norm_fn = dn_mod.function("rmsnorm_vec_probe")
      q4_fn = q4_mod.function("q4_k_gemv_warp4_f32")
      q6_fn = q6_mod.function("q6_k_gemv_warp4_f32")
      output_fn = @output_type.q4_k? ? q4_fn : q6_fn

      sizes = [bytesize_f32(@tokens * @hidden), bytesize_f32(@hidden), bytesize_f32(@hidden),
               @output_raw.size.to_u64, bytesize_f32(@tokens * @hidden), bytesize_f32(@tokens * @vocab)]
      ptrs = sizes.map do |size_bytes|
        buffer = DeviceBuffer.new(size_bytes)
        @buffers << buffer
        buffer.ptr
      end
      d_xs, d_norm_w, d_normed, d_output_w, d_normed_all, d_logits_all = ptrs
      @owned_input_device_ptr = d_xs
      @input_device_base = d_xs
      @logits_device_ptr = d_logits_all

      upload_weights = -> {
        ML::CUDA.copy_htod!(d_norm_w, @norm.to_unsafe.as(Void*), bytesize_f32(@norm.size), "output_norm")
        ML::CUDA.copy_htod!(d_output_w, @output_raw.to_unsafe.as(Void*), @output_raw.size.to_u64, "output_w")
      }

      reset_sequence = -> {
        if @input_device_base == d_xs
          ML::CUDA.copy_htod!(d_xs, @xs.to_unsafe.as(Void*), bytesize_f32(@tokens * @hidden), "output_head_xs")
        end
      }

      hidden_u32 = @hidden.to_u32
      vocab_u32 = @vocab.to_u32
      vocab_grid = ((@vocab + 3) // 4).to_u32
      d_x_cur_ptr = box_ptr(d_xs)
      d_normed_cur_ptr = box_ptr(d_normed)
      d_logits_cur_ptr = box_ptr(d_logits_all)

      norm_params = Pointer(Void*).malloc(5)
      norm_params[0] = d_x_cur_ptr.as(Void*)
      norm_params[1] = box_ptr(d_norm_w).as(Void*)
      norm_params[2] = d_normed_cur_ptr.as(Void*)
      norm_params[3] = box_u32(hidden_u32).as(Void*)
      norm_params[4] = box_f32(@eps).as(Void*)

      output_params = Pointer(Void*).malloc(5)
      output_params[0] = box_ptr(d_output_w).as(Void*)
      output_params[1] = d_normed_cur_ptr.as(Void*)
      output_params[2] = d_logits_cur_ptr.as(Void*)
      output_params[3] = box_u32(hidden_u32).as(Void*)
      output_params[4] = box_u32(vocab_u32).as(Void*)

      run_token = ->(tok : Int32) {
        d_x_cur_ptr.value = @input_device_base.not_nil! + bytesize_f32(tok * @hidden)
        d_normed_cur_ptr.value = d_normed_all + bytesize_f32(tok * @hidden)
        d_logits_cur_ptr.value = d_logits_all + bytesize_f32(tok * @vocab)
        ML::CUDA.launch!(norm_fn, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, norm_params, "output norm")
        ML::CUDA.launch!(output_fn, vocab_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, output_params, "output logits")
      }

      read_outputs = -> {
        ML::CUDA.copy_dtoh!(@logits_gpu_all.to_unsafe.as(Void*), d_logits_all, bytesize_f32(@logits_gpu_all.size), "output_logits")
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
