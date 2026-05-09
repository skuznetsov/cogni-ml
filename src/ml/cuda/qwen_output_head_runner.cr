require "./driver"
require "../gguf/reader"

module ML::CUDA
  # CUDA output-head boundary: output RMSNorm followed by quantized lm_head
  # projection and resident top1. Full logits readback is optional for
  # correctness attribution; the default decode-facing path reads only top1.
  class QwenOutputHeadRunner
    @profile_runner : ResidentSequenceRunner?

    DN_PTX   = {{ read_file("src/ml/cuda/kernels/deltanet_step_probe.ptx") }}
    Q4K_PTX  = {{ read_file("src/ml/cuda/kernels/q4k_gemv_probe.ptx") }}
    Q6K_PTX  = {{ read_file("src/ml/cuda/kernels/q6k_gemv_probe.ptx") }}
    TOP1_PTX = <<-PTX
    .version 8.0
    .target sm_80
    .address_size 64

    .visible .entry output_top1_partial_scan_probe(
        .param .u64 logits,
        .param .u64 partial_ids,
        .param .u64 partial_values,
        .param .u32 vocab,
        .param .u32 partial_stride
    )
    {
        .reg .pred %p<4>;
        .reg .b32 %r<17>;
        .reg .b64 %rd<14>;
        .reg .f32 %f<4>;

        ld.param.u64 %rd1, [logits];
        ld.param.u64 %rd2, [partial_ids];
        ld.param.u64 %rd3, [partial_values];
        ld.param.u32 %r1, [vocab];
        ld.param.u32 %r8, [partial_stride];

        mov.u32 %r2, %tid.x;
        mov.u32 %r9, %ctaid.x;
        setp.ge.u32 %p1, %r2, %r1;
        @%p1 bra EMPTY;

        mul.wide.u32 %rd4, %r2, 4;
        add.s64 %rd5, %rd1, %rd4;
        ld.global.f32 %f1, [%rd5];
        mov.u32 %r3, %r2;
        mov.u32 %r12, %ntid.x;
        add.u32 %r4, %r2, %r12;
        bra LOOP;

    LOOP:
        setp.ge.u32 %p2, %r4, %r1;
        @%p2 bra STORE;
        mul.wide.u32 %rd4, %r4, 4;
        add.s64 %rd5, %rd1, %rd4;
        ld.global.f32 %f2, [%rd5];
        setp.gt.f32 %p3, %f2, %f1;
        @%p3 bra UPDATE;
        bra NEXT;

    UPDATE:
        mov.f32 %f1, %f2;
        mov.u32 %r3, %r4;

    NEXT:
        add.u32 %r4, %r4, %r12;
        bra LOOP;

    STORE:
        mul.lo.u32 %r10, %r9, %r8;
        add.u32 %r11, %r10, %r2;
        mul.wide.u32 %rd6, %r11, 4;
        add.s64 %rd7, %rd2, %rd6;
        add.s64 %rd8, %rd3, %rd6;
        st.global.u32 [%rd7], %r3;
        st.global.f32 [%rd8], %f1;
        bra DONE;

    EMPTY:
        mov.f32 %f1, 0fFF7FFFFF;
        mov.u32 %r3, 0;
        mul.lo.u32 %r10, %r9, %r8;
        add.u32 %r11, %r10, %r2;
        mul.wide.u32 %rd6, %r11, 4;
        add.s64 %rd7, %rd2, %rd6;
        add.s64 %rd8, %rd3, %rd6;
        st.global.u32 [%rd7], %r3;
        st.global.f32 [%rd8], %f1;

    DONE:
        ret;
    }

    .visible .entry output_top1_partial_reduce_probe(
        .param .u64 partial_ids,
        .param .u64 partial_values,
        .param .u64 out_ids,
        .param .u64 out_values,
        .param .u32 partial_stride
    )
    {
        .reg .pred %p<4>;
        .reg .b32 %r<16>;
        .reg .b64 %rd<18>;
        .reg .f32 %f<6>;

        ld.param.u64 %rd1, [partial_ids];
        ld.param.u64 %rd2, [partial_values];
        ld.param.u64 %rd3, [out_ids];
        ld.param.u64 %rd4, [out_values];
        ld.param.u32 %r1, [partial_stride];

        mov.u32 %r2, %tid.x;
        setp.ne.u32 %p1, %r2, 0;
        @%p1 bra DONE2;

        mov.u32 %r3, %ctaid.x;
        mul.lo.u32 %r4, %r3, %r1;
        mul.wide.u32 %rd5, %r4, 4;
        add.s64 %rd6, %rd1, %rd5;
        add.s64 %rd7, %rd2, %rd5;
        ld.global.u32 %r5, [%rd6];
        ld.global.f32 %f1, [%rd7];
        mov.u32 %r6, 1;

    LOOP2:
        setp.ge.u32 %p2, %r6, %r1;
        @%p2 bra STORE2;
        add.u32 %r7, %r4, %r6;
        mul.wide.u32 %rd8, %r7, 4;
        add.s64 %rd9, %rd1, %rd8;
        add.s64 %rd10, %rd2, %rd8;
        ld.global.u32 %r8, [%rd9];
        ld.global.f32 %f2, [%rd10];
        setp.gt.f32 %p3, %f2, %f1;
        @%p3 bra UPDATE2;
        bra NEXT2;

    UPDATE2:
        mov.f32 %f1, %f2;
        mov.u32 %r5, %r8;

    NEXT2:
        add.u32 %r6, %r6, 1;
        bra LOOP2;

    STORE2:
        mul.wide.u32 %rd11, %r3, 4;
        add.s64 %rd12, %rd3, %rd11;
        add.s64 %rd13, %rd4, %rd11;
        st.global.u32 [%rd12], %r5;
        st.global.f32 [%rd13], %f1;

    DONE2:
        ret;
    }
    PTX

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
    getter top1_ids_gpu : Array(Int32)
    getter top1_values_gpu : Array(Float32)
    getter logits_device_ptr : DevicePtr

    def self.from_weights(weights : Weights,
                          tokens : Int32,
                          xs : Array(Float32),
                          eps : Float32,
                          read_logits : Bool = false) : self
      new(tokens, weights.hidden, weights.vocab, xs, weights.norm,
        weights.output_raw, weights.output_type, eps, read_logits)
    end

    private def initialize(@tokens : Int32,
                           @hidden : Int32,
                           @vocab : Int32,
                           @xs : Array(Float32),
                           @norm : Array(Float32),
                           @output_raw : Bytes,
                           @output_type : ML::GGUF::TensorType,
                           @eps : Float32,
                           @read_logits : Bool)
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
      @top1_ids_gpu = Array(Int32).new(@tokens, 0)
      @top1_values_gpu = Array(Float32).new(@tokens, 0.0_f32)
      @profile_head_norm_ms = 0.0
      @profile_head_logits_ms = 0.0
      @profile_head_top1_ms = 0.0
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

    def run_sequence_profiled(phase_lines : Array(String)) : Nil
      @profile_head_norm_ms = 0.0
      @profile_head_logits_ms = 0.0
      @profile_head_top1_ms = 0.0
      t_total = Time.instant
      profile_runner.run_sequence
      phase_lines << "phase_head_norm_ms=#{@profile_head_norm_ms.round(3)}"
      phase_lines << "phase_head_logits_ms=#{@profile_head_logits_ms.round(3)}"
      phase_lines << "phase_head_top1_ms=#{@profile_head_top1_ms.round(3)}"
      phase_lines << "phase_head_profiled_ms=#{((Time.instant - t_total).total_milliseconds).round(3)}"
    end

    def read_outputs : Nil
      runner.read_outputs
    end

    def top1_ids : Array(Int32)
      @top1_ids_gpu
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
      top1_mod = CUDAModule.load(TOP1_PTX, "top1_head")
      @modules.concat([dn_mod, q4_mod, q6_mod, top1_mod])

      norm_fn = dn_mod.function("rmsnorm_vec_probe")
      q4_fn = q4_mod.function("q4_k_gemv_warp4_f32")
      q6_fn = q6_mod.function("q6_k_gemv_warp4_f32")
      top1_scan_fn = top1_mod.function("output_top1_partial_scan_probe")
      top1_reduce_fn = top1_mod.function("output_top1_partial_reduce_probe")
      output_fn = @output_type.q4_k? ? q4_fn : q6_fn
      top1_width = 256_u32

      sizes = [bytesize_f32(@tokens * @hidden), bytesize_f32(@hidden), bytesize_f32(@hidden),
               @output_raw.size.to_u64, bytesize_f32(@tokens * @hidden), bytesize_f32(@tokens * @vocab),
               bytesize_i32(top1_width.to_i32), bytesize_f32(top1_width.to_i32),
               bytesize_i32(@tokens), bytesize_f32(@tokens)]
      ptrs = sizes.map do |size_bytes|
        buffer = DeviceBuffer.new(size_bytes)
        @buffers << buffer
        buffer.ptr
      end
      d_xs, d_norm_w, d_normed, d_output_w, d_normed_all, d_logits_all, d_partial_ids, d_partial_values, d_top1_ids, d_top1_values = ptrs
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
      d_top1_id_cur_ptr = box_ptr(d_top1_ids)
      d_top1_value_cur_ptr = box_ptr(d_top1_values)

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

      top1_scan_params = Pointer(Void*).malloc(5)
      top1_scan_params[0] = d_logits_cur_ptr.as(Void*)
      top1_scan_params[1] = box_ptr(d_partial_ids).as(Void*)
      top1_scan_params[2] = box_ptr(d_partial_values).as(Void*)
      top1_scan_params[3] = box_u32(vocab_u32).as(Void*)
      top1_scan_params[4] = box_u32(top1_width).as(Void*)

      top1_reduce_params = Pointer(Void*).malloc(5)
      top1_reduce_params[0] = box_ptr(d_partial_ids).as(Void*)
      top1_reduce_params[1] = box_ptr(d_partial_values).as(Void*)
      top1_reduce_params[2] = d_top1_id_cur_ptr.as(Void*)
      top1_reduce_params[3] = d_top1_value_cur_ptr.as(Void*)
      top1_reduce_params[4] = box_u32(top1_width).as(Void*)

      run_token = ->(tok : Int32) {
        d_x_cur_ptr.value = @input_device_base.not_nil! + bytesize_f32(tok * @hidden)
        d_normed_cur_ptr.value = d_normed_all + bytesize_f32(tok * @hidden)
        d_logits_cur_ptr.value = d_logits_all + bytesize_f32(tok * @vocab)
        d_top1_id_cur_ptr.value = d_top1_ids + bytesize_i32(tok)
        d_top1_value_cur_ptr.value = d_top1_values + bytesize_f32(tok)
        ML::CUDA.launch!(norm_fn, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, norm_params, "output norm")
        ML::CUDA.launch!(output_fn, vocab_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, output_params, "output logits")
        ML::CUDA.launch!(top1_scan_fn, 1_u32, 1_u32, 1_u32, top1_width, 1_u32, 1_u32, top1_scan_params, "output top1 scan")
        ML::CUDA.launch!(top1_reduce_fn, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, top1_reduce_params, "output top1 reduce")
      }

      profile_run_token = ->(tok : Int32) {
        t_norm = Time.instant
        d_x_cur_ptr.value = @input_device_base.not_nil! + bytesize_f32(tok * @hidden)
        d_normed_cur_ptr.value = d_normed_all + bytesize_f32(tok * @hidden)
        ML::CUDA.launch!(norm_fn, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, norm_params, "output norm")
        ML::CUDA.synchronize!("cuCtxSynchronize(output head norm)")
        @profile_head_norm_ms += (Time.instant - t_norm).total_milliseconds

        t_logits = Time.instant
        d_normed_cur_ptr.value = d_normed_all + bytesize_f32(tok * @hidden)
        d_logits_cur_ptr.value = d_logits_all + bytesize_f32(tok * @vocab)
        ML::CUDA.launch!(output_fn, vocab_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, output_params, "output logits")
        ML::CUDA.synchronize!("cuCtxSynchronize(output head logits)")
        @profile_head_logits_ms += (Time.instant - t_logits).total_milliseconds

        t_top1 = Time.instant
        d_logits_cur_ptr.value = d_logits_all + bytesize_f32(tok * @vocab)
        d_top1_id_cur_ptr.value = d_top1_ids + bytesize_i32(tok)
        d_top1_value_cur_ptr.value = d_top1_values + bytesize_f32(tok)
        ML::CUDA.launch!(top1_scan_fn, 1_u32, 1_u32, 1_u32, top1_width, 1_u32, 1_u32, top1_scan_params, "output top1 scan")
        ML::CUDA.launch!(top1_reduce_fn, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, top1_reduce_params, "output top1 reduce")
        ML::CUDA.synchronize!("cuCtxSynchronize(output head top1)")
        @profile_head_top1_ms += (Time.instant - t_top1).total_milliseconds
      }

      read_outputs = -> {
        if @read_logits
          ML::CUDA.copy_dtoh!(@logits_gpu_all.to_unsafe.as(Void*), d_logits_all, bytesize_f32(@logits_gpu_all.size), "output_logits")
        end
        ML::CUDA.copy_dtoh!(@top1_ids_gpu.to_unsafe.as(Void*), d_top1_ids, bytesize_i32(@top1_ids_gpu.size), "output_top1_ids")
        ML::CUDA.copy_dtoh!(@top1_values_gpu.to_unsafe.as(Void*), d_top1_values, bytesize_f32(@top1_values_gpu.size), "output_top1_values")
      }
      @runner = ResidentSequenceRunner.new(@tokens, upload_weights, reset_sequence, run_token, read_outputs)
      @profile_runner = ResidentSequenceRunner.new(@tokens, upload_weights, reset_sequence, profile_run_token, read_outputs)
    end

    private def runner : ResidentSequenceRunner
      @runner.not_nil!
    end

    private def profile_runner : ResidentSequenceRunner
      @profile_runner.not_nil!
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

    private def bytesize_i32(elements : Int32) : LibC::SizeT
      (elements * sizeof(Int32)).to_u64
    end
  end
end
