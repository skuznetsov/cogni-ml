require "./driver"
require "../gguf/reader"

module ML::CUDA
  # CUDA output-head boundary: output RMSNorm followed by quantized lm_head
  # projection and resident top1. Full logits readback is optional for
  # correctness attribution; the default decode-facing path reads only top1.
  class QwenOutputHeadRunner
    @profile_runner : ResidentSequenceRunner?

    DN_PTX   = {{ read_file("#{__DIR__}/kernels/shared/deltanet_step_probe.ptx") }}
    Q4K_PTX  = {{ read_file("#{__DIR__}/kernels/shared/q4k_gemv_probe.ptx") }}
    Q6K_PTX  = {{ read_file("#{__DIR__}/kernels/shared/q6k_gemv_probe.ptx") }}
    TOP1_PTX = {{ read_file("#{__DIR__}/kernels/qwen35/output_top1_partial_scan_probe.ptx") }}

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
    getter top2_ids_gpu : Array(Int32)
    getter top2_values_gpu : Array(Float32)
    getter logits_device_ptr : DevicePtr
    getter top1_ids_device_ptr : DevicePtr

    def self.from_weights(weights : Weights,
                          tokens : Int32,
                          xs : Array(Float32),
                          eps : Float32,
                          read_logits : Bool = false,
                          read_top2 : Bool = false) : self
      new(tokens, weights.hidden, weights.vocab, xs, weights.norm,
        weights.output_raw, weights.output_type, eps, read_logits, read_top2)
    end

    private def initialize(@tokens : Int32,
                           @hidden : Int32,
                           @vocab : Int32,
                           @xs : Array(Float32),
                           @norm : Array(Float32),
                           @output_raw : Bytes,
                           @output_type : ML::GGUF::TensorType,
                           @eps : Float32,
                           @read_logits : Bool,
                           @read_top2 : Bool)
      raise ArgumentError.new("tokens must be positive") unless @tokens > 0
      raise ArgumentError.new("xs size mismatch") unless @xs.size == @tokens * @hidden
      raise ArgumentError.new("norm size mismatch") unless @norm.size == @hidden

      @modules = [] of CUDAModule
      @buffers = [] of DeviceBuffer
      @param_keepalive = [] of Void*
      @input_device_base = nil.as(DevicePtr?)
      @owned_input_device_ptr = nil.as(DevicePtr?)
      @logits_device_ptr = 0_u64
      @top1_ids_device_ptr = 0_u64
      @logits_gpu_all = Array(Float32).new(@tokens * @vocab, 0.0_f32)
      @top1_ids_gpu = Array(Int32).new(@tokens, 0)
      @top1_values_gpu = Array(Float32).new(@tokens, 0.0_f32)
      @top2_ids_gpu = Array(Int32).new(@tokens, 0)
      @top2_values_gpu = Array(Float32).new(@tokens, 0.0_f32)
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

    def active_tokens=(count : Int32) : Int32
      runner.active_tokens = count
      @profile_runner.try { |profile| profile.active_tokens = count }
      count
    end

    def reset_active_tokens : Nil
      runner.reset_active_tokens
      @profile_runner.try(&.reset_active_tokens)
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

      norm_fn = dn_mod.function("rmsnorm_vec_parallel_probe")
      q4_fn = q4_mod.function("q4_k_gemv_warp4_f32")
      q6_fn = q6_mod.function("q6_k_gemv_warp4_f32")
      q6_tbatch4_fn = q6_mod.function("q6_k_gemv_warp4_f32_tbatch4")
      q6_top1_partial_fn = q6_mod.function("q6_k_gemv_top1_partial_f32")
      top2_scan_fn = top1_mod.function("output_top2_partial_scan_probe")
      top2_reduce_fn = top1_mod.function("output_top2_partial_reduce_probe")
      top2_scan_batched_fn = top1_mod.function("output_top2_partial_scan_batched_probe")
      top2_reduce_batched_fn = top1_mod.function("output_top2_partial_reduce_batched_probe")
      top1_values_reduce_fn = top1_mod.function("output_top1_values_reduce_probe")
      output_fn = @output_type.q4_k? ? q4_fn : q6_fn
      head_q6_tbatch4 = ENV["QWEN_CUDA_HEAD_Q6_TBATCH4_OFF"]? != "1" &&
                        @output_type.q6_k? && @tokens >= 4 && (@tokens % 4 == 0)
      head_top2_batched = ENV["QWEN_CUDA_HEAD_TOP2_BATCHED_OFF"]? != "1" && head_q6_tbatch4
      fused_q6_top1 = @output_type.q6_k? && !@read_logits && !@read_top2 && !head_q6_tbatch4
      vocab_grid = ((@vocab + 3) // 4).to_u32
      top1_width = fused_q6_top1 ? vocab_grid : 256_u32
      logits_elements = fused_q6_top1 ? 1 : @tokens * @vocab

      sizes = [bytesize_f32(@tokens * @hidden), bytesize_f32(@hidden), bytesize_f32(@hidden),
               @output_raw.size.to_u64, bytesize_f32(@tokens * @hidden), bytesize_f32(logits_elements),
               bytesize_i32(@tokens * top1_width.to_i32), bytesize_f32(@tokens * top1_width.to_i32),
               bytesize_i32(@tokens * top1_width.to_i32), bytesize_f32(@tokens * top1_width.to_i32),
               bytesize_i32(@tokens), bytesize_f32(@tokens),
               bytesize_i32(@tokens), bytesize_f32(@tokens)]
      ptrs = sizes.map do |size_bytes|
        buffer = DeviceBuffer.new(size_bytes)
        @buffers << buffer
        buffer.ptr
      end
      d_xs, d_norm_w, d_normed, d_output_w, d_normed_all, d_logits_all, d_partial_ids, d_partial_values, d_partial2_ids, d_partial2_values, d_top1_ids, d_top1_values, d_top2_ids, d_top2_values = ptrs
      @owned_input_device_ptr = d_xs
      @input_device_base = d_xs
      @logits_device_ptr = d_logits_all
      @top1_ids_device_ptr = d_top1_ids

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
      d_x_cur_ptr = box_ptr(d_xs)
      d_normed_cur_ptr = box_ptr(d_normed)
      d_logits_cur_ptr = box_ptr(d_logits_all)
      d_partial_ids_cur_ptr = box_ptr(d_partial_ids)
      d_partial_values_cur_ptr = box_ptr(d_partial_values)
      d_partial2_ids_cur_ptr = box_ptr(d_partial2_ids)
      d_partial2_values_cur_ptr = box_ptr(d_partial2_values)
      d_top1_id_cur_ptr = box_ptr(d_top1_ids)
      d_top1_value_cur_ptr = box_ptr(d_top1_values)
      d_top2_id_cur_ptr = box_ptr(d_top2_ids)
      d_top2_value_cur_ptr = box_ptr(d_top2_values)

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

      q6_top1_partial_params = Pointer(Void*).malloc(6)
      q6_top1_partial_params[0] = box_ptr(d_output_w).as(Void*)
      q6_top1_partial_params[1] = d_normed_cur_ptr.as(Void*)
      q6_top1_partial_params[2] = d_partial_ids_cur_ptr.as(Void*)
      q6_top1_partial_params[3] = d_partial_values_cur_ptr.as(Void*)
      q6_top1_partial_params[4] = box_u32(hidden_u32).as(Void*)
      q6_top1_partial_params[5] = box_u32(vocab_u32).as(Void*)

      top1_reduce_params = Pointer(Void*).malloc(5)
      top1_reduce_params[0] = d_partial_ids_cur_ptr.as(Void*)
      top1_reduce_params[1] = d_partial_values_cur_ptr.as(Void*)
      top1_reduce_params[2] = d_top1_id_cur_ptr.as(Void*)
      top1_reduce_params[3] = d_top1_value_cur_ptr.as(Void*)
      top1_reduce_params[4] = box_u32(top1_width).as(Void*)

      top2_scan_params = Pointer(Void*).malloc(7)
      top2_scan_params[0] = d_logits_cur_ptr.as(Void*)
      top2_scan_params[1] = d_partial_ids_cur_ptr.as(Void*)
      top2_scan_params[2] = d_partial_values_cur_ptr.as(Void*)
      top2_scan_params[3] = d_partial2_ids_cur_ptr.as(Void*)
      top2_scan_params[4] = d_partial2_values_cur_ptr.as(Void*)
      top2_scan_params[5] = box_u32(vocab_u32).as(Void*)
      top2_scan_params[6] = box_u32(top1_width).as(Void*)

      top2_reduce_params = Pointer(Void*).malloc(9)
      top2_reduce_params[0] = d_partial_ids_cur_ptr.as(Void*)
      top2_reduce_params[1] = d_partial_values_cur_ptr.as(Void*)
      top2_reduce_params[2] = d_partial2_ids_cur_ptr.as(Void*)
      top2_reduce_params[3] = d_partial2_values_cur_ptr.as(Void*)
      top2_reduce_params[4] = d_top1_id_cur_ptr.as(Void*)
      top2_reduce_params[5] = d_top1_value_cur_ptr.as(Void*)
      top2_reduce_params[6] = d_top2_id_cur_ptr.as(Void*)
      top2_reduce_params[7] = d_top2_value_cur_ptr.as(Void*)
      top2_reduce_params[8] = box_u32(top1_width).as(Void*)

      run_token = ->(tok : Int32) {
        if head_q6_tbatch4
          if tok % 4 == 0
            4.times do |lane|
              row = tok + lane
              d_x_cur_ptr.value = @input_device_base.not_nil! + bytesize_f32(row * @hidden)
              d_normed_cur_ptr.value = d_normed_all + bytesize_f32(row * @hidden)
              ML::CUDA.launch!(norm_fn, 1_u32, 1_u32, 1_u32, 256_u32, 1_u32, 1_u32, norm_params, "output norm tbatch4")
            end
            d_normed_cur_ptr.value = d_normed_all + bytesize_f32(tok * @hidden)
            d_logits_cur_ptr.value = d_logits_all + bytesize_f32(tok * @vocab)
            ML::CUDA.launch!(q6_tbatch4_fn, vocab_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, output_params, "output logits tbatch4")
            if head_top2_batched
              d_logits_cur_ptr.value = d_logits_all + bytesize_f32(tok * @vocab)
              d_partial_ids_cur_ptr.value = d_partial_ids + bytesize_i32(tok * top1_width.to_i32)
              d_partial_values_cur_ptr.value = d_partial_values + bytesize_f32(tok * top1_width.to_i32)
              d_partial2_ids_cur_ptr.value = d_partial2_ids + bytesize_i32(tok * top1_width.to_i32)
              d_partial2_values_cur_ptr.value = d_partial2_values + bytesize_f32(tok * top1_width.to_i32)
              d_top1_id_cur_ptr.value = d_top1_ids + bytesize_i32(tok)
              d_top1_value_cur_ptr.value = d_top1_values + bytesize_f32(tok)
              d_top2_id_cur_ptr.value = d_top2_ids + bytesize_i32(tok)
              d_top2_value_cur_ptr.value = d_top2_values + bytesize_f32(tok)
              ML::CUDA.launch!(top2_scan_batched_fn, 4_u32, 1_u32, 1_u32, top1_width, 1_u32, 1_u32, top2_scan_params, "output top2 scan batched4")
              ML::CUDA.launch!(top2_reduce_batched_fn, 4_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, top2_reduce_params, "output top2 reduce batched4")
            else
              4.times do |lane|
                row = tok + lane
                d_logits_cur_ptr.value = d_logits_all + bytesize_f32(row * @vocab)
                d_partial_ids_cur_ptr.value = d_partial_ids + bytesize_i32(row * top1_width.to_i32)
                d_partial_values_cur_ptr.value = d_partial_values + bytesize_f32(row * top1_width.to_i32)
                d_partial2_ids_cur_ptr.value = d_partial2_ids + bytesize_i32(row * top1_width.to_i32)
                d_partial2_values_cur_ptr.value = d_partial2_values + bytesize_f32(row * top1_width.to_i32)
                d_top1_id_cur_ptr.value = d_top1_ids + bytesize_i32(row)
                d_top1_value_cur_ptr.value = d_top1_values + bytesize_f32(row)
                d_top2_id_cur_ptr.value = d_top2_ids + bytesize_i32(row)
                d_top2_value_cur_ptr.value = d_top2_values + bytesize_f32(row)
                ML::CUDA.launch!(top2_scan_fn, 1_u32, 1_u32, 1_u32, top1_width, 1_u32, 1_u32, top2_scan_params, "output top2 scan tbatch4")
                ML::CUDA.launch!(top2_reduce_fn, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, top2_reduce_params, "output top2 reduce tbatch4")
              end
            end
          end
        else
          d_x_cur_ptr.value = @input_device_base.not_nil! + bytesize_f32(tok * @hidden)
          d_normed_cur_ptr.value = d_normed_all + bytesize_f32(tok * @hidden)
          d_logits_cur_ptr.value = d_logits_all + bytesize_f32(tok * @vocab)
          d_partial_ids_cur_ptr.value = d_partial_ids + bytesize_i32(tok * top1_width.to_i32)
          d_partial_values_cur_ptr.value = d_partial_values + bytesize_f32(tok * top1_width.to_i32)
          d_partial2_ids_cur_ptr.value = d_partial2_ids + bytesize_i32(tok * top1_width.to_i32)
          d_partial2_values_cur_ptr.value = d_partial2_values + bytesize_f32(tok * top1_width.to_i32)
          d_top1_id_cur_ptr.value = d_top1_ids + bytesize_i32(tok)
          d_top1_value_cur_ptr.value = d_top1_values + bytesize_f32(tok)
          d_top2_id_cur_ptr.value = d_top2_ids + bytesize_i32(tok)
          d_top2_value_cur_ptr.value = d_top2_values + bytesize_f32(tok)
          ML::CUDA.launch!(norm_fn, 1_u32, 1_u32, 1_u32, 256_u32, 1_u32, 1_u32, norm_params, "output norm")
          if fused_q6_top1
            ML::CUDA.launch!(q6_top1_partial_fn, vocab_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, q6_top1_partial_params, "output q6 top1 partial")
            ML::CUDA.launch!(top1_values_reduce_fn, 1_u32, 1_u32, 1_u32, 256_u32, 1_u32, 1_u32, top1_reduce_params, "output top1 values reduce")
          else
            ML::CUDA.launch!(output_fn, vocab_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, output_params, "output logits")
            ML::CUDA.launch!(top2_scan_fn, 1_u32, 1_u32, 1_u32, top1_width, 1_u32, 1_u32, top2_scan_params, "output top2 scan")
            ML::CUDA.launch!(top2_reduce_fn, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, top2_reduce_params, "output top2 reduce")
          end
        end
      }

      profile_run_token = ->(tok : Int32) {
        if head_q6_tbatch4
          if tok % 4 == 0
            t_norm = Time.instant
            4.times do |lane|
              row = tok + lane
              d_x_cur_ptr.value = @input_device_base.not_nil! + bytesize_f32(row * @hidden)
              d_normed_cur_ptr.value = d_normed_all + bytesize_f32(row * @hidden)
              ML::CUDA.launch!(norm_fn, 1_u32, 1_u32, 1_u32, 256_u32, 1_u32, 1_u32, norm_params, "output norm tbatch4")
            end
            ML::CUDA.synchronize!("cuCtxSynchronize(output head norm tbatch4)")
            @profile_head_norm_ms += (Time.instant - t_norm).total_milliseconds

            t_logits = Time.instant
            d_normed_cur_ptr.value = d_normed_all + bytesize_f32(tok * @hidden)
            d_logits_cur_ptr.value = d_logits_all + bytesize_f32(tok * @vocab)
            ML::CUDA.launch!(q6_tbatch4_fn, vocab_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, output_params, "output logits tbatch4")
            ML::CUDA.synchronize!("cuCtxSynchronize(output head logits tbatch4)")
            @profile_head_logits_ms += (Time.instant - t_logits).total_milliseconds

            t_top1 = Time.instant
            if head_top2_batched
              d_logits_cur_ptr.value = d_logits_all + bytesize_f32(tok * @vocab)
              d_partial_ids_cur_ptr.value = d_partial_ids + bytesize_i32(tok * top1_width.to_i32)
              d_partial_values_cur_ptr.value = d_partial_values + bytesize_f32(tok * top1_width.to_i32)
              d_partial2_ids_cur_ptr.value = d_partial2_ids + bytesize_i32(tok * top1_width.to_i32)
              d_partial2_values_cur_ptr.value = d_partial2_values + bytesize_f32(tok * top1_width.to_i32)
              d_top1_id_cur_ptr.value = d_top1_ids + bytesize_i32(tok)
              d_top1_value_cur_ptr.value = d_top1_values + bytesize_f32(tok)
              d_top2_id_cur_ptr.value = d_top2_ids + bytesize_i32(tok)
              d_top2_value_cur_ptr.value = d_top2_values + bytesize_f32(tok)
              ML::CUDA.launch!(top2_scan_batched_fn, 4_u32, 1_u32, 1_u32, top1_width, 1_u32, 1_u32, top2_scan_params, "output top2 scan batched4")
              ML::CUDA.launch!(top2_reduce_batched_fn, 4_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, top2_reduce_params, "output top2 reduce batched4")
            else
              4.times do |lane|
                row = tok + lane
                d_logits_cur_ptr.value = d_logits_all + bytesize_f32(row * @vocab)
                d_partial_ids_cur_ptr.value = d_partial_ids + bytesize_i32(row * top1_width.to_i32)
                d_partial_values_cur_ptr.value = d_partial_values + bytesize_f32(row * top1_width.to_i32)
                d_partial2_ids_cur_ptr.value = d_partial2_ids + bytesize_i32(row * top1_width.to_i32)
                d_partial2_values_cur_ptr.value = d_partial2_values + bytesize_f32(row * top1_width.to_i32)
                d_top1_id_cur_ptr.value = d_top1_ids + bytesize_i32(row)
                d_top1_value_cur_ptr.value = d_top1_values + bytesize_f32(row)
                d_top2_id_cur_ptr.value = d_top2_ids + bytesize_i32(row)
                d_top2_value_cur_ptr.value = d_top2_values + bytesize_f32(row)
                ML::CUDA.launch!(top2_scan_fn, 1_u32, 1_u32, 1_u32, top1_width, 1_u32, 1_u32, top2_scan_params, "output top2 scan tbatch4")
                ML::CUDA.launch!(top2_reduce_fn, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, top2_reduce_params, "output top2 reduce tbatch4")
              end
            end
            ML::CUDA.synchronize!("cuCtxSynchronize(output head top1 tbatch4)")
            @profile_head_top1_ms += (Time.instant - t_top1).total_milliseconds
          end
        else
          t_norm = Time.instant
          d_x_cur_ptr.value = @input_device_base.not_nil! + bytesize_f32(tok * @hidden)
          d_normed_cur_ptr.value = d_normed_all + bytesize_f32(tok * @hidden)
          ML::CUDA.launch!(norm_fn, 1_u32, 1_u32, 1_u32, 256_u32, 1_u32, 1_u32, norm_params, "output norm")
          ML::CUDA.synchronize!("cuCtxSynchronize(output head norm)")
          @profile_head_norm_ms += (Time.instant - t_norm).total_milliseconds

          t_logits = Time.instant
          d_normed_cur_ptr.value = d_normed_all + bytesize_f32(tok * @hidden)
          d_logits_cur_ptr.value = d_logits_all + bytesize_f32(tok * @vocab)
          d_partial_ids_cur_ptr.value = d_partial_ids + bytesize_i32(tok * top1_width.to_i32)
          d_partial_values_cur_ptr.value = d_partial_values + bytesize_f32(tok * top1_width.to_i32)
          d_partial2_ids_cur_ptr.value = d_partial2_ids + bytesize_i32(tok * top1_width.to_i32)
          d_partial2_values_cur_ptr.value = d_partial2_values + bytesize_f32(tok * top1_width.to_i32)
          if fused_q6_top1
            ML::CUDA.launch!(q6_top1_partial_fn, vocab_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, q6_top1_partial_params, "output q6 top1 partial")
          else
            ML::CUDA.launch!(output_fn, vocab_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, output_params, "output logits")
          end
          ML::CUDA.synchronize!("cuCtxSynchronize(output head logits)")
          @profile_head_logits_ms += (Time.instant - t_logits).total_milliseconds

          t_top1 = Time.instant
          d_logits_cur_ptr.value = d_logits_all + bytesize_f32(tok * @vocab)
          d_partial_ids_cur_ptr.value = d_partial_ids + bytesize_i32(tok * top1_width.to_i32)
          d_partial_values_cur_ptr.value = d_partial_values + bytesize_f32(tok * top1_width.to_i32)
          d_partial2_ids_cur_ptr.value = d_partial2_ids + bytesize_i32(tok * top1_width.to_i32)
          d_partial2_values_cur_ptr.value = d_partial2_values + bytesize_f32(tok * top1_width.to_i32)
          d_top1_id_cur_ptr.value = d_top1_ids + bytesize_i32(tok)
          d_top1_value_cur_ptr.value = d_top1_values + bytesize_f32(tok)
          d_top2_id_cur_ptr.value = d_top2_ids + bytesize_i32(tok)
          d_top2_value_cur_ptr.value = d_top2_values + bytesize_f32(tok)
          if fused_q6_top1
            ML::CUDA.launch!(top1_values_reduce_fn, 1_u32, 1_u32, 1_u32, 256_u32, 1_u32, 1_u32, top1_reduce_params, "output top1 values reduce")
          else
            ML::CUDA.launch!(top2_scan_fn, 1_u32, 1_u32, 1_u32, top1_width, 1_u32, 1_u32, top2_scan_params, "output top2 scan")
            ML::CUDA.launch!(top2_reduce_fn, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, top2_reduce_params, "output top2 reduce")
          end
          ML::CUDA.synchronize!("cuCtxSynchronize(output head top1)")
          @profile_head_top1_ms += (Time.instant - t_top1).total_milliseconds
        end
      }

      read_outputs = -> {
        if @read_logits
          ML::CUDA.copy_dtoh!(@logits_gpu_all.to_unsafe.as(Void*), d_logits_all, bytesize_f32(@logits_gpu_all.size), "output_logits")
        end
        ML::CUDA.copy_dtoh!(@top1_ids_gpu.to_unsafe.as(Void*), d_top1_ids, bytesize_i32(@top1_ids_gpu.size), "output_top1_ids")
        ML::CUDA.copy_dtoh!(@top1_values_gpu.to_unsafe.as(Void*), d_top1_values, bytesize_f32(@top1_values_gpu.size), "output_top1_values")
        if @read_logits || @read_top2
          ML::CUDA.copy_dtoh!(@top2_ids_gpu.to_unsafe.as(Void*), d_top2_ids, bytesize_i32(@top2_ids_gpu.size), "output_top2_ids")
          ML::CUDA.copy_dtoh!(@top2_values_gpu.to_unsafe.as(Void*), d_top2_values, bytesize_f32(@top2_values_gpu.size), "output_top2_values")
        end
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
