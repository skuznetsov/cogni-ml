require "./driver"
require "../gguf/reader"

module ML::CUDA
  # Post-projection boundary for full-attention CUDA work:
  # split Q/gate, RMSNorm Q/K, apply precomputed RoPE tables, and append K/V
  # rows to the layer KV cache.
  class QwenFullAttnKVRunner
    @profile_runner : ResidentSequenceRunner?
    @increment_start_pos : Proc(Nil)?

    FULL_ATTN_POST_PTX = {{ read_file("src/ml/cuda/kernels/fullattn_post_probe.ptx") }}
    DN_PTX             = {{ read_file("src/ml/cuda/kernels/deltanet_step_probe.ptx") }}
    Q4K_PTX            = {{ read_file("src/ml/cuda/kernels/q4k_gemv_probe.ptx") }}
    Q6K_PTX            = {{ read_file("src/ml/cuda/kernels/q6k_gemv_probe.ptx") }}

    class Weights
      getter attn_norm : Array(Float32)
      getter q_norm : Array(Float32)
      getter k_norm : Array(Float32)
      getter output_raw : Bytes
      getter output_type : ML::GGUF::TensorType
      getter output_in_dim : Int32
      getter output_out_dim : Int32
      getter post_norm : Array(Float32)
      getter ffn_gate_raw : Bytes
      getter ffn_up_raw : Bytes
      getter ffn_down_raw : Bytes
      getter ffn_down_type : ML::GGUF::TensorType
      getter ffn_dim : Int32

      def self.load(gguf : ML::GGUF::GGUFFile, layer : Int32) : self
        prefix = "blk.#{layer}"
        attn_norm_info = gguf.tensor("#{prefix}.attn_norm.weight") || raise "missing #{prefix}.attn_norm.weight"
        q_norm_info = gguf.tensor("#{prefix}.attn_q_norm.weight") || raise "missing #{prefix}.attn_q_norm.weight"
        k_norm_info = gguf.tensor("#{prefix}.attn_k_norm.weight") || raise "missing #{prefix}.attn_k_norm.weight"
        output_info = gguf.tensor("#{prefix}.attn_output.weight") || raise "missing #{prefix}.attn_output.weight"
        post_norm_info = gguf.tensor("#{prefix}.post_attention_norm.weight") || raise "missing #{prefix}.post_attention_norm.weight"
        ffn_gate_info = gguf.tensor("#{prefix}.ffn_gate.weight") || raise "missing #{prefix}.ffn_gate.weight"
        ffn_up_info = gguf.tensor("#{prefix}.ffn_up.weight") || raise "missing #{prefix}.ffn_up.weight"
        ffn_down_info = gguf.tensor("#{prefix}.ffn_down.weight") || raise "missing #{prefix}.ffn_down.weight"
        raise "expected Q4_K/Q6_K attn_output" unless output_info.type.q4_k? || output_info.type.q6_k?
        raise "expected Q4_K ffn gate/up" unless ffn_gate_info.type.q4_k? && ffn_up_info.type.q4_k?
        raise "expected Q4_K/Q6_K ffn_down" unless ffn_down_info.type.q4_k? || ffn_down_info.type.q6_k?
        hidden = output_info.dims[1].to_i32
        ffn_dim = ffn_gate_info.dims[1].to_i32
        raise "ffn shape mismatch" unless ffn_gate_info.dims[0].to_i32 == hidden &&
                                          ffn_up_info.dims[0].to_i32 == hidden &&
                                          ffn_up_info.dims[1].to_i32 == ffn_dim &&
                                          ffn_down_info.dims[0].to_i32 == ffn_dim &&
                                          ffn_down_info.dims[1].to_i32 == hidden
        new(gguf.read_tensor_f32(attn_norm_info),
          gguf.read_tensor_f32(q_norm_info), gguf.read_tensor_f32(k_norm_info),
          gguf.read_tensor_raw(output_info), output_info.type,
          output_info.dims[0].to_i32, output_info.dims[1].to_i32,
          gguf.read_tensor_f32(post_norm_info),
          gguf.read_tensor_raw(ffn_gate_info), gguf.read_tensor_raw(ffn_up_info),
          gguf.read_tensor_raw(ffn_down_info), ffn_down_info.type, ffn_dim)
      end

      def initialize(@attn_norm : Array(Float32),
                     @q_norm : Array(Float32), @k_norm : Array(Float32),
                     @output_raw : Bytes, @output_type : ML::GGUF::TensorType,
                     @output_in_dim : Int32, @output_out_dim : Int32,
                     @post_norm : Array(Float32),
                     @ffn_gate_raw : Bytes, @ffn_up_raw : Bytes,
                     @ffn_down_raw : Bytes, @ffn_down_type : ML::GGUF::TensorType,
                     @ffn_dim : Int32)
      end
    end

    getter q_gpu_all : Array(Float32)
    getter gate_gpu_all : Array(Float32)
    getter k_gpu_all : Array(Float32)
    getter attn_gpu_all : Array(Float32)
    getter proj_gpu_all : Array(Float32)
    getter final_gpu_all : Array(Float32)
    getter output_device_ptr : DevicePtr
    getter k_cache_device_ptr : DevicePtr
    getter v_cache_device_ptr : DevicePtr
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
                   residual_input : Array(Float32),
                   cos_table : Array(Float32),
                   sin_table : Array(Float32))
      raise ArgumentError.new("tokens must be positive") unless @tokens > 0
      raise ArgumentError.new("max_seq too small") unless @max_seq >= @start_pos + @tokens
      raise ArgumentError.new("rope_dim must be even") unless @rope_dim.even?
      raise ArgumentError.new("q_norm size mismatch") unless weights.q_norm.size == @head_dim
      raise ArgumentError.new("k_norm size mismatch") unless weights.k_norm.size == @head_dim
      raise ArgumentError.new("attn_output input mismatch") unless weights.output_in_dim == @n_head * @head_dim
      raise ArgumentError.new("residual input size mismatch") unless residual_input.size == @tokens * weights.output_out_dim
      half = @rope_dim // 2
      min_rope_table = (@start_pos + @tokens) * half
      raise ArgumentError.new("cos/sin table size mismatch") unless cos_table.size >= min_rope_table && sin_table.size >= min_rope_table

      @attn_norm = weights.attn_norm
      @q_norm = weights.q_norm
      @k_norm = weights.k_norm
      @output_raw = weights.output_raw
      @output_type = weights.output_type
      @output_in_dim = weights.output_in_dim
      @output_out_dim = weights.output_out_dim
      @post_norm = weights.post_norm
      @ffn_gate_raw = weights.ffn_gate_raw
      @ffn_up_raw = weights.ffn_up_raw
      @ffn_down_raw = weights.ffn_down_raw
      @ffn_down_type = weights.ffn_down_type
      @ffn_dim = weights.ffn_dim
      @residual_input = residual_input
      @cos_table = cos_table
      @sin_table = sin_table
      @q_dim = @n_head * @head_dim
      @kv_dim = @n_head_kv * @head_dim
      @q_gpu_all = Array(Float32).new(@tokens * @q_dim, 0.0_f32)
      @gate_gpu_all = Array(Float32).new(@tokens * @q_dim, 0.0_f32)
      @k_gpu_all = Array(Float32).new(@tokens * @kv_dim, 0.0_f32)
      @attn_gpu_all = Array(Float32).new(@tokens * @q_dim, 0.0_f32)
      @proj_gpu_all = Array(Float32).new(@tokens * @output_out_dim, 0.0_f32)
      @final_gpu_all = Array(Float32).new(@tokens * @output_out_dim, 0.0_f32)
      @k_cache_gpu = Array(Float32).new(@max_seq * @kv_dim, 0.0_f32)
      @v_cache_gpu = Array(Float32).new(@max_seq * @kv_dim, 0.0_f32)
      @modules = [] of CUDAModule
      @buffers = [] of DeviceBuffer
      @param_keepalive = [] of Void*
      @residual_device_base = nil.as(DevicePtr?)
      @owned_residual_device_ptr = nil.as(DevicePtr?)
      @output_device_ptr = 0_u64
      @k_cache_device_ptr = 0_u64
      @v_cache_device_ptr = 0_u64
      @cos_device_ptr = 0_u64
      @sin_device_ptr = 0_u64
      @start_pos_device_ptr = 0_u64
      @start_pos_box = nil.as(Pointer(UInt32)?)
      @profile_qk_rope_ms = 0.0
      @profile_attn_decode_ms = 0.0
      @profile_out_proj_ms = 0.0
      @profile_add_rms_ms = 0.0
      @profile_ffn_gate_ms = 0.0
      @profile_ffn_up_ms = 0.0
      @profile_swiglu_ms = 0.0
      @profile_ffn_down_ms = 0.0
      @profile_final_add_ms = 0.0
      @profile_override_detail = false
      @active_tokens = @tokens
      @closed = false

      build_runner
    end

    def upload_constants : Nil
      runner.upload_weights
    end

    def reset_sequence : Nil
      runner.reset_sequence
    end

    def run_sequence : Nil
      runner.reset_sequence
      runner.run_sequence
    end

    def active_tokens=(count : Int32) : Int32
      @active_tokens = count
      runner.active_tokens = count
      @profile_runner.try { |profile| profile.active_tokens = count }
      count
    end

    def reset_active_tokens : Nil
      @active_tokens = @tokens
      runner.reset_active_tokens
      @profile_runner.try(&.reset_active_tokens)
    end

    def run_sequence_profiled(phase_lines : Array(String), prefix : String) : Nil
      runner.reset_sequence
      @profile_qk_rope_ms = 0.0
      @profile_attn_decode_ms = 0.0
      @profile_out_proj_ms = 0.0
      @profile_add_rms_ms = 0.0
      @profile_ffn_gate_ms = 0.0
      @profile_ffn_up_ms = 0.0
      @profile_swiglu_ms = 0.0
      @profile_ffn_down_ms = 0.0
      @profile_final_add_ms = 0.0
      t_total = Time.instant
      if ENV["QWEN_CUDA_FULL_ATTN_BATCHED_FFN_OFF"]? != "1" && @tokens > 1
        batched_norms_profile = ENV["QWEN_CUDA_FULL_ATTN_BATCHED_NORMS_OFF"]? != "1"
        @profile_override_detail = true
        begin
          runner.run_sequence
          ML::CUDA.synchronize!("cuCtxSynchronize(full attention batched tail profile)")
        ensure
          @profile_override_detail = false
        end
        phase_lines << "#{prefix}_profile_route=#{batched_norms_profile ? "batched_tail_norm" : "batched_tail"}"
        phase_lines << "#{prefix}_profile_detail=override_components"
        phase_lines << "#{prefix}_qk_rope_ms=#{@profile_qk_rope_ms.round(3)}"
        phase_lines << "#{prefix}_attn_decode_ms=#{@profile_attn_decode_ms.round(3)}"
        phase_lines << "#{prefix}_out_proj_ms=#{@profile_out_proj_ms.round(3)}"
        phase_lines << "#{prefix}_add_rms_ms=#{@profile_add_rms_ms.round(3)}"
        phase_lines << "#{prefix}_ffn_gate_ms=#{@profile_ffn_gate_ms.round(3)}"
        phase_lines << "#{prefix}_ffn_up_ms=#{@profile_ffn_up_ms.round(3)}"
        phase_lines << "#{prefix}_swiglu_ms=#{@profile_swiglu_ms.round(3)}"
        phase_lines << "#{prefix}_ffn_down_ms=#{@profile_ffn_down_ms.round(3)}"
        phase_lines << "#{prefix}_final_add_ms=#{@profile_final_add_ms.round(3)}"
        phase_lines << "#{prefix}_profiled_ms=#{((Time.instant - t_total).total_milliseconds).round(3)}"
        return
      end

      phase_lines << "#{prefix}_profile_route=per_token"
      phase_lines << "#{prefix}_profile_detail=detailed"
      profile_runner.run_sequence
      phase_lines << "#{prefix}_qk_rope_ms=#{@profile_qk_rope_ms.round(3)}"
      phase_lines << "#{prefix}_attn_decode_ms=#{@profile_attn_decode_ms.round(3)}"
      phase_lines << "#{prefix}_out_proj_ms=#{@profile_out_proj_ms.round(3)}"
      phase_lines << "#{prefix}_add_rms_ms=#{@profile_add_rms_ms.round(3)}"
      phase_lines << "#{prefix}_ffn_gate_ms=#{@profile_ffn_gate_ms.round(3)}"
      phase_lines << "#{prefix}_ffn_up_ms=#{@profile_ffn_up_ms.round(3)}"
      phase_lines << "#{prefix}_swiglu_ms=#{@profile_swiglu_ms.round(3)}"
      phase_lines << "#{prefix}_ffn_down_ms=#{@profile_ffn_down_ms.round(3)}"
      phase_lines << "#{prefix}_final_add_ms=#{@profile_final_add_ms.round(3)}"
      phase_lines << "#{prefix}_profiled_ms=#{((Time.instant - t_total).total_milliseconds).round(3)}"
    end

    def replace_residual_input(residual_input : Array(Float32)) : Nil
      raise ArgumentError.new("residual input size mismatch") unless residual_input.size == @tokens * @output_out_dim

      @residual_input = residual_input
      @residual_device_base = @owned_residual_device_ptr
    end

    def use_device_residual_input(ptr : DevicePtr) : Nil
      raise ArgumentError.new("device residual pointer must be non-zero") if ptr == 0_u64

      @residual_device_base = ptr
    end

    def kv_cache_bytesize : LibC::SizeT
      bytesize_f32(@max_seq * @kv_dim)
    end

    def kv_cache_bytesize_for_tokens(tokens : Int32) : LibC::SizeT
      raise ArgumentError.new("kv tokens must be non-negative") unless tokens >= 0
      raise ArgumentError.new("kv tokens exceed max_seq") unless tokens <= @max_seq

      bytesize_f32(tokens * @kv_dim)
    end

    def update_decode_position(start_pos : Int32, cos_table : Array(Float32), sin_table : Array(Float32)) : Nil
      raise ArgumentError.new("start_pos must be non-negative") unless start_pos >= 0
      half = @rope_dim // 2
      min_rope_table = (start_pos + @active_tokens) * half
      raise ArgumentError.new("cos/sin table size mismatch") unless cos_table.size >= min_rope_table && sin_table.size >= min_rope_table

      @start_pos = start_pos
      @cos_table = cos_table
      @sin_table = sin_table
      @start_pos_box.not_nil!.value = start_pos.to_u32
      ML::CUDA.copy_htod!(@start_pos_device_ptr, @start_pos_box.not_nil!.as(Void*), 4_u64, "start_pos")
      ML::CUDA.copy_htod!(@cos_device_ptr, @cos_table.to_unsafe.as(Void*), bytesize_f32(@cos_table.size), "cos")
      ML::CUDA.copy_htod!(@sin_device_ptr, @sin_table.to_unsafe.as(Void*), bytesize_f32(@sin_table.size), "sin")
    end

    def update_decode_position(start_pos : Int32) : Nil
      raise ArgumentError.new("start_pos must be non-negative") unless start_pos >= 0
      half = @rope_dim // 2
      min_rope_table = (start_pos + @active_tokens) * half
      raise ArgumentError.new("resident cos/sin table too small") unless @cos_table.size >= min_rope_table && @sin_table.size >= min_rope_table

      @start_pos = start_pos
      @start_pos_box.not_nil!.value = start_pos.to_u32
      ML::CUDA.copy_htod!(@start_pos_device_ptr, @start_pos_box.not_nil!.as(Void*), 4_u64, "start_pos")
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
      dn_mod = CUDAModule.load(DN_PTX, "delta")
      q4_mod = CUDAModule.load(Q4K_PTX, "q4")
      q6_mod = CUDAModule.load(Q6K_PTX, "q6")
      @modules.concat([mod, dn_mod, q4_mod, q6_mod])
      q_fn = mod.function("full_attn_q_split_norm_rope_probe")
      increment_u32_fn = mod.function("increment_u32_probe")
      k_fn = mod.function("full_attn_k_norm_rope_cache_probe")
      attn_fn = mod.function("full_attn_decode_cache_probe")
      attn_parallel_fn = mod.function("full_attn_decode_cache_parallel_probe")
      add_rmsnorm_fn = dn_mod.function("add_rmsnorm_vec_parallel_probe")
      add_rmsnorm_batched_fn = dn_mod.function("add_rmsnorm_vec_parallel_batched_probe")
      swiglu_fn = dn_mod.function("swiglu_probe")
      q4_fn = q4_mod.function("q4_k_gemv_warp4_f32")
      q4_add_fn = q4_mod.function("q4_k_gemv_add_warp4_f32")
      q4_batched_fn = q4_mod.function("q4_k_gemv_warp4_f32_batched")
      q4_tbatch4_fn = q4_mod.function("q4_k_gemv_warp4_f32_tbatch4")
      q4_add_batched_fn = q4_mod.function("q4_k_gemv_add_warp4_f32_batched")
      q4_add_tbatch4_fn = q4_mod.function("q4_k_gemv_add_warp4_f32_tbatch4")
      q6_fn = q6_mod.function("q6_k_gemv_warp4_f32")
      q6_add_fn = q6_mod.function("q6_k_gemv_add_warp4_f32")
      q6_batched_fn = q6_mod.function("q6_k_gemv_warp4_f32_batched")
      q6_tbatch4_fn = q6_mod.function("q6_k_gemv_warp4_f32_tbatch4")
      q6_add_batched_fn = q6_mod.function("q6_k_gemv_add_warp4_f32_batched")
      q6_add_tbatch4_fn = q6_mod.function("q6_k_gemv_add_warp4_f32_tbatch4")
      out_proj_fn = @output_type.q4_k? ? q4_fn : q6_fn
      out_proj_batched_fn = @output_type.q4_k? ? q4_batched_fn : q6_batched_fn
      out_proj_tbatch4_fn = @output_type.q4_k? ? q4_tbatch4_fn : q6_tbatch4_fn
      ffn_down_add_fn = @ffn_down_type.q4_k? ? q4_add_fn : q6_add_fn
      ffn_down_add_batched_fn = @ffn_down_type.q4_k? ? q4_add_batched_fn : q6_add_batched_fn
      use_parallel_attn = ENV["QWEN_CUDA_FULL_ATTN_PARALLEL_OFF"]? != "1"
      attn_decode_fn = use_parallel_attn ? attn_parallel_fn : attn_fn
      attn_decode_block = use_parallel_attn ? 256_u32 : 1_u32
      use_batched_tail = ENV["QWEN_CUDA_FULL_ATTN_BATCHED_FFN_OFF"]? != "1" && @tokens > 1
      use_batched_norms = ENV["QWEN_CUDA_FULL_ATTN_BATCHED_NORMS_OFF"]? != "1" && use_batched_tail
      use_q4_tbatch4 = ENV["QWEN_CUDA_Q4_TBATCH4_OFF"]? != "1" && @tokens >= 4 && (@tokens % 4 == 0)
      use_q4_down_add_tbatch4 = ENV["QWEN_CUDA_Q4_DOWN_ADD_TBATCH4_OFF"]? != "1" && @ffn_down_type.q4_k? && use_q4_tbatch4
      use_q6_tbatch4 = ENV["QWEN_CUDA_Q6_TBATCH4_OFF"]? != "1" && @ffn_down_type.q6_k? && @tokens >= 4 && (@tokens % 4 == 0)
      use_attn_output_tbatch4 = ENV["QWEN_CUDA_FULL_ATTN_OUTPUT_TBATCH4_OFF"]? != "1" && @tokens >= 4 && (@tokens % 4 == 0)
      tail_rows = use_batched_tail ? @tokens : 1

      sizes = [bytesize_f32(@q_norm.size), bytesize_f32(@k_norm.size),
               bytesize_f32(@cos_table.size), bytesize_f32(@sin_table.size),
               4_u64,
               bytesize_f32(@tokens * @q_dim), bytesize_f32(@tokens * @q_dim),
               bytesize_f32(@tokens * @kv_dim), bytesize_f32(@max_seq * @kv_dim),
               bytesize_f32(@max_seq * @kv_dim), bytesize_f32(@tokens * @n_head * @max_seq),
               bytesize_f32(@tokens * @q_dim), @output_raw.size.to_u64,
               bytesize_f32(@tokens * @output_out_dim),
               bytesize_f32(@tokens * @output_out_dim), bytesize_f32(@post_norm.size),
               bytesize_f32(tail_rows * @output_out_dim), bytesize_f32(tail_rows * @output_out_dim),
               @ffn_gate_raw.size.to_u64, @ffn_up_raw.size.to_u64, @ffn_down_raw.size.to_u64,
               bytesize_f32(tail_rows * @ffn_dim), bytesize_f32(tail_rows * @ffn_dim), bytesize_f32(tail_rows * @ffn_dim),
               bytesize_f32(@output_out_dim), bytesize_f32(@tokens * @output_out_dim)]
      ptrs = sizes.map do |size_bytes|
        buffer = DeviceBuffer.new(size_bytes)
        @buffers << buffer
        buffer.ptr
      end
      d_q_norm, d_k_norm, d_cos, d_sin, d_start_pos, d_q_out, d_gate_out, d_k_out, d_k_cache, d_v_cache, d_scores, d_attn_out, d_output_w, d_proj_out, d_residual_input, d_post_norm, d_residual, d_cur2, d_ffn_gate_w, d_ffn_up_w, d_ffn_down_w, d_ffn_gate, d_ffn_up, d_ffn_comb, d_ffn_out, d_final_all = ptrs
      @owned_residual_device_ptr = d_residual_input
      @residual_device_base = d_residual_input
      @output_device_ptr = d_final_all
      @k_cache_device_ptr = d_k_cache
      @v_cache_device_ptr = d_v_cache
      @cos_device_ptr = d_cos
      @sin_device_ptr = d_sin
      @start_pos_device_ptr = d_start_pos

      upload_constants = -> {
        ML::CUDA.copy_htod!(d_q_norm, @q_norm.to_unsafe.as(Void*), bytesize_f32(@q_norm.size), "q_norm")
        ML::CUDA.copy_htod!(d_k_norm, @k_norm.to_unsafe.as(Void*), bytesize_f32(@k_norm.size), "k_norm")
        ML::CUDA.copy_htod!(d_cos, @cos_table.to_unsafe.as(Void*), bytesize_f32(@cos_table.size), "cos")
        ML::CUDA.copy_htod!(d_sin, @sin_table.to_unsafe.as(Void*), bytesize_f32(@sin_table.size), "sin")
        ML::CUDA.copy_htod!(d_start_pos, @start_pos_box.not_nil!.as(Void*), 4_u64, "start_pos")
        ML::CUDA.copy_htod!(d_output_w, @output_raw.to_unsafe.as(Void*), @output_raw.size.to_u64, "attn_output_w")
        ML::CUDA.copy_htod!(d_post_norm, @post_norm.to_unsafe.as(Void*), bytesize_f32(@post_norm.size), "post_norm")
        ML::CUDA.copy_htod!(d_ffn_gate_w, @ffn_gate_raw.to_unsafe.as(Void*), @ffn_gate_raw.size.to_u64, "ffn_gate_w")
        ML::CUDA.copy_htod!(d_ffn_up_w, @ffn_up_raw.to_unsafe.as(Void*), @ffn_up_raw.size.to_u64, "ffn_up_w")
        ML::CUDA.copy_htod!(d_ffn_down_w, @ffn_down_raw.to_unsafe.as(Void*), @ffn_down_raw.size.to_u64, "ffn_down_w")
      }

      reset_sequence = -> {
        # Projection outputs are already device-resident inputs. KV cache rows
        # written by this probe are overwritten before comparison.
        if @residual_device_base == d_residual_input
          ML::CUDA.copy_htod!(d_residual_input, @residual_input.to_unsafe.as(Void*), bytesize_f32(@residual_input.size), "residual_input")
        end
      }

      n_head_u32 = @n_head.to_u32
      n_head_kv_u32 = @n_head_kv.to_u32
      head_dim_u32 = @head_dim.to_u32
      rope_dim_u32 = @rope_dim.to_u32
      start_pos_u32 = @start_pos.to_u32
      @start_pos_box = box_u32(start_pos_u32)
      max_seq_u32 = @max_seq.to_u32
      heads_per_group_u32 = (@n_head // @n_head_kv).to_u32
      scale = (1.0_f64 / Math.sqrt(@head_dim.to_f64)).to_f32
      output_in_dim_u32 = @output_in_dim.to_u32
      output_out_dim_u32 = @output_out_dim.to_u32
      ffn_dim_u32 = @ffn_dim.to_u32
      ffn_dim_all_u32 = (@tokens * @ffn_dim).to_u32
      output_grid = ((@output_out_dim + 3) // 4).to_u32
      ffn_grid = ((@ffn_dim + 3) // 4).to_u32
      swiglu_grid = ((@ffn_dim + 127) // 128).to_u32
      swiglu_grid_all = (((@tokens * @ffn_dim) + 127) // 128).to_u32

      q_params = Pointer(Void*).malloc(11)
      q_params[0] = box_ptr(@q_full_device_ptr).as(Void*)
      q_params[1] = box_ptr(d_q_norm).as(Void*)
      q_params[2] = box_ptr(d_q_out).as(Void*)
      q_params[3] = box_ptr(d_gate_out).as(Void*)
      q_params[4] = box_ptr(d_cos).as(Void*)
      q_params[5] = box_ptr(d_sin).as(Void*)
      q_params[6] = box_ptr(d_start_pos).as(Void*)
      q_params[7] = box_u32(n_head_u32).as(Void*)
      q_params[8] = box_u32(head_dim_u32).as(Void*)
      q_params[9] = box_u32(rope_dim_u32).as(Void*)
      q_params[10] = box_f32(@eps).as(Void*)

      increment_params = Pointer(Void*).malloc(1)
      increment_params[0] = box_ptr(d_start_pos).as(Void*)
      @increment_start_pos = -> {
        ML::CUDA.launch!(increment_u32_fn, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32,
          increment_params, "increment full-attn start_pos")
      }

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
      k_params[11] = box_ptr(d_start_pos).as(Void*)
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
      attn_params[10] = box_ptr(d_start_pos).as(Void*)
      attn_params[11] = box_u32(max_seq_u32).as(Void*)
      attn_params[12] = box_f32(scale).as(Void*)

      d_attn_cur_ptr = box_ptr(d_attn_out)
      d_proj_cur_ptr = box_ptr(d_proj_out)
      out_proj_params = Pointer(Void*).malloc(5)
      out_proj_params[0] = box_ptr(d_output_w).as(Void*)
      out_proj_params[1] = d_attn_cur_ptr.as(Void*)
      out_proj_params[2] = d_proj_cur_ptr.as(Void*)
      out_proj_params[3] = box_u32(output_in_dim_u32).as(Void*)
      out_proj_params[4] = box_u32(output_out_dim_u32).as(Void*)

      d_residual_cur_ptr = box_ptr(d_residual_input)
      d_proj_cur_ptr_for_add = box_ptr(d_proj_out)
      d_residual_out_cur_ptr = box_ptr(d_residual)
      d_cur2_cur_ptr = box_ptr(d_cur2)
      d_final_cur_ptr = box_ptr(d_final_all)

      add_rms_params = Pointer(Void*).malloc(7)
      add_rms_params[0] = d_residual_cur_ptr.as(Void*)
      add_rms_params[1] = d_proj_cur_ptr_for_add.as(Void*)
      add_rms_params[2] = box_ptr(d_post_norm).as(Void*)
      add_rms_params[3] = d_residual_out_cur_ptr.as(Void*)
      add_rms_params[4] = d_cur2_cur_ptr.as(Void*)
      add_rms_params[5] = box_u32(output_out_dim_u32).as(Void*)
      add_rms_params[6] = box_f32(@eps).as(Void*)

      d_ffn_input_cur_ptr = box_ptr(d_cur2)
      d_ffn_gate_cur_ptr = box_ptr(d_ffn_gate)
      d_ffn_up_cur_ptr = box_ptr(d_ffn_up)
      d_ffn_comb_cur_ptr = box_ptr(d_ffn_comb)
      d_ffn_residual_cur_ptr = box_ptr(d_residual)
      swiglu_n_param = box_u32(ffn_dim_u32)

      ffn_gate_params = Pointer(Void*).malloc(5)
      ffn_gate_params[0] = box_ptr(d_ffn_gate_w).as(Void*)
      ffn_gate_params[1] = d_ffn_input_cur_ptr.as(Void*)
      ffn_gate_params[2] = d_ffn_gate_cur_ptr.as(Void*)
      ffn_gate_params[3] = box_u32(output_out_dim_u32).as(Void*)
      ffn_gate_params[4] = box_u32(ffn_dim_u32).as(Void*)

      ffn_up_params = Pointer(Void*).malloc(5)
      ffn_up_params[0] = box_ptr(d_ffn_up_w).as(Void*)
      ffn_up_params[1] = d_ffn_input_cur_ptr.as(Void*)
      ffn_up_params[2] = d_ffn_up_cur_ptr.as(Void*)
      ffn_up_params[3] = box_u32(output_out_dim_u32).as(Void*)
      ffn_up_params[4] = box_u32(ffn_dim_u32).as(Void*)

      swiglu_params = Pointer(Void*).malloc(4)
      swiglu_params[0] = d_ffn_gate_cur_ptr.as(Void*)
      swiglu_params[1] = d_ffn_up_cur_ptr.as(Void*)
      swiglu_params[2] = d_ffn_comb_cur_ptr.as(Void*)
      swiglu_params[3] = swiglu_n_param.as(Void*)

      ffn_down_params = Pointer(Void*).malloc(6)
      ffn_down_params[0] = box_ptr(d_ffn_down_w).as(Void*)
      ffn_down_params[1] = d_ffn_comb_cur_ptr.as(Void*)
      ffn_down_params[2] = d_ffn_residual_cur_ptr.as(Void*)
      ffn_down_params[3] = d_final_cur_ptr.as(Void*)
      ffn_down_params[4] = box_u32(ffn_dim_u32).as(Void*)
      ffn_down_params[5] = box_u32(output_out_dim_u32).as(Void*)

      run_q4_weight_stationary = ->(params : Pointer(Void*), x_ptr : Pointer(DevicePtr), out_ptr : Pointer(DevicePtr),
                                    x_base : DevicePtr, out_base : DevicePtr, grid : UInt32,
                                    in_dim : Int32, out_dim : Int32, label : String, active_count : Int32) {
        if use_q4_tbatch4 && active_count % 4 == 0
          groups = active_count // 4
          groups.times do |group|
            x_ptr.value = x_base + bytesize_f32(group * 4 * in_dim)
            out_ptr.value = out_base + bytesize_f32(group * 4 * out_dim)
            ML::CUDA.launch!(q4_tbatch4_fn, grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, params, "#{label} tbatch4")
          end
        else
          x_ptr.value = x_base
          out_ptr.value = out_base
          ML::CUDA.launch!(q4_batched_fn, grid * active_count.to_u32, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, params, label)
        end
      }

      run_ffn_down_add = ->(active_count : Int32) {
        if use_q6_tbatch4 && active_count % 4 == 0
          groups = active_count // 4
          groups.times do |group|
            d_ffn_comb_cur_ptr.value = d_ffn_comb + bytesize_f32(group * 4 * @ffn_dim)
            d_ffn_residual_cur_ptr.value = d_residual + bytesize_f32(group * 4 * @output_out_dim)
            d_final_cur_ptr.value = d_final_all + bytesize_f32(group * 4 * @output_out_dim)
            ML::CUDA.launch!(q6_add_tbatch4_fn, output_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, ffn_down_params, "full ffn down add tbatch4")
          end
        elsif use_q4_down_add_tbatch4 && active_count % 4 == 0
          groups = active_count // 4
          groups.times do |group|
            d_ffn_comb_cur_ptr.value = d_ffn_comb + bytesize_f32(group * 4 * @ffn_dim)
            d_ffn_residual_cur_ptr.value = d_residual + bytesize_f32(group * 4 * @output_out_dim)
            d_final_cur_ptr.value = d_final_all + bytesize_f32(group * 4 * @output_out_dim)
            ML::CUDA.launch!(q4_add_tbatch4_fn, output_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, ffn_down_params, "full ffn down add q4 tbatch4")
          end
        else
          d_ffn_comb_cur_ptr.value = d_ffn_comb
          d_ffn_residual_cur_ptr.value = d_residual
          d_final_cur_ptr.value = d_final_all
          ML::CUDA.launch!(ffn_down_add_batched_fn, output_grid * active_count.to_u32, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, ffn_down_params, "full ffn down add batched")
        end
      }

      run_attn_output_projection = ->(active_count : Int32) {
        if use_attn_output_tbatch4 && active_count % 4 == 0
          groups = active_count // 4
          groups.times do |group|
            d_attn_cur_ptr.value = d_attn_out + bytesize_f32(group * 4 * @output_in_dim)
            d_proj_cur_ptr.value = d_proj_out + bytesize_f32(group * 4 * @output_out_dim)
            ML::CUDA.launch!(out_proj_tbatch4_fn, output_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, out_proj_params, "attn output tbatch4")
          end
          d_attn_cur_ptr.value = d_attn_out
          d_proj_cur_ptr.value = d_proj_out
        elsif use_batched_tail
          d_attn_cur_ptr.value = d_attn_out
          d_proj_cur_ptr.value = d_proj_out
          ML::CUDA.launch!(out_proj_batched_fn, output_grid * active_count.to_u32, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, out_proj_params, "attn output batched")
        else
          active_count.times do |t|
            d_attn_cur_ptr.value = d_attn_out + bytesize_f32(t * @output_in_dim)
            d_proj_cur_ptr.value = d_proj_out + bytesize_f32(t * @output_out_dim)
            ML::CUDA.launch!(out_proj_fn, output_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, out_proj_params, "attn output")
          end
        end
      }

      run_token = ->(tok : Int32) {
        # Kernels index token by block id; launch all token blocks once.
        if tok == 0
          active_count = @active_tokens
          active_ffn_dim_all_u32 = (active_count * @ffn_dim).to_u32
          active_swiglu_grid_all = (((active_count * @ffn_dim) + 127) // 128).to_u32
          if @profile_override_detail
            t_qk = Time.instant
            ML::CUDA.launch!(q_fn, active_count.to_u32, @n_head.to_u32, 1_u32, 1_u32, 1_u32, 1_u32, q_params, "q norm rope")
            ML::CUDA.launch!(k_fn, active_count.to_u32, @n_head_kv.to_u32, 1_u32, 1_u32, 1_u32, 1_u32, k_params, "k norm rope cache")
            ML::CUDA.synchronize!("cuCtxSynchronize(full batched qk rope)")
            @profile_qk_rope_ms += (Time.instant - t_qk).total_milliseconds

            t_attn = Time.instant
            ML::CUDA.launch!(attn_decode_fn, active_count.to_u32, @n_head.to_u32, 1_u32, attn_decode_block, 1_u32, 1_u32, attn_params, "attn decode cache")
            ML::CUDA.synchronize!("cuCtxSynchronize(full batched attn decode)")
            @profile_attn_decode_ms += (Time.instant - t_attn).total_milliseconds
          else
            ML::CUDA.launch!(q_fn, active_count.to_u32, @n_head.to_u32, 1_u32, 1_u32, 1_u32, 1_u32, q_params, "q norm rope")
            ML::CUDA.launch!(k_fn, active_count.to_u32, @n_head_kv.to_u32, 1_u32, 1_u32, 1_u32, 1_u32, k_params, "k norm rope cache")
            ML::CUDA.launch!(attn_decode_fn, active_count.to_u32, @n_head.to_u32, 1_u32, attn_decode_block, 1_u32, 1_u32, attn_params, "attn decode cache")
          end
          if use_batched_tail
            if @profile_override_detail
              t_out = Time.instant
              run_attn_output_projection.call(active_count)
              ML::CUDA.synchronize!("cuCtxSynchronize(full batched attn output)")
              @profile_out_proj_ms += (Time.instant - t_out).total_milliseconds
            else
              run_attn_output_projection.call(active_count)
            end
            if use_batched_norms
              d_residual_cur_ptr.value = @residual_device_base.not_nil!
              d_proj_cur_ptr_for_add.value = d_proj_out
              d_residual_out_cur_ptr.value = d_residual
              d_cur2_cur_ptr.value = d_cur2
              if @profile_override_detail
                t_add_rms = Time.instant
                ML::CUDA.launch!(add_rmsnorm_batched_fn, active_count.to_u32, 1_u32, 1_u32, 256_u32, 1_u32, 1_u32, add_rms_params, "full add rmsnorm batched")
                ML::CUDA.synchronize!("cuCtxSynchronize(full batched add rmsnorm)")
                @profile_add_rms_ms += (Time.instant - t_add_rms).total_milliseconds
              else
                ML::CUDA.launch!(add_rmsnorm_batched_fn, active_count.to_u32, 1_u32, 1_u32, 256_u32, 1_u32, 1_u32, add_rms_params, "full add rmsnorm batched")
              end
            else
              t_add_rms = Time.instant if @profile_override_detail
              active_count.times do |t|
                d_residual_cur_ptr.value = @residual_device_base.not_nil! + bytesize_f32(t * @output_out_dim)
                d_proj_cur_ptr_for_add.value = d_proj_out + bytesize_f32(t * @output_out_dim)
                d_residual_out_cur_ptr.value = d_residual + bytesize_f32(t * @output_out_dim)
                d_cur2_cur_ptr.value = d_cur2 + bytesize_f32(t * @output_out_dim)
                ML::CUDA.launch!(add_rmsnorm_fn, 1_u32, 1_u32, 1_u32, 256_u32, 1_u32, 1_u32, add_rms_params, "full add rmsnorm")
              end
              if @profile_override_detail
                ML::CUDA.synchronize!("cuCtxSynchronize(full batched add rmsnorm loop)")
                @profile_add_rms_ms += (Time.instant - t_add_rms.not_nil!).total_milliseconds
              end
            end
            d_ffn_input_cur_ptr.value = d_cur2
            d_ffn_gate_cur_ptr.value = d_ffn_gate
            d_ffn_up_cur_ptr.value = d_ffn_up
            d_ffn_comb_cur_ptr.value = d_ffn_comb
            d_ffn_residual_cur_ptr.value = d_residual
            d_final_cur_ptr.value = d_final_all
            swiglu_n_param.value = active_ffn_dim_all_u32
            if @profile_override_detail
              t_gate = Time.instant
              run_q4_weight_stationary.call(ffn_gate_params, d_ffn_input_cur_ptr, d_ffn_gate_cur_ptr,
                d_cur2, d_ffn_gate, ffn_grid, @output_out_dim, @ffn_dim, "full ffn gate batched", active_count)
              ML::CUDA.synchronize!("cuCtxSynchronize(full batched ffn gate)")
              @profile_ffn_gate_ms += (Time.instant - t_gate).total_milliseconds

              t_up = Time.instant
              run_q4_weight_stationary.call(ffn_up_params, d_ffn_input_cur_ptr, d_ffn_up_cur_ptr,
                d_cur2, d_ffn_up, ffn_grid, @output_out_dim, @ffn_dim, "full ffn up batched", active_count)
              ML::CUDA.synchronize!("cuCtxSynchronize(full batched ffn up)")
              @profile_ffn_up_ms += (Time.instant - t_up).total_milliseconds
            else
              run_q4_weight_stationary.call(ffn_gate_params, d_ffn_input_cur_ptr, d_ffn_gate_cur_ptr,
                d_cur2, d_ffn_gate, ffn_grid, @output_out_dim, @ffn_dim, "full ffn gate batched", active_count)
              run_q4_weight_stationary.call(ffn_up_params, d_ffn_input_cur_ptr, d_ffn_up_cur_ptr,
                d_cur2, d_ffn_up, ffn_grid, @output_out_dim, @ffn_dim, "full ffn up batched", active_count)
            end
            d_ffn_gate_cur_ptr.value = d_ffn_gate
            d_ffn_up_cur_ptr.value = d_ffn_up
            d_ffn_comb_cur_ptr.value = d_ffn_comb
            if @profile_override_detail
              t_swiglu = Time.instant
              ML::CUDA.launch!(swiglu_fn, active_swiglu_grid_all, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, swiglu_params, "full swiglu batched")
              ML::CUDA.synchronize!("cuCtxSynchronize(full batched swiglu)")
              @profile_swiglu_ms += (Time.instant - t_swiglu).total_milliseconds

              t_down = Time.instant
              run_ffn_down_add.call(active_count)
              ML::CUDA.synchronize!("cuCtxSynchronize(full batched ffn down)")
              @profile_ffn_down_ms += (Time.instant - t_down).total_milliseconds
            else
              ML::CUDA.launch!(swiglu_fn, active_swiglu_grid_all, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, swiglu_params, "full swiglu batched")
              run_ffn_down_add.call(active_count)
            end
            swiglu_n_param.value = ffn_dim_u32
            d_residual_out_cur_ptr.value = d_residual
            d_cur2_cur_ptr.value = d_cur2
          else
            run_attn_output_projection.call(active_count)
            active_count.times do |t|
              d_residual_cur_ptr.value = @residual_device_base.not_nil! + bytesize_f32(t * @output_out_dim)
              d_proj_cur_ptr_for_add.value = d_proj_out + bytesize_f32(t * @output_out_dim)
              d_residual_out_cur_ptr.value = d_residual
              d_cur2_cur_ptr.value = d_cur2
              d_final_cur_ptr.value = d_final_all + bytesize_f32(t * @output_out_dim)
              d_ffn_input_cur_ptr.value = d_cur2
              d_ffn_gate_cur_ptr.value = d_ffn_gate
              d_ffn_up_cur_ptr.value = d_ffn_up
              d_ffn_comb_cur_ptr.value = d_ffn_comb
              d_ffn_residual_cur_ptr.value = d_residual
              swiglu_n_param.value = ffn_dim_u32
              ML::CUDA.launch!(add_rmsnorm_fn, 1_u32, 1_u32, 1_u32, 256_u32, 1_u32, 1_u32, add_rms_params, "full add rmsnorm")
              ML::CUDA.launch!(q4_fn, ffn_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, ffn_gate_params, "full ffn gate")
              ML::CUDA.launch!(q4_fn, ffn_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, ffn_up_params, "full ffn up")
              ML::CUDA.launch!(swiglu_fn, swiglu_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, swiglu_params, "full swiglu")
              ML::CUDA.launch!(ffn_down_add_fn, output_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, ffn_down_params, "full ffn down add")
            end
          end
        end
      }

      profile_run_token = ->(tok : Int32) {
        if tok == 0
          t_qk = Time.instant
          ML::CUDA.launch!(q_fn, @tokens.to_u32, @n_head.to_u32, 1_u32, 1_u32, 1_u32, 1_u32, q_params, "q norm rope")
          ML::CUDA.launch!(k_fn, @tokens.to_u32, @n_head_kv.to_u32, 1_u32, 1_u32, 1_u32, 1_u32, k_params, "k norm rope cache")
          ML::CUDA.synchronize!("cuCtxSynchronize(full qk rope)")
          @profile_qk_rope_ms += (Time.instant - t_qk).total_milliseconds

          t_attn = Time.instant
          ML::CUDA.launch!(attn_decode_fn, @tokens.to_u32, @n_head.to_u32, 1_u32, attn_decode_block, 1_u32, 1_u32, attn_params, "attn decode cache")
          ML::CUDA.synchronize!("cuCtxSynchronize(full attn decode)")
          @profile_attn_decode_ms += (Time.instant - t_attn).total_milliseconds

          t_out = Time.instant
          run_attn_output_projection.call(@active_tokens)
          ML::CUDA.synchronize!("cuCtxSynchronize(full attn output)")
          @profile_out_proj_ms += (Time.instant - t_out).total_milliseconds

          @tokens.times do |t|
            d_residual_cur_ptr.value = @residual_device_base.not_nil! + bytesize_f32(t * @output_out_dim)
            d_proj_cur_ptr_for_add.value = d_proj_out + bytesize_f32(t * @output_out_dim)
            d_final_cur_ptr.value = d_final_all + bytesize_f32(t * @output_out_dim)
            t_add_rms = Time.instant
            ML::CUDA.launch!(add_rmsnorm_fn, 1_u32, 1_u32, 1_u32, 256_u32, 1_u32, 1_u32, add_rms_params, "full add rmsnorm")
            ML::CUDA.synchronize!("cuCtxSynchronize(full add rmsnorm)")
            @profile_add_rms_ms += (Time.instant - t_add_rms).total_milliseconds

            t_gate = Time.instant
            ML::CUDA.launch!(q4_fn, ffn_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, ffn_gate_params, "full ffn gate")
            ML::CUDA.synchronize!("cuCtxSynchronize(full ffn gate)")
            @profile_ffn_gate_ms += (Time.instant - t_gate).total_milliseconds

            t_up = Time.instant
            ML::CUDA.launch!(q4_fn, ffn_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, ffn_up_params, "full ffn up")
            ML::CUDA.synchronize!("cuCtxSynchronize(full ffn up)")
            @profile_ffn_up_ms += (Time.instant - t_up).total_milliseconds

            t_swiglu = Time.instant
            ML::CUDA.launch!(swiglu_fn, swiglu_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, swiglu_params, "full swiglu")
            ML::CUDA.synchronize!("cuCtxSynchronize(full swiglu)")
            @profile_swiglu_ms += (Time.instant - t_swiglu).total_milliseconds

            t_down = Time.instant
            ML::CUDA.launch!(ffn_down_add_fn, output_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, ffn_down_params, "full ffn down add")
            ML::CUDA.synchronize!("cuCtxSynchronize(full ffn down)")
            @profile_ffn_down_ms += (Time.instant - t_down).total_milliseconds
          end
        end
      }

      read_outputs = -> {
        ML::CUDA.copy_dtoh!(@q_gpu_all.to_unsafe.as(Void*), d_q_out, bytesize_f32(@q_gpu_all.size), "q_out")
        ML::CUDA.copy_dtoh!(@gate_gpu_all.to_unsafe.as(Void*), d_gate_out, bytesize_f32(@gate_gpu_all.size), "gate_out")
        ML::CUDA.copy_dtoh!(@k_gpu_all.to_unsafe.as(Void*), d_k_out, bytesize_f32(@k_gpu_all.size), "k_out")
        ML::CUDA.copy_dtoh!(@attn_gpu_all.to_unsafe.as(Void*), d_attn_out, bytesize_f32(@attn_gpu_all.size), "attn_out")
        ML::CUDA.copy_dtoh!(@proj_gpu_all.to_unsafe.as(Void*), d_proj_out, bytesize_f32(@proj_gpu_all.size), "proj_out")
        ML::CUDA.copy_dtoh!(@final_gpu_all.to_unsafe.as(Void*), d_final_all, bytesize_f32(@final_gpu_all.size), "final_out")
        ML::CUDA.copy_dtoh!(@k_cache_gpu.to_unsafe.as(Void*), d_k_cache, bytesize_f32(@k_cache_gpu.size), "k_cache")
        ML::CUDA.copy_dtoh!(@v_cache_gpu.to_unsafe.as(Void*), d_v_cache, bytesize_f32(@v_cache_gpu.size), "v_cache")
      }
      @runner = ResidentSequenceRunner.new(@tokens, upload_constants, reset_sequence, run_token, read_outputs)
      @profile_runner = ResidentSequenceRunner.new(@tokens, upload_constants, reset_sequence, profile_run_token, read_outputs)
    end

    private def runner : ResidentSequenceRunner
      @runner.not_nil!
    end

    def increment_decode_position : Nil
      @increment_start_pos.not_nil!.call
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
  end
end
