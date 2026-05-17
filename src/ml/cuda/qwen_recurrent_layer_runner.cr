require "./driver"
require "../gguf/reader"

module ML::CUDA
  # Correctness-first resident CUDA runner for one Qwen recurrent layer.
  #
  # The runner owns layer-local device buffers, CUDA modules, kernel functions,
  # and launch parameter blocks. GGUF loading and CPU-reference checks stay
  # outside so the extraction remains a narrow backend boundary.
  class QwenRecurrentLayerRunner
    @profile_runner : ResidentSequenceRunner?

    DN_PTX  = {{ read_file("src/ml/cuda/kernels/deltanet_step_probe.ptx") }}
    Q4K_PTX = {{ read_file("src/ml/cuda/kernels/q4k_gemv_probe.ptx") }}
    Q4K_DUAL_PTX = {{ read_file("src/ml/cuda/kernels/q4k_dual_gemv_probe.ptx") }}
    Q4K_RAW_Q8_PTX = {{ read_file("src/ml/cuda/kernels/q4k_raw_q8_dp4a_probe.ptx") }}
    PCA_UPDOWN_PTX = {{ read_file("src/ml/cuda/kernels/pca_updown_probe.ptx") }}
    Q5K_PTX = {{ read_file("src/ml/cuda/kernels/q5k_gemv_probe.ptx") }}
    Q6K_PTX = {{ read_file("src/ml/cuda/kernels/q6k_gemv_probe.ptx") }}

    class Weights
      getter h_k : Int32
      getter h_v : Int32
      getter s : Int32
      getter conv_k : Int32
      getter q_dim : Int32
      getter v_dim : Int32
      getter qkv_dim : Int32
      getter inner_dim : Int32
      getter scale : Float32
      getter eps : Float32
      getter hidden : Int32
      getter ffn_dim : Int32
      getter out_dim : Int32
      getter attn_norm : Array(Float32)
      getter qkv_raw : Bytes
      getter gate_raw : Bytes
      getter alpha_raw : Bytes
      getter beta_raw_w : Bytes
      getter out_raw : Bytes
      getter post_norm : Array(Float32)
      getter ffn_gate_raw : Bytes
      getter ffn_up_raw : Bytes
      getter ffn_down_raw : Bytes
      getter ffn_down_type : ML::GGUF::TensorType
      getter conv1d : Array(Float32)
      getter dt_bias : Array(Float32)
      getter ssm_a : Array(Float32)
      getter ssm_norm : Array(Float32)

      def self.load(gguf : ML::GGUF::GGUFFile, layer : Int32, eps : Float32 = 1.0e-6_f32) : self
        h_k = 16
        h_v = 32
        s = 128
        conv_k = 4
        q_dim = h_k * s
        v_dim = h_v * s
        qkv_dim = 2 * q_dim + v_dim
        inner_dim = v_dim
        scale = (1.0 / Math.sqrt(s.to_f64)).to_f32
        prefix = "blk.#{layer}"

        attn_norm_info = gguf.tensor("#{prefix}.attn_norm.weight") || raise "missing #{prefix}.attn_norm.weight"
        qkv_info = gguf.tensor("#{prefix}.attn_qkv.weight") || raise "missing #{prefix}.attn_qkv.weight"
        gate_info = gguf.tensor("#{prefix}.attn_gate.weight") || raise "missing #{prefix}.attn_gate.weight"
        alpha_info = gguf.tensor("#{prefix}.ssm_alpha.weight") || raise "missing #{prefix}.ssm_alpha.weight"
        beta_info = gguf.tensor("#{prefix}.ssm_beta.weight") || raise "missing #{prefix}.ssm_beta.weight"
        out_info = gguf.tensor("#{prefix}.ssm_out.weight") || raise "missing #{prefix}.ssm_out.weight"
        post_norm_info = gguf.tensor("#{prefix}.post_attention_norm.weight") || raise "missing #{prefix}.post_attention_norm.weight"
        ffn_gate_info = gguf.tensor("#{prefix}.ffn_gate.weight") || raise "missing #{prefix}.ffn_gate.weight"
        ffn_up_info = gguf.tensor("#{prefix}.ffn_up.weight") || raise "missing #{prefix}.ffn_up.weight"
        ffn_down_info = gguf.tensor("#{prefix}.ffn_down.weight") || raise "missing #{prefix}.ffn_down.weight"
        conv_info = gguf.tensor("#{prefix}.ssm_conv1d.weight") || raise "missing #{prefix}.ssm_conv1d.weight"
        dt_info = gguf.tensor("#{prefix}.ssm_dt.bias") || raise "missing #{prefix}.ssm_dt.bias"
        a_info = gguf.tensor("#{prefix}.ssm_a") || raise "missing #{prefix}.ssm_a"
        norm_info = gguf.tensor("#{prefix}.ssm_norm.weight") || raise "missing #{prefix}.ssm_norm.weight"
        raise "expected Q5_K attn_qkv" unless qkv_info.type.q5_k?
        raise "expected Q4_K gate/alpha/beta" unless gate_info.type.q4_k? && alpha_info.type.q4_k? && beta_info.type.q4_k?
        raise "expected Q4_K ssm_out" unless out_info.type.q4_k?
        raise "expected Q4_K ffn gate/up and Q4_K/Q6_K ffn down" unless ffn_gate_info.type.q4_k? && ffn_up_info.type.q4_k? &&
                                                                        (ffn_down_info.type.q4_k? || ffn_down_info.type.q6_k?)

        hidden = qkv_info.dims[0].to_i32
        ffn_dim = ffn_gate_info.dims[1].to_i32
        raise "attn_qkv shape mismatch" unless qkv_info.dims[1].to_i32 == qkv_dim
        raise "attn_gate shape mismatch" unless gate_info.dims[0].to_i32 == hidden && gate_info.dims[1].to_i32 == inner_dim
        raise "ssm_alpha/beta shape mismatch" unless alpha_info.dims[0].to_i32 == hidden && alpha_info.dims[1].to_i32 == h_v &&
                                                     beta_info.dims[0].to_i32 == hidden && beta_info.dims[1].to_i32 == h_v
        raise "ssm_out input mismatch" unless out_info.dims[0].to_i32 == inner_dim
        raise "ffn shape mismatch" unless ffn_gate_info.dims[0].to_i32 == hidden && ffn_up_info.dims[0].to_i32 == hidden &&
                                          ffn_up_info.dims[1].to_i32 == ffn_dim && ffn_down_info.dims[0].to_i32 == ffn_dim &&
                                          ffn_down_info.dims[1].to_i32 == hidden
        out_dim = out_info.dims[1].to_i32

        attn_norm = gguf.read_tensor_f32(attn_norm_info)
        qkv_raw = gguf.read_tensor_raw(qkv_info)
        gate_raw = gguf.read_tensor_raw(gate_info)
        alpha_raw = gguf.read_tensor_raw(alpha_info)
        beta_raw_w = gguf.read_tensor_raw(beta_info)
        out_raw = gguf.read_tensor_raw(out_info)
        post_norm = gguf.read_tensor_f32(post_norm_info)
        ffn_gate_raw = gguf.read_tensor_raw(ffn_gate_info)
        ffn_up_raw = gguf.read_tensor_raw(ffn_up_info)
        ffn_down_raw = gguf.read_tensor_raw(ffn_down_info)
        conv1d = gguf.read_tensor_f32(conv_info)
        dt_bias = gguf.read_tensor_f32(dt_info)
        ssm_a = gguf.read_tensor_f32(a_info)
        ssm_norm = gguf.read_tensor_f32(norm_info)
        raise "conv1d size mismatch" unless conv1d.size == qkv_dim * conv_k
        raise "dt/ssm_a size mismatch" unless dt_bias.size == h_v && ssm_a.size == h_v
        raise "ssm_norm size mismatch" unless ssm_norm.size == s
        raise "norm size mismatch" unless attn_norm.size == hidden && post_norm.size == hidden

        new(h_k, h_v, s, conv_k, q_dim, v_dim, qkv_dim, inner_dim, scale, eps,
          hidden, ffn_dim, out_dim, attn_norm, qkv_raw, gate_raw, alpha_raw,
          beta_raw_w, out_raw, post_norm, ffn_gate_raw, ffn_up_raw,
          ffn_down_raw, ffn_down_info.type, conv1d, dt_bias, ssm_a, ssm_norm)
      end

      def initialize(@h_k : Int32,
                     @h_v : Int32,
                     @s : Int32,
                     @conv_k : Int32,
                     @q_dim : Int32,
                     @v_dim : Int32,
                     @qkv_dim : Int32,
                     @inner_dim : Int32,
                     @scale : Float32,
                     @eps : Float32,
                     @hidden : Int32,
                     @ffn_dim : Int32,
                     @out_dim : Int32,
                     @attn_norm : Array(Float32),
                     @qkv_raw : Bytes,
                     @gate_raw : Bytes,
                     @alpha_raw : Bytes,
                     @beta_raw_w : Bytes,
                     @out_raw : Bytes,
                     @post_norm : Array(Float32),
                     @ffn_gate_raw : Bytes,
                     @ffn_up_raw : Bytes,
                     @ffn_down_raw : Bytes,
                     @ffn_down_type : ML::GGUF::TensorType,
                     @conv1d : Array(Float32),
                     @dt_bias : Array(Float32),
                     @ssm_a : Array(Float32),
                     @ssm_norm : Array(Float32))
      end
    end

    getter tokens : Int32
    getter hidden : Int32
    getter ffn_dim : Int32
    getter qkv_dim : Int32
    getter inner_dim : Int32
    getter conv_state_gpu : Array(Float32)
    getter ssm_state_gpu : Array(Float32)
    getter attn_out_gpu : Array(Float32)
    getter final_gpu_all : Array(Float32)
    getter output_device_ptr : DevicePtr
    getter conv_state_device_ptr : DevicePtr
    getter ssm_state_device_ptr : DevicePtr

    def self.from_weights(weights : Weights,
                          tokens : Int32,
                          xs : Array(Float32),
                          conv_state_init : Array(Float32),
                          ssm_state_init : Array(Float32)) : self
      new(tokens, weights.hidden, weights.ffn_dim, weights.qkv_dim, weights.q_dim,
        weights.inner_dim, weights.h_k, weights.h_v, weights.scale, weights.eps,
        xs, conv_state_init, ssm_state_init, weights.attn_norm, weights.qkv_raw,
        weights.gate_raw, weights.alpha_raw, weights.beta_raw_w, weights.out_raw,
        weights.post_norm, weights.ffn_gate_raw, weights.ffn_up_raw,
        weights.ffn_down_raw, weights.ffn_down_type, weights.conv1d,
        weights.dt_bias, weights.ssm_a, weights.ssm_norm)
    end

    private def initialize(@tokens : Int32,
                           @hidden : Int32,
                           @ffn_dim : Int32,
                           @qkv_dim : Int32,
                           @q_dim : Int32,
                           @inner_dim : Int32,
                           @h_k : Int32,
                           @h_v : Int32,
                           @scale : Float32,
                           @eps : Float32,
                           @xs : Array(Float32),
                           @conv_state_init : Array(Float32),
                           @ssm_state_init : Array(Float32),
                           @attn_norm : Array(Float32),
                           @qkv_raw : Bytes,
                           @gate_raw : Bytes,
                           @alpha_raw : Bytes,
                           @beta_raw_w : Bytes,
                           @out_raw : Bytes,
                           @post_norm : Array(Float32),
                           @ffn_gate_raw : Bytes,
                           @ffn_up_raw : Bytes,
                           @ffn_down_raw : Bytes,
                           @ffn_down_type : ML::GGUF::TensorType,
                           @conv1d : Array(Float32),
                           @dt_bias : Array(Float32),
                           @ssm_a : Array(Float32),
                           @ssm_norm : Array(Float32))
      raise ArgumentError.new("tokens must be positive") unless @tokens > 0
      raise ArgumentError.new("xs size mismatch") unless @xs.size == @tokens * @hidden

      @modules = [] of CUDAModule
      @buffers = [] of DeviceBuffer
      @param_keepalive = [] of Void*
      @conv_state_gpu = Array(Float32).new(@conv_state_init.size, 0.0_f32)
      @ssm_state_gpu = Array(Float32).new(@ssm_state_init.size, 0.0_f32)
      @attn_out_gpu = Array(Float32).new(@hidden, 0.0_f32)
      @final_gpu_all = Array(Float32).new(@tokens * @hidden, 0.0_f32)
      @input_device_base = nil.as(DevicePtr?)
      @owned_input_device_ptr = nil.as(DevicePtr?)
      @output_device_ptr = 0_u64
      @conv_state_device_ptr = 0_u64
      @ssm_state_device_ptr = 0_u64
      @profile_attn_norm_ms = 0.0
      @profile_projection_ms = 0.0
      @profile_recurrent_core_ms = 0.0
      @profile_ffn_ms = 0.0
      @profile_qkv_ms = 0.0
      @profile_gate_ms = 0.0
      @profile_alpha_beta_ms = 0.0
      @profile_ffn_gate_ms = 0.0
      @profile_ffn_up_ms = 0.0
      @profile_swiglu_ms = 0.0
      @profile_ffn_down_ms = 0.0
      @profile_ffn_pca_updown_ms = 0.0
      @profile_final_add_ms = 0.0
      @ffn_raw_q8_enabled = ENV["QWEN_CUDA_Q4_RAW_Q8_FFN"]? == "1"
      @ffn_skip_enabled = false
      @ffn_pca_updown_enabled = false
      @ffn_pca_updown_buffers = [] of DeviceBuffer
      @ffn_pca_updown_x_mean_param = Pointer(DevicePtr).null
      @ffn_pca_updown_c_mean_param = Pointer(DevicePtr).null
      @ffn_pca_updown_coeff_w_param = Pointer(DevicePtr).null
      @ffn_pca_updown_down_param = Pointer(DevicePtr).null
      @ffn_pca_updown_rank_param = Pointer(UInt32).null
      @profile_override_detail = false
      @closed = false

      build_runner
    end

    def upload_weights : Nil
      runner.upload_weights
    end

    def reset_sequence : Nil
      runner.reset_sequence
    end

    def replace_sequence_input(xs : Array(Float32)) : Nil
      raise ArgumentError.new("xs size mismatch") unless xs.size == @tokens * @hidden

      @xs = xs
      @input_device_base = @owned_input_device_ptr
    end

    def upload_sequence_input(xs : Array(Float32)) : Nil
      replace_sequence_input(xs)
      ML::CUDA.copy_htod!(@owned_input_device_ptr.not_nil!, @xs.to_unsafe.as(Void*),
        bytesize_f32(@tokens * @hidden), "xs")
    end

    def use_device_sequence_input(ptr : DevicePtr) : Nil
      raise ArgumentError.new("device input pointer must be non-zero") if ptr == 0_u64

      @input_device_base = ptr
    end

    def sequence_input_device_ptr : DevicePtr
      @input_device_base.not_nil!
    end

    def run_sequence : Nil
      runner.run_sequence
    end

    def ffn_raw_q8_enabled=(@ffn_raw_q8_enabled : Bool)
    end

    def ffn_raw_q8_enabled : Bool
      @ffn_raw_q8_enabled
    end

    def ffn_skip_enabled=(@ffn_skip_enabled : Bool)
    end

    def ffn_skip_enabled : Bool
      @ffn_skip_enabled
    end

    def set_ffn_pca_updown_adapter(x_mean : Array(Float32),
                                   c_mean : Array(Float32),
                                   coeff_w : Array(Float32),
                                   down : Array(Float32),
                                   rank : Int32) : Nil
      raise ArgumentError.new("PCA-updown rank must be in 1..64") unless rank > 0 && rank <= 64
      raise ArgumentError.new("PCA-updown x_mean size mismatch") unless x_mean.size == @hidden
      raise ArgumentError.new("PCA-updown c_mean size mismatch") unless c_mean.size >= rank
      raise ArgumentError.new("PCA-updown coeff_w size mismatch") unless coeff_w.size >= rank * @hidden
      raise ArgumentError.new("PCA-updown down size mismatch") unless down.size >= rank * @hidden
      raise "PCA-updown runner parameters are not initialized" if @ffn_pca_updown_x_mean_param.null? ||
                                                                 @ffn_pca_updown_c_mean_param.null? ||
                                                                 @ffn_pca_updown_coeff_w_param.null? ||
                                                                 @ffn_pca_updown_down_param.null? ||
                                                                 @ffn_pca_updown_rank_param.null?

      clear_ffn_pca_updown_adapter(close_buffers: true)
      x_buf = DeviceBuffer.new(bytesize_f32(@hidden))
      c_buf = DeviceBuffer.new(bytesize_f32(rank))
      coeff_buf = DeviceBuffer.new(bytesize_f32(rank * @hidden))
      down_buf = DeviceBuffer.new(bytesize_f32(rank * @hidden))
      @ffn_pca_updown_buffers = [x_buf, c_buf, coeff_buf, down_buf]

      ML::CUDA.copy_htod!(x_buf.ptr, x_mean.to_unsafe.as(Void*), bytesize_f32(@hidden), "ffn_pca_x_mean")
      ML::CUDA.copy_htod!(c_buf.ptr, c_mean.to_unsafe.as(Void*), bytesize_f32(rank), "ffn_pca_c_mean")
      ML::CUDA.copy_htod!(coeff_buf.ptr, coeff_w.to_unsafe.as(Void*), bytesize_f32(rank * @hidden), "ffn_pca_coeff_w")
      ML::CUDA.copy_htod!(down_buf.ptr, down.to_unsafe.as(Void*), bytesize_f32(rank * @hidden), "ffn_pca_down")

      @ffn_pca_updown_x_mean_param.value = x_buf.ptr
      @ffn_pca_updown_c_mean_param.value = c_buf.ptr
      @ffn_pca_updown_coeff_w_param.value = coeff_buf.ptr
      @ffn_pca_updown_down_param.value = down_buf.ptr
      @ffn_pca_updown_rank_param.value = rank.to_u32
      @ffn_pca_updown_enabled = true
      @ffn_raw_q8_enabled = false
      @ffn_skip_enabled = false
    end

    def set_zero_ffn_pca_updown_adapter(rank : Int32) : Nil
      raise ArgumentError.new("PCA-updown rank must be in 1..64") unless rank > 0 && rank <= 64

      x_mean = Array(Float32).new(@hidden, 0.0_f32)
      c_mean = Array(Float32).new(rank, 0.0_f32)
      coeff_w = Array(Float32).new(rank * @hidden, 0.0_f32)
      down = Array(Float32).new(rank * @hidden, 0.0_f32)
      set_ffn_pca_updown_adapter(x_mean, c_mean, coeff_w, down, rank)
    end

    def clear_ffn_pca_updown_adapter(close_buffers : Bool = true) : Nil
      @ffn_pca_updown_enabled = false
      unless @ffn_pca_updown_x_mean_param.null?
        @ffn_pca_updown_x_mean_param.value = 0_u64
        @ffn_pca_updown_c_mean_param.value = 0_u64
        @ffn_pca_updown_coeff_w_param.value = 0_u64
        @ffn_pca_updown_down_param.value = 0_u64
        @ffn_pca_updown_rank_param.value = 0_u32
      end
      if close_buffers
        @ffn_pca_updown_buffers.each(&.close)
        @ffn_pca_updown_buffers.clear
      end
    end

    def ffn_pca_updown_enabled : Bool
      @ffn_pca_updown_enabled
    end

    def ffn_pca_updown_enabled=(enabled : Bool) : Nil
      if enabled && @ffn_pca_updown_buffers.empty?
        raise "PCA-updown adapter buffers are not installed"
      end
      @ffn_pca_updown_enabled = enabled
      if enabled
        @ffn_raw_q8_enabled = false
        @ffn_skip_enabled = false
      end
    end

    def conv_state_bytesize : LibC::SizeT
      bytesize_f32(@conv_state_init.size)
    end

    def ssm_state_bytesize : LibC::SizeT
      bytesize_f32(@ssm_state_init.size)
    end

    def run_sequence_profiled(phase_lines : Array(String), prefix : String) : Nil
      @profile_attn_norm_ms = 0.0
      @profile_projection_ms = 0.0
      @profile_recurrent_core_ms = 0.0
      @profile_ffn_ms = 0.0
      @profile_qkv_ms = 0.0
      @profile_gate_ms = 0.0
      @profile_alpha_beta_ms = 0.0
      @profile_ffn_gate_ms = 0.0
      @profile_ffn_up_ms = 0.0
      @profile_swiglu_ms = 0.0
      @profile_ffn_down_ms = 0.0
      @profile_ffn_pca_updown_ms = 0.0
      @profile_final_add_ms = 0.0
      t_total = Time.instant
      batched_ffn_profile = ENV["QWEN_CUDA_BATCHED_FFN_OFF"]? != "1" && @tokens > 1 &&
                            !@ffn_skip_enabled && !@ffn_pca_updown_enabled && !@ffn_raw_q8_enabled
      if batched_ffn_profile
        batched_projection_profile = ENV["QWEN_CUDA_BATCHED_PROJECTIONS_OFF"]? != "1"
        @profile_override_detail = batched_projection_profile
        begin
          runner.run_sequence
          ML::CUDA.synchronize!("cuCtxSynchronize(recurrent batched WBA profile)")
        ensure
          @profile_override_detail = false
        end
        phase_lines << "#{prefix}_profile_route=#{batched_projection_profile ? "batched_projection_ffn" : "batched_ffn"}"
        phase_lines << "#{prefix}_profile_detail=#{batched_projection_profile ? "override_components" : "route_only"}"
        if batched_projection_profile
          phase_lines << "#{prefix}_attn_norm_ms=#{@profile_attn_norm_ms.round(3)}"
          phase_lines << "#{prefix}_projection_ms=#{@profile_projection_ms.round(3)}"
          phase_lines << "#{prefix}_qkv_ms=#{@profile_qkv_ms.round(3)}"
          phase_lines << "#{prefix}_gate_ms=#{@profile_gate_ms.round(3)}"
          phase_lines << "#{prefix}_alpha_beta_proj_ms=#{@profile_alpha_beta_ms.round(3)}"
          phase_lines << "#{prefix}_recurrent_core_ms=#{@profile_recurrent_core_ms.round(3)}"
          phase_lines << "#{prefix}_ffn_ms=#{@profile_ffn_ms.round(3)}"
          phase_lines << "#{prefix}_ffn_gate_ms=#{@profile_ffn_gate_ms.round(3)}"
          phase_lines << "#{prefix}_ffn_up_ms=#{@profile_ffn_up_ms.round(3)}"
          phase_lines << "#{prefix}_swiglu_ms=#{@profile_swiglu_ms.round(3)}"
          phase_lines << "#{prefix}_ffn_down_ms=#{@profile_ffn_down_ms.round(3)}"
        end
        phase_lines << "#{prefix}_profiled_ms=#{((Time.instant - t_total).total_milliseconds).round(3)}"
        return
      end

      phase_lines << "#{prefix}_profile_route=per_token"
      phase_lines << "#{prefix}_profile_detail=detailed"
      profile_runner.run_sequence
      phase_lines << "#{prefix}_attn_norm_ms=#{@profile_attn_norm_ms.round(3)}"
      phase_lines << "#{prefix}_projection_ms=#{@profile_projection_ms.round(3)}"
      phase_lines << "#{prefix}_qkv_ms=#{@profile_qkv_ms.round(3)}"
      phase_lines << "#{prefix}_gate_ms=#{@profile_gate_ms.round(3)}"
      phase_lines << "#{prefix}_alpha_beta_proj_ms=#{@profile_alpha_beta_ms.round(3)}"
      phase_lines << "#{prefix}_recurrent_core_ms=#{@profile_recurrent_core_ms.round(3)}"
      phase_lines << "#{prefix}_ffn_ms=#{@profile_ffn_ms.round(3)}"
      phase_lines << "#{prefix}_ffn_gate_ms=#{@profile_ffn_gate_ms.round(3)}"
      phase_lines << "#{prefix}_ffn_up_ms=#{@profile_ffn_up_ms.round(3)}"
      phase_lines << "#{prefix}_swiglu_ms=#{@profile_swiglu_ms.round(3)}"
      phase_lines << "#{prefix}_ffn_down_ms=#{@profile_ffn_down_ms.round(3)}"
      phase_lines << "#{prefix}_ffn_pca_updown_ms=#{@profile_ffn_pca_updown_ms.round(3)}"
      phase_lines << "#{prefix}_final_add_ms=#{@profile_final_add_ms.round(3)}"
      phase_lines << "#{prefix}_profiled_ms=#{((Time.instant - t_total).total_milliseconds).round(3)}"
    end

    def run_repeated(reps : Int32) : Int32
      runner.run_repeated(reps)
    end

    def read_outputs : Nil
      runner.read_outputs
    end

    def close : Nil
      return if @closed

      @buffers.each(&.close)
      @ffn_pca_updown_buffers.each(&.close)
      @modules.each(&.close)
      @closed = true
    end

    private def build_runner : Nil
      dn_mod = CUDAModule.load(DN_PTX, "delta")
      q4_mod = CUDAModule.load(Q4K_PTX, "q4")
      q4_dual_mod = CUDAModule.load(Q4K_DUAL_PTX, "q4_dual")
      q4_raw_q8_mod = CUDAModule.load(Q4K_RAW_Q8_PTX, "q4_raw_q8")
      pca_updown_mod = CUDAModule.load(PCA_UPDOWN_PTX, "pca_updown")
      q5_mod = CUDAModule.load(Q5K_PTX, "q5")
      q6_mod = CUDAModule.load(Q6K_PTX, "q6")
      @modules.concat([dn_mod, q4_mod, q4_dual_mod, q4_raw_q8_mod, pca_updown_mod, q5_mod, q6_mod])

      attn_norm_fn = dn_mod.function("rmsnorm_vec_parallel_probe")
      add_rmsnorm_fn = dn_mod.function("add_rmsnorm_vec_parallel_probe")
      swiglu_fn = dn_mod.function("swiglu_probe")
      add_vec_fn = dn_mod.function("add_vec_probe")
      conv_fn = dn_mod.function("recurrent_conv1d_silu_step_probe")
      norm_fn = dn_mod.function("l2_norm_128_probe")
      ab_fn = dn_mod.function("alpha_beta_transform_probe")
      dn_fn = dn_mod.function("deltanet_step_128_probe")
      post_fn = dn_mod.function("deltanet_post_norm_gate_128_probe")
      q4_fn = q4_mod.function("q4_k_gemv_warp4_f32")
      q4_add_fn = q4_mod.function("q4_k_gemv_add_warp4_f32")
      q4_batched_fn = q4_mod.function("q4_k_gemv_warp4_f32_batched")
      q4_add_batched_fn = q4_mod.function("q4_k_gemv_add_warp4_f32_batched")
      q4_dual_fn = q4_dual_mod.function("q4_k_dual_gemv_warp4_f32")
      q4_raw_q8_fn = q4_raw_q8_mod.function("q4_k_raw_q8_dp4a_gemv_warp4_f32")
      q8_quant_fn = q4_raw_q8_mod.function("quantize_q8_1_f32")
      pca_updown_fn = pca_updown_mod.function("ffn_pca_updown_fused_parallel_probe")
      q5_fn = q5_mod.function("q5_k_gemv_warp4_f32")
      q5_batched_fn = q5_mod.function("q5_k_gemv_warp4_f32_batched")
      q6_fn = q6_mod.function("q6_k_gemv_warp4_f32")
      q6_add_fn = q6_mod.function("q6_k_gemv_add_warp4_f32")
      q6_batched_fn = q6_mod.function("q6_k_gemv_warp4_f32_batched")
      q6_add_batched_fn = q6_mod.function("q6_k_gemv_add_warp4_f32_batched")
      ffn_down_add_fn = @ffn_down_type.q4_k? ? q4_add_fn : q6_add_fn
      ffn_down_add_batched_fn = @ffn_down_type.q4_k? ? q4_add_batched_fn : q6_add_batched_fn
      use_alpha_beta_dual = ENV["QWEN_CUDA_Q4_ALPHA_BETA_DUAL_OFF"]? != "1"
      use_batched_ffn = ENV["QWEN_CUDA_BATCHED_FFN_OFF"]? != "1" && @tokens > 1
      use_batched_projections = ENV["QWEN_CUDA_BATCHED_PROJECTIONS_OFF"]? != "1" && use_batched_ffn

      sizes = [bytesize_f32(@tokens * @hidden), bytesize_f32(@hidden), bytesize_f32(@tokens * @hidden),
               @qkv_raw.size.to_u64, @gate_raw.size.to_u64, @alpha_raw.size.to_u64, @beta_raw_w.size.to_u64,
               bytesize_f32(@conv_state_init.size), bytesize_f32(@ssm_state_init.size), bytesize_f32(@tokens * @qkv_dim), bytesize_f32(@conv1d.size),
               bytesize_f32(@qkv_dim), bytesize_f32(@tokens * @h_v), bytesize_f32(@tokens * @h_v), bytesize_f32(@dt_bias.size), bytesize_f32(@ssm_a.size),
               bytesize_f32(@h_v), bytesize_f32(@h_v), bytesize_f32(@tokens * @inner_dim), bytesize_f32(@ssm_norm.size), @out_raw.size.to_u64,
               bytesize_f32(@hidden), bytesize_f32(@hidden), bytesize_f32(@tokens * @hidden), bytesize_f32(@tokens * @hidden),
               @ffn_gate_raw.size.to_u64, @ffn_up_raw.size.to_u64, @ffn_down_raw.size.to_u64,
               bytesize_f32(@tokens * @ffn_dim), bytesize_f32(@tokens * @ffn_dim), bytesize_f32(@tokens * @ffn_dim), bytesize_f32(@hidden), bytesize_f32(@tokens * @hidden),
               q8_pack_bytes(@hidden), bytesize_f32(@hidden // 32), bytesize_f32(@hidden)]
      ptrs = sizes.map do |size_bytes|
        buffer = DeviceBuffer.new(size_bytes)
        @buffers << buffer
        buffer.ptr
      end

      d_xs, d_attn_norm_w, d_cur, d_qkv_w, d_gate_w, d_alpha_w, d_beta_w, d_conv_state, d_ssm_state, d_qkv, d_conv_w, d_conv_out, d_alpha, d_beta_raw, d_dt, d_a, d_g, d_b, d_z, d_norm, d_out_w, d_attn_out, d_post_norm_w, d_residual, d_cur2, d_ffn_gate_w, d_ffn_up_w, d_ffn_down_w, d_ffn_gate, d_ffn_up, d_ffn_comb, d_ffn_out, d_final_all, d_ffn_q8_packs, d_ffn_q8_scales, d_zero_hidden = ptrs
      @owned_input_device_ptr = d_xs
      @input_device_base = d_xs
      @output_device_ptr = d_final_all
      @conv_state_device_ptr = d_conv_state
      @ssm_state_device_ptr = d_ssm_state

      upload_weights = -> {
        ML::CUDA.copy_htod!(d_attn_norm_w, @attn_norm.to_unsafe.as(Void*), bytesize_f32(@hidden), "attn_norm")
        ML::CUDA.copy_htod!(d_qkv_w, @qkv_raw.to_unsafe.as(Void*), @qkv_raw.size.to_u64, "qkv_w")
        ML::CUDA.copy_htod!(d_gate_w, @gate_raw.to_unsafe.as(Void*), @gate_raw.size.to_u64, "gate_w")
        ML::CUDA.copy_htod!(d_alpha_w, @alpha_raw.to_unsafe.as(Void*), @alpha_raw.size.to_u64, "alpha_w")
        ML::CUDA.copy_htod!(d_beta_w, @beta_raw_w.to_unsafe.as(Void*), @beta_raw_w.size.to_u64, "beta_w")
        ML::CUDA.copy_htod!(d_conv_w, @conv1d.to_unsafe.as(Void*), bytesize_f32(@conv1d.size), "conv_w")
        ML::CUDA.copy_htod!(d_dt, @dt_bias.to_unsafe.as(Void*), bytesize_f32(@dt_bias.size), "dt")
        ML::CUDA.copy_htod!(d_a, @ssm_a.to_unsafe.as(Void*), bytesize_f32(@ssm_a.size), "a")
        ML::CUDA.copy_htod!(d_norm, @ssm_norm.to_unsafe.as(Void*), bytesize_f32(@ssm_norm.size), "norm")
        ML::CUDA.copy_htod!(d_out_w, @out_raw.to_unsafe.as(Void*), @out_raw.size.to_u64, "out_w")
        ML::CUDA.copy_htod!(d_post_norm_w, @post_norm.to_unsafe.as(Void*), bytesize_f32(@hidden), "post_norm")
        ML::CUDA.copy_htod!(d_ffn_gate_w, @ffn_gate_raw.to_unsafe.as(Void*), @ffn_gate_raw.size.to_u64, "ffn_gate_w")
        ML::CUDA.copy_htod!(d_ffn_up_w, @ffn_up_raw.to_unsafe.as(Void*), @ffn_up_raw.size.to_u64, "ffn_up_w")
        ML::CUDA.copy_htod!(d_ffn_down_w, @ffn_down_raw.to_unsafe.as(Void*), @ffn_down_raw.size.to_u64, "ffn_down_w")
        zero_hidden = Array(Float32).new(@hidden, 0.0_f32)
        ML::CUDA.copy_htod!(d_zero_hidden, zero_hidden.to_unsafe.as(Void*), bytesize_f32(@hidden), "zero_hidden")
      }

      reset_sequence = -> {
        if @input_device_base == d_xs
          ML::CUDA.copy_htod!(d_xs, @xs.to_unsafe.as(Void*), bytesize_f32(@tokens * @hidden), "xs")
        end
        ML::CUDA.copy_htod!(d_conv_state, @conv_state_init.to_unsafe.as(Void*), bytesize_f32(@conv_state_init.size), "conv_state")
        ML::CUDA.copy_htod!(d_ssm_state, @ssm_state_init.to_unsafe.as(Void*), bytesize_f32(@ssm_state_init.size), "ssm_state")
      }

      hidden_u32 = @hidden.to_u32
      ffn_dim_u32 = @ffn_dim.to_u32
      ffn_dim_all_u32 = (@tokens * @ffn_dim).to_u32
      qkv_dim_u32 = @qkv_dim.to_u32
      h_k_u32 = @h_k.to_u32
      h_v_u32 = @h_v.to_u32
      inner_u32 = @inner_dim.to_u32
      qkv_grid = ((@qkv_dim + 3) // 4).to_u32
      inner_grid = ((@inner_dim + 3) // 4).to_u32
      h_v_grid = ((@h_v + 3) // 4).to_u32
      alpha_beta_dual_grid = (((@h_v * 2) + 3) // 4).to_u32
      ffn_grid = ((@ffn_dim + 3) // 4).to_u32
      hidden_grid = ((@hidden + 3) // 4).to_u32
      swiglu_grid = ((@ffn_dim + 127) // 128).to_u32
      swiglu_grid_all = (((@tokens * @ffn_dim) + 127) // 128).to_u32
      conv_grid = ((@qkv_dim + 127) // 128).to_u32
      d_q = d_conv_out
      d_k = d_conv_out + bytesize_f32(@q_dim)
      d_v = d_conv_out + bytesize_f32(2 * @q_dim)
      d_x_cur_ptr = box_ptr(d_xs)
      d_cur_cur_ptr = box_ptr(d_cur)
      d_qkv_cur_ptr = box_ptr(d_qkv)
      d_alpha_cur_ptr = box_ptr(d_alpha)
      d_beta_raw_cur_ptr = box_ptr(d_beta_raw)
      d_z_cur_ptr = box_ptr(d_z)
      d_final_cur_ptr = box_ptr(d_final_all)
      d_residual_cur_ptr = box_ptr(d_residual)
      d_cur2_cur_ptr = box_ptr(d_cur2)
      d_ffn_gate_cur_ptr = box_ptr(d_ffn_gate)
      d_ffn_up_cur_ptr = box_ptr(d_ffn_up)
      d_ffn_comb_cur_ptr = box_ptr(d_ffn_comb)

      attn_norm_params = Pointer(Void*).malloc(5)
      attn_norm_params[0] = d_x_cur_ptr.as(Void*)
      attn_norm_params[1] = box_ptr(d_attn_norm_w).as(Void*)
      attn_norm_params[2] = d_cur_cur_ptr.as(Void*)
      attn_norm_params[3] = box_u32(hidden_u32).as(Void*)
      attn_norm_params[4] = box_f32(@eps).as(Void*)

      qkv_proj_params = Pointer(Void*).malloc(5)
      qkv_proj_params[0] = box_ptr(d_qkv_w).as(Void*)
      qkv_proj_params[1] = d_cur_cur_ptr.as(Void*)
      qkv_proj_params[2] = d_qkv_cur_ptr.as(Void*)
      qkv_proj_params[3] = box_u32(hidden_u32).as(Void*)
      qkv_proj_params[4] = box_u32(qkv_dim_u32).as(Void*)

      gate_proj_params = Pointer(Void*).malloc(5)
      gate_proj_params[0] = box_ptr(d_gate_w).as(Void*)
      gate_proj_params[1] = d_cur_cur_ptr.as(Void*)
      gate_proj_params[2] = d_z_cur_ptr.as(Void*)
      gate_proj_params[3] = box_u32(hidden_u32).as(Void*)
      gate_proj_params[4] = box_u32(inner_u32).as(Void*)

      alpha_proj_params = Pointer(Void*).malloc(5)
      alpha_proj_params[0] = box_ptr(d_alpha_w).as(Void*)
      alpha_proj_params[1] = d_cur_cur_ptr.as(Void*)
      alpha_proj_params[2] = d_alpha_cur_ptr.as(Void*)
      alpha_proj_params[3] = box_u32(hidden_u32).as(Void*)
      alpha_proj_params[4] = box_u32(h_v_u32).as(Void*)

      beta_proj_params = Pointer(Void*).malloc(5)
      beta_proj_params[0] = box_ptr(d_beta_w).as(Void*)
      beta_proj_params[1] = d_cur_cur_ptr.as(Void*)
      beta_proj_params[2] = d_beta_raw_cur_ptr.as(Void*)
      beta_proj_params[3] = box_u32(hidden_u32).as(Void*)
      beta_proj_params[4] = box_u32(h_v_u32).as(Void*)

      alpha_beta_dual_proj_params = Pointer(Void*).malloc(7)
      alpha_beta_dual_proj_params[0] = box_ptr(d_alpha_w).as(Void*)
      alpha_beta_dual_proj_params[1] = box_ptr(d_beta_w).as(Void*)
      alpha_beta_dual_proj_params[2] = d_cur_cur_ptr.as(Void*)
      alpha_beta_dual_proj_params[3] = d_alpha_cur_ptr.as(Void*)
      alpha_beta_dual_proj_params[4] = d_beta_raw_cur_ptr.as(Void*)
      alpha_beta_dual_proj_params[5] = box_u32(hidden_u32).as(Void*)
      alpha_beta_dual_proj_params[6] = box_u32(h_v_u32).as(Void*)

      conv_params = Pointer(Void*).malloc(5)
      conv_params[0] = box_ptr(d_conv_state).as(Void*)
      conv_params[1] = d_qkv_cur_ptr.as(Void*)
      conv_params[2] = box_ptr(d_conv_w).as(Void*)
      conv_params[3] = box_ptr(d_conv_out).as(Void*)
      conv_params[4] = box_u32(qkv_dim_u32).as(Void*)

      q_norm_params = Pointer(Void*).malloc(3)
      q_norm_params[0] = box_ptr(d_q).as(Void*)
      q_norm_params[1] = box_u32(h_k_u32).as(Void*)
      q_norm_params[2] = box_f32(@eps).as(Void*)
      k_norm_params = Pointer(Void*).malloc(3)
      k_norm_params[0] = box_ptr(d_k).as(Void*)
      k_norm_params[1] = box_u32(h_k_u32).as(Void*)
      k_norm_params[2] = box_f32(@eps).as(Void*)

      ab_params = Pointer(Void*).malloc(7)
      ab_params[0] = d_alpha_cur_ptr.as(Void*)
      ab_params[1] = d_beta_raw_cur_ptr.as(Void*)
      ab_params[2] = box_ptr(d_dt).as(Void*)
      ab_params[3] = box_ptr(d_a).as(Void*)
      ab_params[4] = box_ptr(d_g).as(Void*)
      ab_params[5] = box_ptr(d_b).as(Void*)
      ab_params[6] = box_u32(h_v_u32).as(Void*)

      dn_params = Pointer(Void*).malloc(10)
      dn_params[0] = box_ptr(d_ssm_state).as(Void*)
      dn_params[1] = box_ptr(d_q).as(Void*)
      dn_params[2] = box_ptr(d_k).as(Void*)
      dn_params[3] = box_ptr(d_v).as(Void*)
      dn_params[4] = box_ptr(d_g).as(Void*)
      dn_params[5] = box_ptr(d_b).as(Void*)
      dn_params[6] = box_ptr(d_v).as(Void*)
      dn_params[7] = box_u32(h_k_u32).as(Void*)
      dn_params[8] = box_u32(h_v_u32).as(Void*)
      dn_params[9] = box_f32(@scale).as(Void*)

      post_params = Pointer(Void*).malloc(5)
      post_params[0] = box_ptr(d_v).as(Void*)
      post_params[1] = d_z_cur_ptr.as(Void*)
      post_params[2] = box_ptr(d_norm).as(Void*)
      post_params[3] = box_u32(h_v_u32).as(Void*)
      post_params[4] = box_f32(@eps).as(Void*)

      out_proj_params = Pointer(Void*).malloc(5)
      out_proj_params[0] = box_ptr(d_out_w).as(Void*)
      out_proj_params[1] = box_ptr(d_v).as(Void*)
      out_proj_params[2] = box_ptr(d_attn_out).as(Void*)
      out_proj_params[3] = box_u32(inner_u32).as(Void*)
      out_proj_params[4] = box_u32(hidden_u32).as(Void*)

      add_rms_params = Pointer(Void*).malloc(7)
      add_rms_params[0] = d_x_cur_ptr.as(Void*)
      add_rms_params[1] = box_ptr(d_attn_out).as(Void*)
      add_rms_params[2] = box_ptr(d_post_norm_w).as(Void*)
      add_rms_params[3] = d_residual_cur_ptr.as(Void*)
      add_rms_params[4] = d_cur2_cur_ptr.as(Void*)
      add_rms_params[5] = box_u32(hidden_u32).as(Void*)
      add_rms_params[6] = box_f32(@eps).as(Void*)

      ffn_gate_params = Pointer(Void*).malloc(5)
      ffn_gate_params[0] = box_ptr(d_ffn_gate_w).as(Void*)
      ffn_gate_params[1] = d_cur2_cur_ptr.as(Void*)
      ffn_gate_params[2] = d_ffn_gate_cur_ptr.as(Void*)
      ffn_gate_params[3] = box_u32(hidden_u32).as(Void*)
      ffn_gate_params[4] = box_u32(ffn_dim_u32).as(Void*)

      ffn_up_params = Pointer(Void*).malloc(5)
      ffn_up_params[0] = box_ptr(d_ffn_up_w).as(Void*)
      ffn_up_params[1] = d_cur2_cur_ptr.as(Void*)
      ffn_up_params[2] = d_ffn_up_cur_ptr.as(Void*)
      ffn_up_params[3] = box_u32(hidden_u32).as(Void*)
      ffn_up_params[4] = box_u32(ffn_dim_u32).as(Void*)

      ffn_q8_quant_params = Pointer(Void*).malloc(4)
      ffn_q8_quant_params[0] = d_cur2_cur_ptr.as(Void*)
      ffn_q8_quant_params[1] = box_ptr(d_ffn_q8_packs).as(Void*)
      ffn_q8_quant_params[2] = box_ptr(d_ffn_q8_scales).as(Void*)
      ffn_q8_quant_params[3] = box_u32(hidden_u32).as(Void*)

      ffn_gate_raw_q8_params = Pointer(Void*).malloc(6)
      ffn_gate_raw_q8_params[0] = box_ptr(d_ffn_gate_w).as(Void*)
      ffn_gate_raw_q8_params[1] = box_ptr(d_ffn_q8_packs).as(Void*)
      ffn_gate_raw_q8_params[2] = box_ptr(d_ffn_q8_scales).as(Void*)
      ffn_gate_raw_q8_params[3] = d_ffn_gate_cur_ptr.as(Void*)
      ffn_gate_raw_q8_params[4] = box_u32(hidden_u32).as(Void*)
      ffn_gate_raw_q8_params[5] = box_u32(ffn_dim_u32).as(Void*)

      ffn_up_raw_q8_params = Pointer(Void*).malloc(6)
      ffn_up_raw_q8_params[0] = box_ptr(d_ffn_up_w).as(Void*)
      ffn_up_raw_q8_params[1] = box_ptr(d_ffn_q8_packs).as(Void*)
      ffn_up_raw_q8_params[2] = box_ptr(d_ffn_q8_scales).as(Void*)
      ffn_up_raw_q8_params[3] = d_ffn_up_cur_ptr.as(Void*)
      ffn_up_raw_q8_params[4] = box_u32(hidden_u32).as(Void*)
      ffn_up_raw_q8_params[5] = box_u32(ffn_dim_u32).as(Void*)

      swiglu_params = Pointer(Void*).malloc(4)
      swiglu_params[0] = d_ffn_gate_cur_ptr.as(Void*)
      swiglu_params[1] = d_ffn_up_cur_ptr.as(Void*)
      swiglu_params[2] = d_ffn_comb_cur_ptr.as(Void*)
      swiglu_n_param = box_u32(ffn_dim_u32)
      swiglu_params[3] = swiglu_n_param.as(Void*)

      ffn_down_params = Pointer(Void*).malloc(6)
      ffn_down_params[0] = box_ptr(d_ffn_down_w).as(Void*)
      ffn_down_params[1] = d_ffn_comb_cur_ptr.as(Void*)
      ffn_down_params[2] = d_residual_cur_ptr.as(Void*)
      ffn_down_params[3] = d_final_cur_ptr.as(Void*)
      ffn_down_params[4] = box_u32(ffn_dim_u32).as(Void*)
      ffn_down_params[5] = box_u32(hidden_u32).as(Void*)

      ffn_skip_copy_params = Pointer(Void*).malloc(4)
      ffn_skip_copy_params[0] = d_residual_cur_ptr.as(Void*)
      ffn_skip_copy_params[1] = box_ptr(d_zero_hidden).as(Void*)
      ffn_skip_copy_params[2] = d_final_cur_ptr.as(Void*)
      ffn_skip_copy_params[3] = box_u32(hidden_u32).as(Void*)

      @ffn_pca_updown_x_mean_param = box_ptr(0_u64)
      @ffn_pca_updown_c_mean_param = box_ptr(0_u64)
      @ffn_pca_updown_coeff_w_param = box_ptr(0_u64)
      @ffn_pca_updown_down_param = box_ptr(0_u64)
      @ffn_pca_updown_rank_param = box_u32(0_u32)

      ffn_pca_updown_params = Pointer(Void*).malloc(8)
      ffn_pca_updown_params[0] = d_cur2_cur_ptr.as(Void*)
      ffn_pca_updown_params[1] = @ffn_pca_updown_x_mean_param.as(Void*)
      ffn_pca_updown_params[2] = @ffn_pca_updown_c_mean_param.as(Void*)
      ffn_pca_updown_params[3] = @ffn_pca_updown_coeff_w_param.as(Void*)
      ffn_pca_updown_params[4] = @ffn_pca_updown_down_param.as(Void*)
      ffn_pca_updown_params[5] = box_ptr(d_ffn_out).as(Void*)
      ffn_pca_updown_params[6] = box_u32(hidden_u32).as(Void*)
      ffn_pca_updown_params[7] = @ffn_pca_updown_rank_param.as(Void*)

      ffn_pca_add_params = Pointer(Void*).malloc(4)
      ffn_pca_add_params[0] = d_residual_cur_ptr.as(Void*)
      ffn_pca_add_params[1] = box_ptr(d_ffn_out).as(Void*)
      ffn_pca_add_params[2] = d_final_cur_ptr.as(Void*)
      ffn_pca_add_params[3] = box_u32(hidden_u32).as(Void*)

      run_token = ->(tok : Int32) {
        offset = bytesize_f32(tok * @hidden)
        qkv_offset = bytesize_f32(tok * @qkv_dim)
        h_v_offset = bytesize_f32(tok * @h_v)
        inner_offset = bytesize_f32(tok * @inner_dim)
        ffn_offset = bytesize_f32(tok * @ffn_dim)
        d_x_cur_ptr.value = @input_device_base.not_nil! + offset
        d_cur_cur_ptr.value = d_cur + offset
        d_qkv_cur_ptr.value = d_qkv + qkv_offset
        d_alpha_cur_ptr.value = d_alpha + h_v_offset
        d_beta_raw_cur_ptr.value = d_beta_raw + h_v_offset
        d_z_cur_ptr.value = d_z + inner_offset
        d_final_cur_ptr.value = d_final_all + offset
        d_residual_cur_ptr.value = d_residual + offset
        d_cur2_cur_ptr.value = d_cur2 + offset
        d_ffn_gate_cur_ptr.value = d_ffn_gate + ffn_offset
        d_ffn_up_cur_ptr.value = d_ffn_up + ffn_offset
        d_ffn_comb_cur_ptr.value = d_ffn_comb + ffn_offset
        swiglu_n_param.value = ffn_dim_u32
        ML::CUDA.launch!(attn_norm_fn, 1_u32, 1_u32, 1_u32, 256_u32, 1_u32, 1_u32, attn_norm_params, "attn norm")
        ML::CUDA.launch!(q5_fn, qkv_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, qkv_proj_params, "qkv proj")
        ML::CUDA.launch!(q4_fn, inner_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, gate_proj_params, "gate proj")
        if use_alpha_beta_dual
          ML::CUDA.launch!(q4_dual_fn, alpha_beta_dual_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, alpha_beta_dual_proj_params, "alpha beta dual proj")
        else
          ML::CUDA.launch!(q4_fn, h_v_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, alpha_proj_params, "alpha proj")
          ML::CUDA.launch!(q4_fn, h_v_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, beta_proj_params, "beta proj")
        end
        ML::CUDA.launch!(conv_fn, conv_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, conv_params, "conv prep")
        ML::CUDA.launch!(norm_fn, @h_k.to_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, q_norm_params, "q norm")
        ML::CUDA.launch!(norm_fn, @h_k.to_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, k_norm_params, "k norm")
        ML::CUDA.launch!(ab_fn, 1_u32, 1_u32, 1_u32, 32_u32, 1_u32, 1_u32, ab_params, "alpha beta")
        ML::CUDA.launch!(dn_fn, @h_v.to_u32, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, dn_params, "delta step")
        ML::CUDA.launch!(post_fn, @h_v.to_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, post_params, "post gate")
        ML::CUDA.launch!(q4_fn, hidden_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, out_proj_params, "ssm_out")
        ML::CUDA.launch!(add_rmsnorm_fn, 1_u32, 1_u32, 1_u32, 256_u32, 1_u32, 1_u32, add_rms_params, "add rmsnorm")
        if @ffn_skip_enabled
          ML::CUDA.launch!(add_vec_fn, hidden_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, ffn_skip_copy_params, "ffn skip copy")
        elsif @ffn_pca_updown_enabled
          ML::CUDA.launch!(pca_updown_fn, 1_u32, 1_u32, 1_u32, 256_u32, 1_u32, 1_u32, ffn_pca_updown_params, "ffn pca updown")
          ML::CUDA.launch!(add_vec_fn, hidden_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, ffn_pca_add_params, "ffn pca add")
        elsif @ffn_raw_q8_enabled
          ML::CUDA.launch!(q8_quant_fn, (@hidden // 32).to_u32, 1_u32, 1_u32, 32_u32, 1_u32, 1_u32, ffn_q8_quant_params, "ffn q8 quant")
          ML::CUDA.launch!(q4_raw_q8_fn, ffn_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, ffn_gate_raw_q8_params, "ffn gate raw q8")
          ML::CUDA.launch!(q4_raw_q8_fn, ffn_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, ffn_up_raw_q8_params, "ffn up raw q8")
        else
          ML::CUDA.launch!(q4_fn, ffn_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, ffn_gate_params, "ffn gate")
          ML::CUDA.launch!(q4_fn, ffn_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, ffn_up_params, "ffn up")
        end
        unless @ffn_skip_enabled || @ffn_pca_updown_enabled
          ML::CUDA.launch!(swiglu_fn, swiglu_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, swiglu_params, "swiglu")
          ML::CUDA.launch!(ffn_down_add_fn, hidden_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, ffn_down_params, "ffn down add")
        end
      }

      run_core_token = ->(tok : Int32) {
        offset = bytesize_f32(tok * @hidden)
        qkv_offset = bytesize_f32(tok * @qkv_dim)
        h_v_offset = bytesize_f32(tok * @h_v)
        inner_offset = bytesize_f32(tok * @inner_dim)
        ffn_offset = bytesize_f32(tok * @ffn_dim)
        d_x_cur_ptr.value = @input_device_base.not_nil! + offset
        d_cur_cur_ptr.value = d_cur + offset
        d_qkv_cur_ptr.value = d_qkv + qkv_offset
        d_alpha_cur_ptr.value = d_alpha + h_v_offset
        d_beta_raw_cur_ptr.value = d_beta_raw + h_v_offset
        d_z_cur_ptr.value = d_z + inner_offset
        d_final_cur_ptr.value = d_final_all + offset
        d_residual_cur_ptr.value = d_residual + offset
        d_cur2_cur_ptr.value = d_cur2 + offset
        d_ffn_gate_cur_ptr.value = d_ffn_gate + ffn_offset
        d_ffn_up_cur_ptr.value = d_ffn_up + ffn_offset
        d_ffn_comb_cur_ptr.value = d_ffn_comb + ffn_offset
        swiglu_n_param.value = ffn_dim_u32

        ML::CUDA.launch!(attn_norm_fn, 1_u32, 1_u32, 1_u32, 256_u32, 1_u32, 1_u32, attn_norm_params, "attn norm")
        ML::CUDA.launch!(q5_fn, qkv_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, qkv_proj_params, "qkv proj")
        ML::CUDA.launch!(q4_fn, inner_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, gate_proj_params, "gate proj")
        if use_alpha_beta_dual
          ML::CUDA.launch!(q4_dual_fn, alpha_beta_dual_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, alpha_beta_dual_proj_params, "alpha beta dual proj")
        else
          ML::CUDA.launch!(q4_fn, h_v_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, alpha_proj_params, "alpha proj")
          ML::CUDA.launch!(q4_fn, h_v_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, beta_proj_params, "beta proj")
        end
        ML::CUDA.launch!(conv_fn, conv_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, conv_params, "conv prep")
        ML::CUDA.launch!(norm_fn, @h_k.to_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, q_norm_params, "q norm")
        ML::CUDA.launch!(norm_fn, @h_k.to_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, k_norm_params, "k norm")
        ML::CUDA.launch!(ab_fn, 1_u32, 1_u32, 1_u32, 32_u32, 1_u32, 1_u32, ab_params, "alpha beta")
        ML::CUDA.launch!(dn_fn, @h_v.to_u32, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, dn_params, "delta step")
        ML::CUDA.launch!(post_fn, @h_v.to_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, post_params, "post gate")
        ML::CUDA.launch!(q4_fn, hidden_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, out_proj_params, "ssm_out")
        ML::CUDA.launch!(add_rmsnorm_fn, 1_u32, 1_u32, 1_u32, 256_u32, 1_u32, 1_u32, add_rms_params, "add rmsnorm")
      }

      run_post_projection_core_token = ->(tok : Int32) {
        offset = bytesize_f32(tok * @hidden)
        qkv_offset = bytesize_f32(tok * @qkv_dim)
        h_v_offset = bytesize_f32(tok * @h_v)
        inner_offset = bytesize_f32(tok * @inner_dim)
        ffn_offset = bytesize_f32(tok * @ffn_dim)
        d_x_cur_ptr.value = @input_device_base.not_nil! + offset
        d_cur_cur_ptr.value = d_cur + offset
        d_qkv_cur_ptr.value = d_qkv + qkv_offset
        d_alpha_cur_ptr.value = d_alpha + h_v_offset
        d_beta_raw_cur_ptr.value = d_beta_raw + h_v_offset
        d_z_cur_ptr.value = d_z + inner_offset
        d_final_cur_ptr.value = d_final_all + offset
        d_residual_cur_ptr.value = d_residual + offset
        d_cur2_cur_ptr.value = d_cur2 + offset
        d_ffn_gate_cur_ptr.value = d_ffn_gate + ffn_offset
        d_ffn_up_cur_ptr.value = d_ffn_up + ffn_offset
        d_ffn_comb_cur_ptr.value = d_ffn_comb + ffn_offset
        swiglu_n_param.value = ffn_dim_u32

        ML::CUDA.launch!(conv_fn, conv_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, conv_params, "conv prep")
        ML::CUDA.launch!(norm_fn, @h_k.to_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, q_norm_params, "q norm")
        ML::CUDA.launch!(norm_fn, @h_k.to_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, k_norm_params, "k norm")
        ML::CUDA.launch!(ab_fn, 1_u32, 1_u32, 1_u32, 32_u32, 1_u32, 1_u32, ab_params, "alpha beta")
        ML::CUDA.launch!(dn_fn, @h_v.to_u32, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, dn_params, "delta step")
        ML::CUDA.launch!(post_fn, @h_v.to_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, post_params, "post gate")
        ML::CUDA.launch!(q4_fn, hidden_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, out_proj_params, "ssm_out")
        ML::CUDA.launch!(add_rmsnorm_fn, 1_u32, 1_u32, 1_u32, 256_u32, 1_u32, 1_u32, add_rms_params, "add rmsnorm")
      }

      run_batched_ffn = -> {
        d_cur2_cur_ptr.value = d_cur2
        d_ffn_gate_cur_ptr.value = d_ffn_gate
        d_ffn_up_cur_ptr.value = d_ffn_up
        d_ffn_comb_cur_ptr.value = d_ffn_comb
        d_residual_cur_ptr.value = d_residual
        d_final_cur_ptr.value = d_final_all
        swiglu_n_param.value = ffn_dim_all_u32

        if @profile_override_detail
          t_ffn_gate = Time.instant
          ML::CUDA.launch!(q4_batched_fn, ffn_grid * @tokens.to_u32, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, ffn_gate_params, "ffn gate batched")
          ML::CUDA.synchronize!("cuCtxSynchronize(recurrent batched ffn gate)")
          @profile_ffn_gate_ms += (Time.instant - t_ffn_gate).total_milliseconds

          t_ffn_up = Time.instant
          ML::CUDA.launch!(q4_batched_fn, ffn_grid * @tokens.to_u32, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, ffn_up_params, "ffn up batched")
          ML::CUDA.synchronize!("cuCtxSynchronize(recurrent batched ffn up)")
          @profile_ffn_up_ms += (Time.instant - t_ffn_up).total_milliseconds

          t_swiglu = Time.instant
          ML::CUDA.launch!(swiglu_fn, swiglu_grid_all, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, swiglu_params, "swiglu batched")
          ML::CUDA.synchronize!("cuCtxSynchronize(recurrent batched swiglu)")
          @profile_swiglu_ms += (Time.instant - t_swiglu).total_milliseconds

          t_ffn_down = Time.instant
          ML::CUDA.launch!(ffn_down_add_batched_fn, hidden_grid * @tokens.to_u32, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, ffn_down_params, "ffn down add batched")
          ML::CUDA.synchronize!("cuCtxSynchronize(recurrent batched ffn down)")
          @profile_ffn_down_ms += (Time.instant - t_ffn_down).total_milliseconds
        else
          ML::CUDA.launch!(q4_batched_fn, ffn_grid * @tokens.to_u32, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, ffn_gate_params, "ffn gate batched")
          ML::CUDA.launch!(q4_batched_fn, ffn_grid * @tokens.to_u32, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, ffn_up_params, "ffn up batched")
          ML::CUDA.launch!(swiglu_fn, swiglu_grid_all, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, swiglu_params, "swiglu batched")
          ML::CUDA.launch!(ffn_down_add_batched_fn, hidden_grid * @tokens.to_u32, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, ffn_down_params, "ffn down add batched")
        end
      }

      run_sequence_override = -> {
        if use_batched_projections && !@ffn_skip_enabled && !@ffn_pca_updown_enabled && !@ffn_raw_q8_enabled
          if @profile_override_detail
            t_norm = Time.instant
            @tokens.times do |tok|
              offset = bytesize_f32(tok * @hidden)
              d_x_cur_ptr.value = @input_device_base.not_nil! + offset
              d_cur_cur_ptr.value = d_cur + offset
              ML::CUDA.launch!(attn_norm_fn, 1_u32, 1_u32, 1_u32, 256_u32, 1_u32, 1_u32, attn_norm_params, "attn norm")
            end
            ML::CUDA.synchronize!("cuCtxSynchronize(recurrent batched attn norm)")
            @profile_attn_norm_ms += (Time.instant - t_norm).total_milliseconds
          else
            @tokens.times do |tok|
              offset = bytesize_f32(tok * @hidden)
              d_x_cur_ptr.value = @input_device_base.not_nil! + offset
              d_cur_cur_ptr.value = d_cur + offset
              ML::CUDA.launch!(attn_norm_fn, 1_u32, 1_u32, 1_u32, 256_u32, 1_u32, 1_u32, attn_norm_params, "attn norm")
            end
          end

          d_cur_cur_ptr.value = d_cur
          d_qkv_cur_ptr.value = d_qkv
          d_z_cur_ptr.value = d_z
          d_alpha_cur_ptr.value = d_alpha
          d_beta_raw_cur_ptr.value = d_beta_raw
          if @profile_override_detail
            t_proj = Time.instant
            t_qkv = Time.instant
            ML::CUDA.launch!(q5_batched_fn, qkv_grid * @tokens.to_u32, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, qkv_proj_params, "qkv proj batched")
            ML::CUDA.synchronize!("cuCtxSynchronize(recurrent batched qkv)")
            @profile_qkv_ms += (Time.instant - t_qkv).total_milliseconds

            t_gate = Time.instant
            ML::CUDA.launch!(q4_batched_fn, inner_grid * @tokens.to_u32, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, gate_proj_params, "gate proj batched")
            ML::CUDA.synchronize!("cuCtxSynchronize(recurrent batched gate)")
            @profile_gate_ms += (Time.instant - t_gate).total_milliseconds

            t_alpha_beta = Time.instant
            ML::CUDA.launch!(q4_batched_fn, h_v_grid * @tokens.to_u32, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, alpha_proj_params, "alpha proj batched")
            ML::CUDA.launch!(q4_batched_fn, h_v_grid * @tokens.to_u32, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, beta_proj_params, "beta proj batched")
            ML::CUDA.synchronize!("cuCtxSynchronize(recurrent batched alpha/beta)")
            @profile_alpha_beta_ms += (Time.instant - t_alpha_beta).total_milliseconds
            @profile_projection_ms += (Time.instant - t_proj).total_milliseconds

            t_core = Time.instant
            @tokens.times { |tok| run_post_projection_core_token.call(tok) }
            ML::CUDA.synchronize!("cuCtxSynchronize(recurrent batched serial core)")
            @profile_recurrent_core_ms += (Time.instant - t_core).total_milliseconds

            t_ffn = Time.instant
            run_batched_ffn.call
            @profile_ffn_ms += (Time.instant - t_ffn).total_milliseconds
          else
            ML::CUDA.launch!(q5_batched_fn, qkv_grid * @tokens.to_u32, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, qkv_proj_params, "qkv proj batched")
            ML::CUDA.launch!(q4_batched_fn, inner_grid * @tokens.to_u32, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, gate_proj_params, "gate proj batched")
            ML::CUDA.launch!(q4_batched_fn, h_v_grid * @tokens.to_u32, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, alpha_proj_params, "alpha proj batched")
            ML::CUDA.launch!(q4_batched_fn, h_v_grid * @tokens.to_u32, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, beta_proj_params, "beta proj batched")

            @tokens.times { |tok| run_post_projection_core_token.call(tok) }
            run_batched_ffn.call
          end
        elsif use_batched_ffn && !@ffn_skip_enabled && !@ffn_pca_updown_enabled && !@ffn_raw_q8_enabled
          @tokens.times { |tok| run_core_token.call(tok) }
          run_batched_ffn.call
        else
          @tokens.times { |tok| run_token.call(tok) }
        end
      }

      profile_run_token = ->(tok : Int32) {
        offset = bytesize_f32(tok * @hidden)
        qkv_offset = bytesize_f32(tok * @qkv_dim)
        h_v_offset = bytesize_f32(tok * @h_v)
        inner_offset = bytesize_f32(tok * @inner_dim)
        ffn_offset = bytesize_f32(tok * @ffn_dim)
        d_x_cur_ptr.value = @input_device_base.not_nil! + offset
        d_cur_cur_ptr.value = d_cur + offset
        d_qkv_cur_ptr.value = d_qkv + qkv_offset
        d_alpha_cur_ptr.value = d_alpha + h_v_offset
        d_beta_raw_cur_ptr.value = d_beta_raw + h_v_offset
        d_z_cur_ptr.value = d_z + inner_offset
        d_final_cur_ptr.value = d_final_all + offset
        d_residual_cur_ptr.value = d_residual + offset
        d_cur2_cur_ptr.value = d_cur2 + offset
        d_ffn_gate_cur_ptr.value = d_ffn_gate + ffn_offset
        d_ffn_up_cur_ptr.value = d_ffn_up + ffn_offset
        d_ffn_comb_cur_ptr.value = d_ffn_comb + ffn_offset
        swiglu_n_param.value = ffn_dim_u32

        t_norm = Time.instant
        ML::CUDA.launch!(attn_norm_fn, 1_u32, 1_u32, 1_u32, 256_u32, 1_u32, 1_u32, attn_norm_params, "attn norm")
        ML::CUDA.synchronize!("cuCtxSynchronize(recurrent attn norm)")
        @profile_attn_norm_ms += (Time.instant - t_norm).total_milliseconds

        t_proj = Time.instant
        t_qkv = Time.instant
        ML::CUDA.launch!(q5_fn, qkv_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, qkv_proj_params, "qkv proj")
        ML::CUDA.synchronize!("cuCtxSynchronize(recurrent qkv)")
        @profile_qkv_ms += (Time.instant - t_qkv).total_milliseconds

        t_gate = Time.instant
        ML::CUDA.launch!(q4_fn, inner_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, gate_proj_params, "gate proj")
        ML::CUDA.synchronize!("cuCtxSynchronize(recurrent gate)")
        @profile_gate_ms += (Time.instant - t_gate).total_milliseconds

        t_alpha_beta = Time.instant
        if use_alpha_beta_dual
          ML::CUDA.launch!(q4_dual_fn, alpha_beta_dual_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, alpha_beta_dual_proj_params, "alpha beta dual proj")
        else
          ML::CUDA.launch!(q4_fn, h_v_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, alpha_proj_params, "alpha proj")
          ML::CUDA.launch!(q4_fn, h_v_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, beta_proj_params, "beta proj")
        end
        ML::CUDA.synchronize!("cuCtxSynchronize(recurrent alpha/beta)")
        @profile_alpha_beta_ms += (Time.instant - t_alpha_beta).total_milliseconds
        @profile_projection_ms += (Time.instant - t_proj).total_milliseconds

        t_core = Time.instant
        ML::CUDA.launch!(conv_fn, conv_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, conv_params, "conv prep")
        ML::CUDA.launch!(norm_fn, @h_k.to_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, q_norm_params, "q norm")
        ML::CUDA.launch!(norm_fn, @h_k.to_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, k_norm_params, "k norm")
        ML::CUDA.launch!(ab_fn, 1_u32, 1_u32, 1_u32, 32_u32, 1_u32, 1_u32, ab_params, "alpha beta")
        ML::CUDA.launch!(dn_fn, @h_v.to_u32, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, dn_params, "delta step")
        ML::CUDA.launch!(post_fn, @h_v.to_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, post_params, "post gate")
        ML::CUDA.launch!(q4_fn, hidden_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, out_proj_params, "ssm_out")
        ML::CUDA.launch!(add_rmsnorm_fn, 1_u32, 1_u32, 1_u32, 256_u32, 1_u32, 1_u32, add_rms_params, "add rmsnorm")
        ML::CUDA.synchronize!("cuCtxSynchronize(recurrent core)")
        @profile_recurrent_core_ms += (Time.instant - t_core).total_milliseconds

        t_ffn = Time.instant
        if @ffn_skip_enabled
          t_skip = Time.instant
          ML::CUDA.launch!(add_vec_fn, hidden_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, ffn_skip_copy_params, "ffn skip copy")
          ML::CUDA.synchronize!("cuCtxSynchronize(recurrent ffn skip)")
          @profile_final_add_ms += (Time.instant - t_skip).total_milliseconds
        elsif @ffn_pca_updown_enabled
          t_pca = Time.instant
          ML::CUDA.launch!(pca_updown_fn, 1_u32, 1_u32, 1_u32, 256_u32, 1_u32, 1_u32, ffn_pca_updown_params, "ffn pca updown")
          ML::CUDA.synchronize!("cuCtxSynchronize(recurrent ffn pca updown)")
          @profile_ffn_pca_updown_ms += (Time.instant - t_pca).total_milliseconds

          t_pca_add = Time.instant
          ML::CUDA.launch!(add_vec_fn, hidden_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, ffn_pca_add_params, "ffn pca add")
          ML::CUDA.synchronize!("cuCtxSynchronize(recurrent ffn pca add)")
          @profile_final_add_ms += (Time.instant - t_pca_add).total_milliseconds
        else
          t_ffn_gate = Time.instant
          if @ffn_raw_q8_enabled
            ML::CUDA.launch!(q8_quant_fn, (@hidden // 32).to_u32, 1_u32, 1_u32, 32_u32, 1_u32, 1_u32, ffn_q8_quant_params, "ffn q8 quant")
            ML::CUDA.launch!(q4_raw_q8_fn, ffn_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, ffn_gate_raw_q8_params, "ffn gate raw q8")
          else
            ML::CUDA.launch!(q4_fn, ffn_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, ffn_gate_params, "ffn gate")
          end
          ML::CUDA.synchronize!("cuCtxSynchronize(recurrent ffn gate)")
          @profile_ffn_gate_ms += (Time.instant - t_ffn_gate).total_milliseconds

          t_ffn_up = Time.instant
          if @ffn_raw_q8_enabled
            ML::CUDA.launch!(q4_raw_q8_fn, ffn_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, ffn_up_raw_q8_params, "ffn up raw q8")
          else
            ML::CUDA.launch!(q4_fn, ffn_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, ffn_up_params, "ffn up")
          end
          ML::CUDA.synchronize!("cuCtxSynchronize(recurrent ffn up)")
          @profile_ffn_up_ms += (Time.instant - t_ffn_up).total_milliseconds

          t_swiglu = Time.instant
          ML::CUDA.launch!(swiglu_fn, swiglu_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, swiglu_params, "swiglu")
          ML::CUDA.synchronize!("cuCtxSynchronize(recurrent swiglu)")
          @profile_swiglu_ms += (Time.instant - t_swiglu).total_milliseconds

          t_ffn_down = Time.instant
          ML::CUDA.launch!(ffn_down_add_fn, hidden_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, ffn_down_params, "ffn down add")
          ML::CUDA.synchronize!("cuCtxSynchronize(recurrent ffn down)")
          @profile_ffn_down_ms += (Time.instant - t_ffn_down).total_milliseconds
        end

        @profile_ffn_ms += (Time.instant - t_ffn).total_milliseconds
      }

      read_outputs = -> {
        ML::CUDA.copy_dtoh!(@conv_state_gpu.to_unsafe.as(Void*), d_conv_state, bytesize_f32(@conv_state_gpu.size), "conv_state")
        ML::CUDA.copy_dtoh!(@ssm_state_gpu.to_unsafe.as(Void*), d_ssm_state, bytesize_f32(@ssm_state_gpu.size), "ssm_state")
        ML::CUDA.copy_dtoh!(@attn_out_gpu.to_unsafe.as(Void*), d_attn_out, bytesize_f32(@attn_out_gpu.size), "attn_out")
        ML::CUDA.copy_dtoh!(@final_gpu_all.to_unsafe.as(Void*), d_final_all, bytesize_f32(@final_gpu_all.size), "finals")
      }
      @runner = ResidentSequenceRunner.new(@tokens, upload_weights, reset_sequence, run_token, read_outputs, run_sequence_override)
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

    private def q8_pack_bytes(elements : Int32) : LibC::SizeT
      raise ArgumentError.new("Q8_1 quantization requires multiples of 32") unless elements % 32 == 0

      ((elements // 32) * 8 * sizeof(UInt32)).to_u64
    end
  end
end
