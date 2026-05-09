require "./driver"
require "../gguf/reader"

module ML::CUDA
  # Correctness-first resident CUDA runner for one Qwen recurrent layer.
  #
  # The runner owns layer-local device buffers, CUDA modules, kernel functions,
  # and launch parameter blocks. GGUF loading and CPU-reference checks stay
  # outside so the extraction remains a narrow backend boundary.
  class QwenRecurrentLayerRunner
    DN_PTX  = {{ read_file("src/ml/cuda/kernels/deltanet_step_probe.ptx") }}
    Q4K_PTX = {{ read_file("src/ml/cuda/kernels/q4k_gemv_probe.ptx") }}
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
        raise "expected Q4_K ffn gate/up and Q6_K ffn down" unless ffn_gate_info.type.q4_k? && ffn_up_info.type.q4_k? && ffn_down_info.type.q6_k?

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
          ffn_down_raw, conv1d, dt_bias, ssm_a, ssm_norm)
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
        weights.ffn_down_raw, weights.conv1d, weights.dt_bias, weights.ssm_a,
        weights.ssm_norm)
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
      @closed = false

      build_runner
    end

    def upload_weights : Nil
      runner.upload_weights
    end

    def reset_sequence : Nil
      runner.reset_sequence
    end

    def run_sequence : Nil
      runner.run_sequence
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
      @modules.each(&.close)
      @closed = true
    end

    private def build_runner : Nil
      dn_mod = CUDAModule.load(DN_PTX, "delta")
      q4_mod = CUDAModule.load(Q4K_PTX, "q4")
      q5_mod = CUDAModule.load(Q5K_PTX, "q5")
      q6_mod = CUDAModule.load(Q6K_PTX, "q6")
      @modules.concat([dn_mod, q4_mod, q5_mod, q6_mod])

      attn_norm_fn = dn_mod.function("rmsnorm_vec_probe")
      add_rmsnorm_fn = dn_mod.function("add_rmsnorm_vec_probe")
      swiglu_fn = dn_mod.function("swiglu_probe")
      add_fn = dn_mod.function("add_vec_probe")
      conv_fn = dn_mod.function("recurrent_conv1d_silu_step_probe")
      norm_fn = dn_mod.function("l2_norm_128_probe")
      ab_fn = dn_mod.function("alpha_beta_transform_probe")
      dn_fn = dn_mod.function("deltanet_step_128_probe")
      post_fn = dn_mod.function("deltanet_post_norm_gate_128_probe")
      q4_fn = q4_mod.function("q4_k_gemv_warp4_f32")
      q5_fn = q5_mod.function("q5_k_gemv_warp4_f32")
      q6_fn = q6_mod.function("q6_k_gemv_warp4_f32")

      sizes = [bytesize_f32(@tokens * @hidden), bytesize_f32(@hidden), bytesize_f32(@hidden),
               @qkv_raw.size.to_u64, @gate_raw.size.to_u64, @alpha_raw.size.to_u64, @beta_raw_w.size.to_u64,
               bytesize_f32(@conv_state_init.size), bytesize_f32(@ssm_state_init.size), bytesize_f32(@qkv_dim), bytesize_f32(@conv1d.size),
               bytesize_f32(@qkv_dim), bytesize_f32(@h_v), bytesize_f32(@h_v), bytesize_f32(@dt_bias.size), bytesize_f32(@ssm_a.size),
               bytesize_f32(@h_v), bytesize_f32(@h_v), bytesize_f32(@inner_dim), bytesize_f32(@ssm_norm.size), @out_raw.size.to_u64,
               bytesize_f32(@hidden), bytesize_f32(@hidden), bytesize_f32(@hidden), bytesize_f32(@hidden),
               @ffn_gate_raw.size.to_u64, @ffn_up_raw.size.to_u64, @ffn_down_raw.size.to_u64,
               bytesize_f32(@ffn_dim), bytesize_f32(@ffn_dim), bytesize_f32(@ffn_dim), bytesize_f32(@hidden), bytesize_f32(@tokens * @hidden)]
      ptrs = sizes.map do |size_bytes|
        buffer = DeviceBuffer.new(size_bytes)
        @buffers << buffer
        buffer.ptr
      end

      d_xs, d_attn_norm_w, d_cur, d_qkv_w, d_gate_w, d_alpha_w, d_beta_w, d_conv_state, d_ssm_state, d_qkv, d_conv_w, d_conv_out, d_alpha, d_beta_raw, d_dt, d_a, d_g, d_b, d_z, d_norm, d_out_w, d_attn_out, d_post_norm_w, d_residual, d_cur2, d_ffn_gate_w, d_ffn_up_w, d_ffn_down_w, d_ffn_gate, d_ffn_up, d_ffn_comb, d_ffn_out, d_final_all = ptrs

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
      }

      reset_sequence = -> {
        ML::CUDA.copy_htod!(d_xs, @xs.to_unsafe.as(Void*), bytesize_f32(@tokens * @hidden), "xs")
        ML::CUDA.copy_htod!(d_conv_state, @conv_state_init.to_unsafe.as(Void*), bytesize_f32(@conv_state_init.size), "conv_state")
        ML::CUDA.copy_htod!(d_ssm_state, @ssm_state_init.to_unsafe.as(Void*), bytesize_f32(@ssm_state_init.size), "ssm_state")
      }

      hidden_u32 = @hidden.to_u32
      ffn_dim_u32 = @ffn_dim.to_u32
      qkv_dim_u32 = @qkv_dim.to_u32
      h_k_u32 = @h_k.to_u32
      h_v_u32 = @h_v.to_u32
      inner_u32 = @inner_dim.to_u32
      qkv_grid = ((@qkv_dim + 3) // 4).to_u32
      inner_grid = ((@inner_dim + 3) // 4).to_u32
      h_v_grid = ((@h_v + 3) // 4).to_u32
      ffn_grid = ((@ffn_dim + 3) // 4).to_u32
      hidden_grid = ((@hidden + 3) // 4).to_u32
      swiglu_grid = ((@ffn_dim + 127) // 128).to_u32
      add_grid = ((@hidden + 127) // 128).to_u32
      conv_grid = ((@qkv_dim + 127) // 128).to_u32
      d_q = d_conv_out
      d_k = d_conv_out + bytesize_f32(@q_dim)
      d_v = d_conv_out + bytesize_f32(2 * @q_dim)
      d_x_cur_ptr = box_ptr(d_xs)
      d_final_cur_ptr = box_ptr(d_final_all)

      attn_norm_params = Pointer(Void*).malloc(5)
      attn_norm_params[0] = d_x_cur_ptr.as(Void*)
      attn_norm_params[1] = box_ptr(d_attn_norm_w).as(Void*)
      attn_norm_params[2] = box_ptr(d_cur).as(Void*)
      attn_norm_params[3] = box_u32(hidden_u32).as(Void*)
      attn_norm_params[4] = box_f32(@eps).as(Void*)

      qkv_proj_params = Pointer(Void*).malloc(5)
      qkv_proj_params[0] = box_ptr(d_qkv_w).as(Void*)
      qkv_proj_params[1] = box_ptr(d_cur).as(Void*)
      qkv_proj_params[2] = box_ptr(d_qkv).as(Void*)
      qkv_proj_params[3] = box_u32(hidden_u32).as(Void*)
      qkv_proj_params[4] = box_u32(qkv_dim_u32).as(Void*)

      gate_proj_params = Pointer(Void*).malloc(5)
      gate_proj_params[0] = box_ptr(d_gate_w).as(Void*)
      gate_proj_params[1] = box_ptr(d_cur).as(Void*)
      gate_proj_params[2] = box_ptr(d_z).as(Void*)
      gate_proj_params[3] = box_u32(hidden_u32).as(Void*)
      gate_proj_params[4] = box_u32(inner_u32).as(Void*)

      alpha_proj_params = Pointer(Void*).malloc(5)
      alpha_proj_params[0] = box_ptr(d_alpha_w).as(Void*)
      alpha_proj_params[1] = box_ptr(d_cur).as(Void*)
      alpha_proj_params[2] = box_ptr(d_alpha).as(Void*)
      alpha_proj_params[3] = box_u32(hidden_u32).as(Void*)
      alpha_proj_params[4] = box_u32(h_v_u32).as(Void*)

      beta_proj_params = Pointer(Void*).malloc(5)
      beta_proj_params[0] = box_ptr(d_beta_w).as(Void*)
      beta_proj_params[1] = box_ptr(d_cur).as(Void*)
      beta_proj_params[2] = box_ptr(d_beta_raw).as(Void*)
      beta_proj_params[3] = box_u32(hidden_u32).as(Void*)
      beta_proj_params[4] = box_u32(h_v_u32).as(Void*)

      conv_params = Pointer(Void*).malloc(5)
      conv_params[0] = box_ptr(d_conv_state).as(Void*)
      conv_params[1] = box_ptr(d_qkv).as(Void*)
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
      ab_params[0] = box_ptr(d_alpha).as(Void*)
      ab_params[1] = box_ptr(d_beta_raw).as(Void*)
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
      post_params[1] = box_ptr(d_z).as(Void*)
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
      add_rms_params[3] = box_ptr(d_residual).as(Void*)
      add_rms_params[4] = box_ptr(d_cur2).as(Void*)
      add_rms_params[5] = box_u32(hidden_u32).as(Void*)
      add_rms_params[6] = box_f32(@eps).as(Void*)

      ffn_gate_params = Pointer(Void*).malloc(5)
      ffn_gate_params[0] = box_ptr(d_ffn_gate_w).as(Void*)
      ffn_gate_params[1] = box_ptr(d_cur2).as(Void*)
      ffn_gate_params[2] = box_ptr(d_ffn_gate).as(Void*)
      ffn_gate_params[3] = box_u32(hidden_u32).as(Void*)
      ffn_gate_params[4] = box_u32(ffn_dim_u32).as(Void*)

      ffn_up_params = Pointer(Void*).malloc(5)
      ffn_up_params[0] = box_ptr(d_ffn_up_w).as(Void*)
      ffn_up_params[1] = box_ptr(d_cur2).as(Void*)
      ffn_up_params[2] = box_ptr(d_ffn_up).as(Void*)
      ffn_up_params[3] = box_u32(hidden_u32).as(Void*)
      ffn_up_params[4] = box_u32(ffn_dim_u32).as(Void*)

      swiglu_params = Pointer(Void*).malloc(4)
      swiglu_params[0] = box_ptr(d_ffn_gate).as(Void*)
      swiglu_params[1] = box_ptr(d_ffn_up).as(Void*)
      swiglu_params[2] = box_ptr(d_ffn_comb).as(Void*)
      swiglu_params[3] = box_u32(ffn_dim_u32).as(Void*)

      ffn_down_params = Pointer(Void*).malloc(5)
      ffn_down_params[0] = box_ptr(d_ffn_down_w).as(Void*)
      ffn_down_params[1] = box_ptr(d_ffn_comb).as(Void*)
      ffn_down_params[2] = box_ptr(d_ffn_out).as(Void*)
      ffn_down_params[3] = box_u32(ffn_dim_u32).as(Void*)
      ffn_down_params[4] = box_u32(hidden_u32).as(Void*)

      final_add_params = Pointer(Void*).malloc(4)
      final_add_params[0] = box_ptr(d_residual).as(Void*)
      final_add_params[1] = box_ptr(d_ffn_out).as(Void*)
      final_add_params[2] = d_final_cur_ptr.as(Void*)
      final_add_params[3] = box_u32(hidden_u32).as(Void*)

      run_token = ->(tok : Int32) {
        offset = bytesize_f32(tok * @hidden)
        d_x_cur_ptr.value = d_xs + offset
        d_final_cur_ptr.value = d_final_all + offset
        ML::CUDA.launch!(attn_norm_fn, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, attn_norm_params, "attn norm")
        ML::CUDA.launch!(q5_fn, qkv_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, qkv_proj_params, "qkv proj")
        ML::CUDA.launch!(q4_fn, inner_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, gate_proj_params, "gate proj")
        ML::CUDA.launch!(q4_fn, h_v_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, alpha_proj_params, "alpha proj")
        ML::CUDA.launch!(q4_fn, h_v_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, beta_proj_params, "beta proj")
        ML::CUDA.launch!(conv_fn, conv_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, conv_params, "conv prep")
        ML::CUDA.launch!(norm_fn, @h_k.to_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, q_norm_params, "q norm")
        ML::CUDA.launch!(norm_fn, @h_k.to_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, k_norm_params, "k norm")
        ML::CUDA.launch!(ab_fn, 1_u32, 1_u32, 1_u32, 32_u32, 1_u32, 1_u32, ab_params, "alpha beta")
        ML::CUDA.launch!(dn_fn, @h_v.to_u32, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, dn_params, "delta step")
        ML::CUDA.launch!(post_fn, @h_v.to_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, post_params, "post gate")
        ML::CUDA.launch!(q4_fn, hidden_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, out_proj_params, "ssm_out")
        ML::CUDA.launch!(add_rmsnorm_fn, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, add_rms_params, "add rmsnorm")
        ML::CUDA.launch!(q4_fn, ffn_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, ffn_gate_params, "ffn gate")
        ML::CUDA.launch!(q4_fn, ffn_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, ffn_up_params, "ffn up")
        ML::CUDA.launch!(swiglu_fn, swiglu_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, swiglu_params, "swiglu")
        ML::CUDA.launch!(q6_fn, hidden_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, ffn_down_params, "ffn down")
        ML::CUDA.launch!(add_fn, add_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, final_add_params, "final add")
      }

      read_outputs = -> {
        ML::CUDA.copy_dtoh!(@conv_state_gpu.to_unsafe.as(Void*), d_conv_state, bytesize_f32(@conv_state_gpu.size), "conv_state")
        ML::CUDA.copy_dtoh!(@ssm_state_gpu.to_unsafe.as(Void*), d_ssm_state, bytesize_f32(@ssm_state_gpu.size), "ssm_state")
        ML::CUDA.copy_dtoh!(@attn_out_gpu.to_unsafe.as(Void*), d_attn_out, bytesize_f32(@attn_out_gpu.size), "attn_out")
        ML::CUDA.copy_dtoh!(@final_gpu_all.to_unsafe.as(Void*), d_final_all, bytesize_f32(@final_gpu_all.size), "finals")
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
