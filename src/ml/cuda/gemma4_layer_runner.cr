require "./driver"
require "./gemma4_swa_context_ptx"
require "../gguf/gemma4_weights"

module ML::CUDA
  # Correctness-first CUDA runner for one Gemma4 text layer.
  #
  # It covers both Gemma4 layer classes:
  # - SWA layers with explicit attn_v.weight
  # - full-attention layers where V is derived from the raw K projection
  #
  # This is intentionally probe-grade. It owns all buffers for one layer and
  # exposes device-resident input/output pointers so a mixed-stack probe can
  # validate handoff before we optimize scheduling and scratch reuse.
  class Gemma4LayerRunner
    NORM_PTX = {{ read_file("src/ml/cuda/kernels/deltanet_step_probe.ptx") }}
    Q4K_PTX  = {{ read_file("src/ml/cuda/kernels/q4k_gemv_probe.ptx") }}
    Q6K_PTX  = {{ read_file("src/ml/cuda/kernels/q6k_gemv_probe.ptx") }}

    ROPE_PTX = <<-PTX
.version 8.0
.target sm_80
.address_size 64

.visible .entry rope_neox_apply_batched_probe(
    .param .u64 x,
    .param .u64 cos_t,
    .param .u64 sin_t,
    .param .u32 head_dim,
    .param .u32 n_rot
)
{
    .reg .pred %p<4>;
    .reg .b32 %r<18>;
    .reg .b64 %rd<18>;
    .reg .f32 %f<9>;

    ld.param.u64 %rd1, [x];
    ld.param.u64 %rd2, [cos_t];
    ld.param.u64 %rd3, [sin_t];
    ld.param.u32 %r1, [head_dim];
    ld.param.u32 %r2, [n_rot];

    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %tid.x;
    shr.u32 %r5, %r2, 1;
    setp.ge.u32 %p1, %r4, %r5;
    @%p1 bra DONE;

    mul.lo.u32 %r6, %r3, %r1;
    add.u32 %r7, %r6, %r4;
    add.u32 %r8, %r7, %r5;

    mul.wide.u32 %rd4, %r7, 4;
    add.s64 %rd5, %rd1, %rd4;
    mul.wide.u32 %rd6, %r8, 4;
    add.s64 %rd7, %rd1, %rd6;
    mul.wide.u32 %rd8, %r4, 4;
    add.s64 %rd9, %rd2, %rd8;
    add.s64 %rd10, %rd3, %rd8;

    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd7];
    ld.global.f32 %f3, [%rd9];
    ld.global.f32 %f4, [%rd10];

    mul.rn.f32 %f5, %f1, %f3;
    neg.f32 %f7, %f4;
    fma.rn.f32 %f5, %f2, %f7, %f5;
    mul.rn.f32 %f6, %f1, %f4;
    fma.rn.f32 %f6, %f2, %f3, %f6;
    st.global.f32 [%rd5], %f5;
    st.global.f32 [%rd7], %f6;

DONE:
    ret;
}
PTX

    ELEM_PTX = <<-PTX
.version 8.0
.target sm_80
.address_size 64

.visible .entry gelu_mul_f32(
    .param .u64 gate,
    .param .u64 up,
    .param .u64 out,
    .param .u32 n
)
{
    .reg .pred %p;
    .reg .b32 %r<8>;
    .reg .b64 %rd<12>;
    .reg .f32 %f<24>;

    ld.param.u64 %rd1, [gate];
    ld.param.u64 %rd2, [up];
    ld.param.u64 %rd3, [out];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mad.lo.s32 %r5, %r3, %r4, %r2;
    setp.ge.u32 %p, %r5, %r1;
    @%p bra GELU_DONE;

    mul.wide.u32 %rd4, %r5, 4;
    add.s64 %rd5, %rd1, %rd4;
    add.s64 %rd6, %rd2, %rd4;
    add.s64 %rd7, %rd3, %rd4;
    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];

    mul.rn.f32 %f3, %f1, %f1;
    mov.f32 %f4, 0f3D372713;
    mul.rn.f32 %f5, %f3, %f4;
    add.rn.f32 %f6, %f5, 0f3F800000;
    mul.rn.f32 %f7, %f1, %f6;
    mov.f32 %f8, 0f3F4C422A;
    mul.rn.f32 %f9, %f7, %f8;

    mov.f32 %f10, 0fC0000000;
    mul.rn.f32 %f11, %f9, %f10;
    mov.f32 %f12, 0f3FB8AA3B;
    mul.rn.f32 %f13, %f11, %f12;
    ex2.approx.ftz.f32 %f14, %f13;
    add.rn.f32 %f15, %f14, 0f3F800000;
    rcp.approx.ftz.f32 %f16, %f15;
    mov.f32 %f17, 0f40000000;
    mul.rn.f32 %f18, %f17, %f16;
    add.rn.f32 %f19, %f18, 0fBF800000;

    add.rn.f32 %f20, %f19, 0f3F800000;
    mov.f32 %f21, 0f3F000000;
    mul.rn.f32 %f22, %f21, %f1;
    mul.rn.f32 %f22, %f22, %f20;
    mul.rn.f32 %f23, %f22, %f2;
    st.global.f32 [%rd7], %f23;

GELU_DONE:
    ret;
}

.visible .entry add_f32(
    .param .u64 a,
    .param .u64 b,
    .param .u64 out,
    .param .u32 n
)
{
    .reg .pred %p;
    .reg .b32 %r<8>;
    .reg .b64 %rd<12>;
    .reg .f32 %f<4>;
    ld.param.u64 %rd1, [a];
    ld.param.u64 %rd2, [b];
    ld.param.u64 %rd3, [out];
    ld.param.u32 %r1, [n];
    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mad.lo.s32 %r5, %r3, %r4, %r2;
    setp.ge.u32 %p, %r5, %r1;
    @%p bra ADD_DONE;
    mul.wide.u32 %rd4, %r5, 4;
    add.s64 %rd5, %rd1, %rd4;
    add.s64 %rd6, %rd2, %rd4;
    add.s64 %rd7, %rd3, %rd4;
    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    add.rn.f32 %f3, %f1, %f2;
    st.global.f32 [%rd7], %f3;
ADD_DONE:
    ret;
}

.visible .entry add_scale_f32(
    .param .u64 a,
    .param .u64 b,
    .param .u64 out,
    .param .u32 n,
    .param .f32 scale
)
{
    .reg .pred %p;
    .reg .b32 %r<8>;
    .reg .b64 %rd<12>;
    .reg .f32 %f<5>;
    ld.param.u64 %rd1, [a];
    ld.param.u64 %rd2, [b];
    ld.param.u64 %rd3, [out];
    ld.param.u32 %r1, [n];
    ld.param.f32 %f4, [scale];
    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mad.lo.s32 %r5, %r3, %r4, %r2;
    setp.ge.u32 %p, %r5, %r1;
    @%p bra ADDS_DONE;
    mul.wide.u32 %rd4, %r5, 4;
    add.s64 %rd5, %rd1, %rd4;
    add.s64 %rd6, %rd2, %rd4;
    add.s64 %rd7, %rd3, %rd4;
    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    add.rn.f32 %f3, %f1, %f2;
    mul.rn.f32 %f3, %f3, %f4;
    st.global.f32 [%rd7], %f3;
ADDS_DONE:
    ret;
}
PTX

    getter output_device_ptr : DevicePtr
    getter final_gpu_all : Array(Float32)
    getter layer : Int32

    @hp : ML::GGUF::Gemma4Hparams
    @lw : ML::GGUF::Gemma4LayerWeights
    @upload_weights : Proc(Nil)?
    @reset_sequence : Proc(Nil)?
    @run_sequence : Proc(Nil)?
    @hidden : Int32
    @head_dim : Int32
    @n_head : Int32
    @n_head_kv : Int32
    @heads_per_group : Int32
    @q_dim : Int32
    @kv_dim : Int32
    @ffn_dim : Int32
    @full_attention : Bool
    @sliding_window : Int32
    @modules : Array(CUDAModule)
    @buffers : Array(DeviceBuffer)
    @param_keepalive : Array(Void*)
    @input_device_base : DevicePtr
    @owned_input_device_ptr : DevicePtr
    @closed : Bool

    def initialize(weights : ML::GGUF::Gemma4Weights,
                   @layer : Int32,
                   @tokens : Int32,
                   @max_seq : Int32,
                   @start_pos : Int32,
                   input : Array(Float32))
      @weights = weights
      @hp = weights.hparams
      @lw = weights.layers[@layer]
      @hidden = @hp.n_embd
      @head_dim = @hp.head_dim_for_layer(@layer)
      @n_head = @hp.n_head
      @n_head_kv = @hp.n_head_kv(@layer)
      @heads_per_group = @n_head // @n_head_kv
      @q_dim = @n_head * @head_dim
      @kv_dim = @n_head_kv * @head_dim
      @ffn_dim = @lw.ffn_gate_qw.out_dim
      @full_attention = @hp.full_attention?(@layer)
      @sliding_window = @full_attention ? @max_seq : @hp.sliding_window
      raise ArgumentError.new("input size mismatch") unless input.size == @tokens * @hidden
      raise ArgumentError.new("unsupported V layout") if !@full_attention && @lw.attn_v_qw.nil?

      @input = input
      @modules = [] of CUDAModule
      @buffers = [] of DeviceBuffer
      @param_keepalive = [] of Void*
      @input_device_base = 0_u64
      @owned_input_device_ptr = 0_u64
      @output_device_ptr = 0_u64
      @final_gpu_all = Array(Float32).new(@tokens * @hidden, 0.0_f32)
      @closed = false

      build
    end

    def use_device_sequence_input(ptr : DevicePtr) : Nil
      raise ArgumentError.new("device input pointer must be non-zero") if ptr == 0_u64
      @input_device_base = ptr
    end

    def upload_weights : Nil
      @upload_weights.not_nil!.call
    end

    def reset_sequence : Nil
      @reset_sequence.not_nil!.call
    end

    def run_sequence : Nil
      @run_sequence.not_nil!.call
    end

    def read_outputs : Nil
      ML::CUDA.copy_dtoh!(@final_gpu_all.to_unsafe.as(Void*), @output_device_ptr, bytesize_f32(@final_gpu_all.size), "gemma layer final")
    end

    def close : Nil
      return if @closed
      @buffers.each(&.close)
      @modules.each(&.close)
      @closed = true
    end

    private def build : Nil
      norm_mod = CUDAModule.load(NORM_PTX, "gemma4_layer_norm")
      rope_mod = CUDAModule.load(ROPE_PTX, "gemma4_layer_rope")
      attn_mod = CUDAModule.load(Gemma4SWAContextPTX::ATTN_PTX, "gemma4_layer_attn")
      kv_mod = CUDAModule.load(Gemma4SWAContextPTX::KV_WRITE_PTX, "gemma4_layer_kv")
      q4_mod = CUDAModule.load(Q4K_PTX, "gemma4_layer_q4")
      q6_mod = CUDAModule.load(Q6K_PTX, "gemma4_layer_q6")
      elem_mod = CUDAModule.load(ELEM_PTX, "gemma4_layer_elem")
      @modules.concat([norm_mod, rope_mod, attn_mod, kv_mod, q4_mod, q6_mod, elem_mod])

      norm_fn = norm_mod.function("rmsnorm_vec_parallel_batched_probe")
      rope_fn = rope_mod.function("rope_neox_apply_batched_probe")
      kv_write_fn = kv_mod.function("gemma4_kv_cache_write_probe")
      splitk_part_fn = attn_mod.function("gemma4_swa_ungated_attn_splitk_part_probe")
      splitk_reduce_fn = attn_mod.function("gemma4_swa_ungated_attn_splitk_reduce_probe")
      q4_fn = q4_mod.function("q4_k_gemv_warp4_f32")
      q4_batched_fn = q4_mod.function("q4_k_gemv_warp4_f32_batched")
      q6_fn = q6_mod.function("q6_k_gemv_warp4_f32")
      q6_batched_fn = q6_mod.function("q6_k_gemv_warp4_f32_batched")
      gelu_fn = elem_mod.function("gelu_mul_f32")
      add_fn = elem_mod.function("add_f32")
      add_scale_fn = elem_mod.function("add_scale_f32")

      v_qw = @lw.attn_v_qw
      out_fn = @lw.attn_output_qw.type.q4_k? ? q4_fn : q6_fn
      ffn_down_fn = @lw.ffn_down_qw.type.q4_k? ? q4_fn : q6_fn
      v_batched_fn = v_qw && v_qw.not_nil!.type.q6_k? ? q6_batched_fn : q4_batched_fn

      d_input = alloc_f32(@tokens * @hidden)
      @owned_input_device_ptr = d_input
      @input_device_base = d_input
      d_input_norm = alloc_f32(@hidden)
      d_x_norm = alloc_f32(@tokens * @hidden)
      d_q_w = alloc_bytes(@lw.attn_q_qw.raw.size)
      d_k_w = alloc_bytes(@lw.attn_k_qw.raw.size)
      d_v_w = @full_attention ? 0_u64 : alloc_bytes(v_qw.not_nil!.raw.size)
      d_q_raw = alloc_f32(@tokens * @q_dim)
      d_k_raw = alloc_f32(@tokens * @kv_dim)
      d_q = alloc_f32(@tokens * @q_dim)
      d_k = alloc_f32(@tokens * @kv_dim)
      d_v = alloc_f32(@tokens * @kv_dim)
      d_q_norm = alloc_f32(@lw.attn_q_norm.size)
      d_k_norm = alloc_f32(@lw.attn_k_norm.size)
      d_ones = alloc_f32(@head_dim)
      d_cos = alloc_f32(@hp.rope_dim_for_layer(@layer) // 2)
      d_sin = alloc_f32(@hp.rope_dim_for_layer(@layer) // 2)
      d_k_cache = alloc_f32(@max_seq * @kv_dim)
      d_v_cache = alloc_f32(@max_seq * @kv_dim)
      d_scores = alloc_f32(@tokens * @n_head * @max_seq)
      d_ctx = alloc_f32(@tokens * @q_dim)
      splitk_chunks = 1
      d_splitk_m = alloc_f32(@tokens * @n_head * splitk_chunks)
      d_splitk_l = alloc_f32(@tokens * @n_head * splitk_chunks)
      d_splitk_o = alloc_f32(@tokens * @n_head * splitk_chunks * @head_dim)
      d_start = alloc_bytes(sizeof(UInt32))
      d_out_w = alloc_bytes(@lw.attn_output_qw.raw.size)
      d_attn_projected = alloc_f32(@tokens * @hidden)
      d_post_attn_norm = alloc_f32(@hidden)
      d_ffn_norm = alloc_f32(@hidden)
      d_post_ffw_norm = alloc_f32(@hidden)
      d_attn_normed = alloc_f32(@tokens * @hidden)
      d_attn_residual = alloc_f32(@tokens * @hidden)
      d_ffn_in = alloc_f32(@tokens * @hidden)
      d_ffn_gate_w = alloc_bytes(@lw.ffn_gate_qw.raw.size)
      d_ffn_up_w = alloc_bytes(@lw.ffn_up_qw.raw.size)
      d_ffn_down_w = alloc_bytes(@lw.ffn_down_qw.raw.size)
      d_ffn_gate = alloc_f32(@tokens * @ffn_dim)
      d_ffn_up = alloc_f32(@tokens * @ffn_dim)
      d_ffn_comb = alloc_f32(@tokens * @ffn_dim)
      d_ffn_raw = alloc_f32(@tokens * @hidden)
      d_ffn_normed = alloc_f32(@tokens * @hidden)
      d_out = alloc_f32(@tokens * @hidden)
      @output_device_ptr = d_out

      @upload_weights = -> {
        ML::CUDA.copy_htod!(d_input_norm, @lw.attn_norm.to_unsafe.as(Void*), bytesize_f32(@lw.attn_norm.size), "gemma input norm")
        ML::CUDA.copy_htod!(d_q_w, @lw.attn_q_qw.raw.to_unsafe.as(Void*), @lw.attn_q_qw.raw.size.to_u64, "gemma q w")
        ML::CUDA.copy_htod!(d_k_w, @lw.attn_k_qw.raw.to_unsafe.as(Void*), @lw.attn_k_qw.raw.size.to_u64, "gemma k w")
        ML::CUDA.copy_htod!(d_v_w, v_qw.not_nil!.raw.to_unsafe.as(Void*), v_qw.not_nil!.raw.size.to_u64, "gemma v w") unless @full_attention
        ML::CUDA.copy_htod!(d_q_norm, @lw.attn_q_norm.to_unsafe.as(Void*), bytesize_f32(@lw.attn_q_norm.size), "gemma q norm")
        ML::CUDA.copy_htod!(d_k_norm, @lw.attn_k_norm.to_unsafe.as(Void*), bytesize_f32(@lw.attn_k_norm.size), "gemma k norm")
        ones = Array(Float32).new(@head_dim, 1.0_f32)
        ML::CUDA.copy_htod!(d_ones, ones.to_unsafe.as(Void*), bytesize_f32(ones.size), "gemma ones")
        ML::CUDA.copy_htod!(d_out_w, @lw.attn_output_qw.raw.to_unsafe.as(Void*), @lw.attn_output_qw.raw.size.to_u64, "gemma out w")
        ML::CUDA.copy_htod!(d_post_attn_norm, @lw.post_attention_norm.to_unsafe.as(Void*), bytesize_f32(@lw.post_attention_norm.size), "gemma post attn norm")
        ML::CUDA.copy_htod!(d_ffn_norm, @lw.ffn_norm.to_unsafe.as(Void*), bytesize_f32(@lw.ffn_norm.size), "gemma ffn norm")
        ML::CUDA.copy_htod!(d_post_ffw_norm, @lw.post_ffw_norm.to_unsafe.as(Void*), bytesize_f32(@lw.post_ffw_norm.size), "gemma post ffw norm")
        ML::CUDA.copy_htod!(d_ffn_gate_w, @lw.ffn_gate_qw.raw.to_unsafe.as(Void*), @lw.ffn_gate_qw.raw.size.to_u64, "gemma ffn gate w")
        ML::CUDA.copy_htod!(d_ffn_up_w, @lw.ffn_up_qw.raw.to_unsafe.as(Void*), @lw.ffn_up_qw.raw.size.to_u64, "gemma ffn up w")
        ML::CUDA.copy_htod!(d_ffn_down_w, @lw.ffn_down_qw.raw.to_unsafe.as(Void*), @lw.ffn_down_qw.raw.size.to_u64, "gemma ffn down w")
        start_value = @start_pos.to_u32
        ML::CUDA.copy_htod!(d_start, pointerof(start_value).as(Void*), sizeof(UInt32).to_u64, "gemma start")
      }

      @reset_sequence = -> {
        if @input_device_base == d_input
          ML::CUDA.copy_htod!(d_input, @input.to_unsafe.as(Void*), bytesize_f32(@input.size), "gemma input")
        end
      }

      @run_sequence = -> {
        launch_norm(norm_fn, @input_device_base, d_input_norm, d_x_norm, @tokens, @hidden, @hp.rms_eps, "gemma input norm")
        launch_q_batched(q4_batched_fn, d_q_w, d_x_norm, d_q_raw, @tokens, @hidden, @q_dim, "gemma q")
        launch_q_batched(q4_batched_fn, d_k_w, d_x_norm, d_k_raw, @tokens, @hidden, @kv_dim, "gemma k")
        if @full_attention
          launch_norm(norm_fn, d_k_raw, d_ones, d_v, @tokens * @n_head_kv, @head_dim, @hp.rms_eps, "gemma v from k")
        else
          launch_q_batched(v_batched_fn, d_v_w, d_x_norm, d_v, @tokens, @hidden, @kv_dim, "gemma v")
          launch_norm(norm_fn, d_v, d_ones, d_v, @tokens * @n_head_kv, @head_dim, @hp.rms_eps, "gemma v norm")
        end
        launch_norm(norm_fn, d_q_raw, d_q_norm, d_q, @tokens * @n_head, @head_dim, @hp.rms_eps, "gemma q norm")
        launch_norm(norm_fn, d_k_raw, d_k_norm, d_k, @tokens * @n_head_kv, @head_dim, @hp.rms_eps, "gemma k norm")

        @tokens.times do |tok|
          abs_pos = @start_pos + tok
          cos, sin = rope_tables(abs_pos)
          ML::CUDA.copy_htod!(d_cos, cos.to_unsafe.as(Void*), bytesize_f32(cos.size), "gemma rope cos")
          ML::CUDA.copy_htod!(d_sin, sin.to_unsafe.as(Void*), bytesize_f32(sin.size), "gemma rope sin")
          launch_rope(rope_fn, d_q + bytesize_f32(tok * @q_dim), d_cos, d_sin, @n_head, @head_dim, @hp.rope_dim_for_layer(@layer), "gemma q rope")
          launch_rope(rope_fn, d_k + bytesize_f32(tok * @kv_dim), d_cos, d_sin, @n_head_kv, @head_dim, @hp.rope_dim_for_layer(@layer), "gemma k rope")
        end

        launch_kv_write(kv_write_fn, d_k, d_v, d_k_cache, d_v_cache)
        launch_attn(splitk_part_fn, splitk_reduce_fn, d_q, d_k_cache, d_v_cache, d_scores, d_splitk_m, d_splitk_l, d_splitk_o, d_ctx, d_start, splitk_chunks)
        @tokens.times do |tok|
          launch_gemv(out_fn, d_out_w, d_ctx + bytesize_f32(tok * @q_dim), d_attn_projected + bytesize_f32(tok * @hidden), @q_dim, @hidden, "gemma attn out")
          launch_norm(norm_fn, d_attn_projected + bytesize_f32(tok * @hidden), d_post_attn_norm, d_attn_normed + bytesize_f32(tok * @hidden), 1, @hidden, @hp.rms_eps, "gemma post attn norm")
          launch_add(add_fn, @input_device_base + bytesize_f32(tok * @hidden), d_attn_normed + bytesize_f32(tok * @hidden), d_attn_residual + bytesize_f32(tok * @hidden), @hidden, "gemma attn residual")
          launch_norm(norm_fn, d_attn_residual + bytesize_f32(tok * @hidden), d_ffn_norm, d_ffn_in + bytesize_f32(tok * @hidden), 1, @hidden, @hp.rms_eps, "gemma ffn norm")
          launch_gemv(q4_fn, d_ffn_gate_w, d_ffn_in + bytesize_f32(tok * @hidden), d_ffn_gate + bytesize_f32(tok * @ffn_dim), @hidden, @ffn_dim, "gemma ffn gate")
          launch_gemv(q4_fn, d_ffn_up_w, d_ffn_in + bytesize_f32(tok * @hidden), d_ffn_up + bytesize_f32(tok * @ffn_dim), @hidden, @ffn_dim, "gemma ffn up")
        end
        launch_gelu(gelu_fn, d_ffn_gate, d_ffn_up, d_ffn_comb, @tokens * @ffn_dim, "gemma gelu")
        @tokens.times do |tok|
          launch_gemv(ffn_down_fn, d_ffn_down_w, d_ffn_comb + bytesize_f32(tok * @ffn_dim), d_ffn_raw + bytesize_f32(tok * @hidden), @ffn_dim, @hidden, "gemma ffn down")
          launch_norm(norm_fn, d_ffn_raw + bytesize_f32(tok * @hidden), d_post_ffw_norm, d_ffn_normed + bytesize_f32(tok * @hidden), 1, @hidden, @hp.rms_eps, "gemma post ffw norm")
          layer_scale = @lw.layer_output_scale.first? || 1.0_f32
          launch_add_scale(add_scale_fn, d_attn_residual + bytesize_f32(tok * @hidden), d_ffn_normed + bytesize_f32(tok * @hidden), d_out + bytesize_f32(tok * @hidden), @hidden, layer_scale, "gemma layer out")
        end
      }
      @closed = false
    end

    private def rope_tables(pos : Int32) : {Array(Float32), Array(Float32)}
      n_rot = @hp.rope_dim_for_layer(@layer)
      base = @hp.rope_freq_base_for_layer(@layer)
      factors = @full_attention ? @weights.rope_freqs : nil
      half = n_rot // 2
      cos = Array(Float32).new(half, 0.0_f32)
      sin = Array(Float32).new(half, 0.0_f32)
      half.times do |i|
        i0 = 2 * i
        factor = factors ? factors.not_nil![i] : 1.0_f32
        theta = pos.to_f32 * (base ** (-i0.to_f32 / n_rot.to_f32)) / factor
        cos[i] = Math.cos(theta).to_f32
        sin[i] = Math.sin(theta).to_f32
      end
      {cos, sin}
    end

    private def alloc_f32(elements : Int32) : DevicePtr
      alloc_bytes((elements * sizeof(Float32)).to_u64)
    end

    private def alloc_bytes(bytes : Int | UInt64) : DevicePtr
      buf = DeviceBuffer.new(bytes.to_u64)
      @buffers << buf
      buf.ptr
    end

    private def bytesize_f32(elements : Int32) : LibC::SizeT
      (elements * sizeof(Float32)).to_u64
    end

    private def launch_norm(fn, x, norm, out_ptr, rows, dim, eps, label)
      d_x = x; d_norm = norm; d_out = out_ptr; dim_u32 = dim.to_u32; eps_f32 = eps
      params = Pointer(Void*).malloc(5)
      params[0] = pointerof(d_x).as(Void*); params[1] = pointerof(d_norm).as(Void*); params[2] = pointerof(d_out).as(Void*)
      params[3] = pointerof(dim_u32).as(Void*); params[4] = pointerof(eps_f32).as(Void*)
      ML::CUDA.launch!(fn, rows.to_u32, 1_u32, 1_u32, 256_u32, 1_u32, 1_u32, params, label)
    end

    private def launch_q_batched(fn, w, x, out_ptr, tokens, in_dim, out_dim, label)
      d_w = w; d_x = x; d_out = out_ptr; in_dim_u32 = in_dim.to_u32; out_dim_u32 = out_dim.to_u32
      params = Pointer(Void*).malloc(5)
      params[0] = pointerof(d_w).as(Void*); params[1] = pointerof(d_x).as(Void*); params[2] = pointerof(d_out).as(Void*)
      params[3] = pointerof(in_dim_u32).as(Void*); params[4] = pointerof(out_dim_u32).as(Void*)
      grid = (((out_dim + 3) // 4) * tokens).to_u32
      ML::CUDA.launch!(fn, grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, params, label)
    end

    private def launch_gemv(fn, w, x, out_ptr, in_dim, out_dim, label)
      d_w = w; d_x = x; d_out = out_ptr; in_dim_u32 = in_dim.to_u32; out_dim_u32 = out_dim.to_u32
      params = Pointer(Void*).malloc(5)
      params[0] = pointerof(d_w).as(Void*); params[1] = pointerof(d_x).as(Void*); params[2] = pointerof(d_out).as(Void*)
      params[3] = pointerof(in_dim_u32).as(Void*); params[4] = pointerof(out_dim_u32).as(Void*)
      ML::CUDA.launch!(fn, ((out_dim + 3) // 4).to_u32, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, params, label)
    end

    private def launch_rope(fn, x, cos, sin, rows, head_dim, n_rot, label)
      d_x = x; d_cos = cos; d_sin = sin; head_dim_u32 = head_dim.to_u32; n_rot_u32 = n_rot.to_u32
      params = Pointer(Void*).malloc(5)
      params[0] = pointerof(d_x).as(Void*); params[1] = pointerof(d_cos).as(Void*); params[2] = pointerof(d_sin).as(Void*)
      params[3] = pointerof(head_dim_u32).as(Void*); params[4] = pointerof(n_rot_u32).as(Void*)
      block = {n_rot // 2, 256}.min.to_u32
      ML::CUDA.launch!(fn, rows.to_u32, 1_u32, 1_u32, block, 1_u32, 1_u32, params, label)
    end

    private def launch_kv_write(fn, k, v, kc, vc)
      d_k = k; d_v = v; d_kc = kc; d_vc = vc; kv_dim_u32 = @kv_dim.to_u32; start_u32 = @start_pos.to_u32
      params = Pointer(Void*).malloc(6)
      params[0] = pointerof(d_k).as(Void*); params[1] = pointerof(d_v).as(Void*); params[2] = pointerof(d_kc).as(Void*)
      params[3] = pointerof(d_vc).as(Void*); params[4] = pointerof(kv_dim_u32).as(Void*); params[5] = pointerof(start_u32).as(Void*)
      ML::CUDA.launch!(fn, @tokens.to_u32, 1_u32, 1_u32, 256_u32, 1_u32, 1_u32, params, "gemma kv write")
    end

    private def launch_attn(part_fn, reduce_fn, q, kc, vc, scores, m, l, o, ctx, start_ptr, chunks)
      d_q = q; d_kc = kc; d_vc = vc; d_scores = scores; d_m = m; d_l = l; d_o = o; d_ctx = ctx; d_start = start_ptr
      n_head_u32 = @n_head.to_u32; n_head_kv_u32 = @n_head_kv.to_u32; head_dim_u32 = @head_dim.to_u32
      hpg_u32 = @heads_per_group.to_u32; max_seq_u32 = @max_seq.to_u32; window_u32 = @sliding_window.to_u32
      scale = 1.0_f32; chunk_u32 = @sliding_window.to_u32; chunks_u32 = chunks.to_u32
      params = Pointer(Void*).malloc(17)
      params[0] = pointerof(d_q).as(Void*); params[1] = pointerof(d_kc).as(Void*); params[2] = pointerof(d_vc).as(Void*)
      params[3] = pointerof(d_scores).as(Void*); params[4] = pointerof(d_m).as(Void*); params[5] = pointerof(d_l).as(Void*); params[6] = pointerof(d_o).as(Void*)
      params[7] = pointerof(n_head_u32).as(Void*); params[8] = pointerof(n_head_kv_u32).as(Void*); params[9] = pointerof(head_dim_u32).as(Void*)
      params[10] = pointerof(hpg_u32).as(Void*); params[11] = pointerof(d_start).as(Void*); params[12] = pointerof(max_seq_u32).as(Void*)
      params[13] = pointerof(window_u32).as(Void*); params[14] = pointerof(scale).as(Void*); params[15] = pointerof(chunk_u32).as(Void*); params[16] = pointerof(chunks_u32).as(Void*)
      ML::CUDA.launch!(part_fn, @tokens.to_u32, @n_head.to_u32, chunks_u32, 256_u32, 1_u32, 1_u32, params, "gemma attn part")
      rparams = Pointer(Void*).malloc(9)
      rparams[0] = pointerof(d_m).as(Void*); rparams[1] = pointerof(d_l).as(Void*); rparams[2] = pointerof(d_o).as(Void*)
      rparams[3] = pointerof(d_ctx).as(Void*); rparams[4] = pointerof(n_head_u32).as(Void*); rparams[5] = pointerof(head_dim_u32).as(Void*)
      rparams[6] = pointerof(d_start).as(Void*); rparams[7] = pointerof(max_seq_u32).as(Void*); rparams[8] = pointerof(chunks_u32).as(Void*)
      ML::CUDA.launch!(reduce_fn, @tokens.to_u32, @n_head.to_u32, 1_u32, 256_u32, 1_u32, 1_u32, rparams, "gemma attn reduce")
    end

    private def launch_gelu(fn, gate, up, out_ptr, n, label)
      d_gate = gate; d_up = up; d_out = out_ptr; n_u32 = n.to_u32
      params = Pointer(Void*).malloc(4)
      params[0] = pointerof(d_gate).as(Void*); params[1] = pointerof(d_up).as(Void*); params[2] = pointerof(d_out).as(Void*); params[3] = pointerof(n_u32).as(Void*)
      ML::CUDA.launch!(fn, ((n + 255) // 256).to_u32, 1_u32, 1_u32, 256_u32, 1_u32, 1_u32, params, label)
    end

    private def launch_add(fn, a, b, out_ptr, n, label)
      d_a = a; d_b = b; d_out = out_ptr; n_u32 = n.to_u32
      params = Pointer(Void*).malloc(4)
      params[0] = pointerof(d_a).as(Void*); params[1] = pointerof(d_b).as(Void*); params[2] = pointerof(d_out).as(Void*); params[3] = pointerof(n_u32).as(Void*)
      ML::CUDA.launch!(fn, ((n + 255) // 256).to_u32, 1_u32, 1_u32, 256_u32, 1_u32, 1_u32, params, label)
    end

    private def launch_add_scale(fn, a, b, out_ptr, n, scale, label)
      d_a = a; d_b = b; d_out = out_ptr; n_u32 = n.to_u32; scale_f32 = scale
      params = Pointer(Void*).malloc(5)
      params[0] = pointerof(d_a).as(Void*); params[1] = pointerof(d_b).as(Void*); params[2] = pointerof(d_out).as(Void*)
      params[3] = pointerof(n_u32).as(Void*); params[4] = pointerof(scale_f32).as(Void*)
      ML::CUDA.launch!(fn, ((n + 255) // 256).to_u32, 1_u32, 1_u32, 256_u32, 1_u32, 1_u32, params, label)
    end
  end
end
