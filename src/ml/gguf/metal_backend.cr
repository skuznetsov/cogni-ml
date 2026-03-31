# Metal GPU backend — all BERT ops on GPU, minimal CPU↔GPU syncs.
# MoE: CPU gating (tiny) + GPU expert matmuls (per-expert dispatch).

{% unless flag?(:cpu_only) %}

require "./compute"
require "../metal/compute_graph"

module ML::GGUF
  BERT_FUSED_SOURCE = {{ read_file("#{__DIR__}/kernels/bert_fp16.metal") }}
  GEMM_SOURCE = {{ read_file("#{__DIR__}/kernels/gemm_q5k.metal") }}
  SIMD_GEMM_SOURCE = {{ read_file("#{__DIR__}/kernels/gemm_simd.metal") }}
  GEMM_MM_SOURCE = {{ read_file("#{__DIR__}/kernels/gemm_mm.metal") }}
  GEMM_MM_F16_SOURCE = {{ read_file("#{__DIR__}/kernels/gemm_mm_f16.metal") }}
  FLASH_ATTN_SOURCE = {{ read_file("#{__DIR__}/kernels/attention_flash.metal") }}

  class GPUWeight
    getter buffer : ML::MetalBuffer       # FP16 pre-dequantized weights [out_dim, in_dim]
    getter bias_buffer : ML::MetalBuffer
    getter type : TensorType
    getter out_dim : Int32
    getter in_dim : Int32
    getter fp16 : Bool                     # true = pre-dequantized FP16

    def initialize(qw : QuantWeight, bias : Array(Float32))
      @type = qw.type; @out_dim = qw.out_dim; @in_dim = qw.in_dim
      @fp16 = false  # FP16 pre-dequant disabled — bandwidth penalty > dequant savings

      @buffer = ML::MetalBuffer.new(qw.raw.size.to_i64)
      @buffer.write_bytes(qw.raw.to_unsafe, qw.raw.size)
      @bias_buffer = ML::MetalBuffer.new(bias.size.to_i64 * 4)
      @bias_buffer.write(bias)
    end
  end

  class GPUWorkspace
    getter hidden : ML::MetalBuffer
    getter hidden2 : ML::MetalBuffer  # double buffer for pipeline overlap
    getter qkv : ML::MetalBuffer
    getter q : ML::MetalBuffer
    getter k : ML::MetalBuffer
    getter v : ML::MetalBuffer
    getter v_t : ML::MetalBuffer
    getter attn_out : ML::MetalBuffer
    getter ffn_mid : ML::MetalBuffer
    getter ffn_out : ML::MetalBuffer
    getter ffn_out_f32 : ML::MetalBuffer
    getter output : ML::MetalBuffer
    getter cos_cache : ML::MetalBuffer
    getter sin_cache : ML::MetalBuffer
    # Pre-allocated MoE batched buffers (contiguous)
    getter moe_input : ML::MetalBuffer    # [max_routing, dim]
    getter moe_mid : ML::MetalBuffer      # [max_routing, ffn_dim]
    getter moe_output : ML::MetalBuffer   # [max_routing, dim]
    getter gate_logits : ML::MetalBuffer  # [max_seq, 8]
    getter routing_ids : ML::MetalBuffer  # [max_seq, k] int32
    getter routing_wts : ML::MetalBuffer  # [max_seq, k] float32
    getter gather_map : ML::MetalBuffer    # [max_routing] int32 (also used as scatter_map)
    getter scatter_wts : ML::MetalBuffer  # [max_routing] float32 — weights for scatter
    getter expert_counts : ML::MetalBuffer  # [8] int32 atomic counters
    getter expert_offsets : ML::MetalBuffer  # [9] int32 prefix sums
    getter expert_mid : ML::MetalBuffer   # [n_experts, max_seq, ffn_dim]
    getter expert_out : ML::MetalBuffer  # [n_experts, max_seq, dim]
    getter dispatch_args : ML::MetalBuffer  # indirect dispatch args for MoE
    getter expert_tg_offs : ML::MetalBuffer  # [9] int32 — TG prefix sums for batched expert GEMM
    getter batched_up_grid : ML::MetalBuffer   # [3] uint32 — indirect dispatch for batched UP
    getter batched_down_grid : ML::MetalBuffer # [3] uint32 — indirect dispatch for batched DOWN

    def initialize(max_seq : Int32, dim : Int32, ffn_dim : Int32, n_heads : Int32, head_dim : Int32,
                   n_experts_used : Int32, rope_cos : Array(Float32), rope_sin : Array(Float32))
      # FP16 intermediate buffers (2 bytes per element)
      @hidden   = ML::MetalBuffer.new(max_seq.to_i64 * dim * 2)
      @hidden2  = ML::MetalBuffer.new(max_seq.to_i64 * dim * 2)
      @qkv      = ML::MetalBuffer.new(max_seq.to_i64 * 3 * dim * 2)
      @q        = ML::MetalBuffer.new(n_heads.to_i64 * max_seq * head_dim * 2)
      @k        = ML::MetalBuffer.new(n_heads.to_i64 * max_seq * head_dim * 2)
      @v        = ML::MetalBuffer.new(n_heads.to_i64 * max_seq * head_dim * 2)
      @v_t      = ML::MetalBuffer.new(n_heads.to_i64 * max_seq * head_dim * 2)
      @attn_out = ML::MetalBuffer.new(max_seq.to_i64 * dim * 2)
      @ffn_mid  = ML::MetalBuffer.new(max_seq.to_i64 * ffn_dim * 2)
      @ffn_out  = ML::MetalBuffer.new(max_seq.to_i64 * dim * 2)
      @ffn_out_f32 = ML::MetalBuffer.new(max_seq.to_i64 * dim * 4)
      @output   = ML::MetalBuffer.new(dim.to_i64 * 4)  # final output stays F32
      @cos_cache = ML::MetalBuffer.new(rope_cos.size.to_i64 * 4); @cos_cache.write(rope_cos)
      @sin_cache = ML::MetalBuffer.new(rope_sin.size.to_i64 * 4); @sin_cache.write(rope_sin)
      # Pre-allocated MoE batched buffers: one contiguous buffer per data type
      # Max tokens routed = max_seq * n_experts_used (each token → n_experts_used experts)
      max_routing = max_seq * n_experts_used
      @moe_input  = ML::MetalBuffer.new(max_routing.to_i64 * dim * 2)      # FP16
      @moe_mid    = ML::MetalBuffer.new(max_routing.to_i64 * ffn_dim * 2)  # FP16
      @moe_output = ML::MetalBuffer.new(max_routing.to_i64 * dim * 2)      # FP16
      # Gate + routing buffers
      @gate_logits = ML::MetalBuffer.new(max_seq.to_i64 * 8 * 4)
      @routing_ids = ML::MetalBuffer.new(max_routing.to_i64 * 4)  # int32
      @routing_wts = ML::MetalBuffer.new(max_routing.to_i64 * 4)  # float32
      @gather_map  = ML::MetalBuffer.new(max_routing.to_i64 * 4)  # int32
      @scatter_wts = ML::MetalBuffer.new(max_routing.to_i64 * 4)  # float32
      @expert_counts = ML::MetalBuffer.new(8_i64 * 4)   # int32
      @expert_offsets = ML::MetalBuffer.new(9_i64 * 4)  # int32 (n_experts + 1)
      # All-experts buffers FP16
      @expert_mid  = ML::MetalBuffer.new(8_i64 * max_seq * ffn_dim * 2)
      @expert_out  = ML::MetalBuffer.new(8_i64 * max_seq * dim * 2)
      # Indirect dispatch args: 8 UP + 8 DOWN + 8 scatter = 24 entries × 12 bytes
      @dispatch_args = ML::MetalBuffer.new(24_i64 * 12)
      @expert_tg_offs = ML::MetalBuffer.new(9_i64 * 4)     # [n_experts+1] int32
      @batched_up_grid = ML::MetalBuffer.new(3_i64 * 4)   # [3] uint32 indirect dispatch
      @batched_down_grid = ML::MetalBuffer.new(3_i64 * 4) # [3] uint32 indirect dispatch
    end
  end

  # F32 → F16 conversion (IEEE 754 half-precision)
  def self.f32_to_f16(f : Float32) : UInt16
    bits = f.unsafe_as(UInt32)
    sign = (bits >> 16) & 0x8000_u32
    exp = ((bits >> 23) & 0xFF).to_i32 - 127 + 15
    mant = (bits >> 13) & 0x03FF_u32
    if exp <= 0
      (sign).to_u16
    elsif exp >= 31
      (sign | 0x7C00_u32).to_u16  # Inf
    else
      (sign | (exp.to_u32 << 10) | mant).to_u16
    end
  end

  class MetalBackend
    include ComputeBackend

    @gpu_weights : Hash(UInt64, GPUWeight)
    @gpu_f32_bufs : Hash(UInt64, ML::MetalBuffer)
    @pipelines : Hash(String, ML::Metal::ComputePipeline)
    @workspace : GPUWorkspace?
    @norm_bufs_cache : Array(Tuple(ML::MetalBuffer, ML::MetalBuffer, ML::MetalBuffer, ML::MetalBuffer))?

    def initialize
      raise "Metal not available" unless ML::Metal::Device.available?
      @gpu_weights = {} of UInt64 => GPUWeight
      @gpu_f32_bufs = {} of UInt64 => ML::MetalBuffer
      @pipelines = {} of String => ML::Metal::ComputePipeline
      %w[qkv_split rope_neox_inplace attention_forward
         layernorm_inplace residual_add weighted_add zero_region mean_pool_l2
         gelu_inplace gate_matmul softmax_topk moe_gather scatter_weighted_add
         moe_weighted_scatter residual_layernorm residual_layernorm_copy
         moe_count_experts moe_prefix_sum moe_build_routing zero_int
         moe_write_dispatch_args scatter_weighted_add_moe
         moe_scatter_atomic f32_to_f16 residual_layernorm_f32
         gate_softmax_topk_count qkv_split_rope
         moe_route_and_dispatch moe_write_batched_args].each do |name|
        @pipelines[name] = ML::Metal::PipelineCache.get(name) {
          ML::Metal::ComputePipeline.new(name, BERT_FUSED_SOURCE)
        }
      end
      # Tiled GEMM kernels (fallback)
      %w[gemm_q5k gemm_q6k].each do |name|
        @pipelines[name] = ML::Metal::PipelineCache.get(name) {
          ML::Metal::ComputePipeline.new(name, GEMM_SOURCE)
        }
      end
      # Flash attention
      @pipelines["attention_flash"] = ML::Metal::PipelineCache.get("attention_flash") {
        ML::Metal::ComputePipeline.new("attention_flash", FLASH_ATTN_SOURCE)
      }
      # simdgroup_matrix flash attention
      matmul_attn_src = {{ read_file("#{__DIR__}/kernels/attention_matmul.metal") }}
      @pipelines["attention_matmul"] = ML::Metal::PipelineCache.get("attention_matmul") {
        ML::Metal::ComputePipeline.new("attention_matmul", matmul_attn_src)
      }
      # SIMD GEMM kernels (scalar path — for small batch or MoE per-expert)
      %w[simd_gemm_q5k simd_gemm_q6k simd_gemm_q5k_moe simd_gemm_q6k_moe].each do |name|
        @pipelines[name] = ML::Metal::PipelineCache.get(name) {
          ML::Metal::ComputePipeline.new(name, SIMD_GEMM_SOURCE)
        }
      end
      # Matrix-matrix GEMM kernels (simdgroup_matrix — for batch > 8)
      %w[simd_mm_q5k simd_mm_q6k simd_mm_q5k_moe simd_mm_q6k_moe
         batched_mm_q5k batched_mm_q6k].each do |name|
        @pipelines[name] = ML::Metal::PipelineCache.get(name) {
          ML::Metal::ComputePipeline.new(name, GEMM_MM_SOURCE)
        }
      end
      # FP16 pre-dequantized GEMM kernels (fastest — no dequant overhead)
      %w[simd_mm_f16 simd_mm_f16_moe].each do |name|
        @pipelines[name] = ML::Metal::PipelineCache.get(name) {
          ML::Metal::ComputePipeline.new(name, GEMM_MM_F16_SOURCE)
        }
      end
    end

    def upload_weight(qw : QuantWeight, bias : Array(Float32)) : Nil
      key = qw.raw.to_unsafe.address
      @gpu_weights[key] ||= GPUWeight.new(qw, bias)
    end

    def upload_f32(data : Array(Float32)) : ML::MetalBuffer
      key = data.to_unsafe.address
      @gpu_f32_bufs[key] ||= begin
        buf = ML::MetalBuffer.new(data.size.to_i64 * 4)
        buf.write(data)
        buf
      end
    end

    def init_workspace(max_seq : Int32, dim : Int32, ffn_dim : Int32, n_heads : Int32,
                       head_dim : Int32, n_experts_used : Int32, rope_cos : Array(Float32), rope_sin : Array(Float32))
      @workspace = GPUWorkspace.new(max_seq, dim, ffn_dim, n_heads, head_dim, n_experts_used, rope_cos, rope_sin)
    end

    private def gw(qw : QuantWeight, bias : Array(Float32)) : GPUWeight
      key = qw.raw.to_unsafe.address
      @gpu_weights[key] ||= GPUWeight.new(qw, bias)
    end

    private def pipe(name : String) : ML::Metal::ComputePipeline
      @pipelines[name]
    end

    SIMD_N_ROWS = 2  # Must match N_ROWS in gemm_simd.metal
    SIMD_NR0    = 2  # Must match NR0 in gemm_simd.metal — output rows per simdgroup

    # Scalar SIMD GEMM (mul_mv) — good for small batch
    private def matmul_kernel(type : TensorType) : String
      type.q5_k? ? "simd_gemm_q5k" : "simd_gemm_q6k"
    end

    # Matrix-matrix GEMM (mul_mm) — simdgroup_matrix, good for batch > 8
    private def matmul_kernel_mm(type : TensorType, fp16 : Bool = false) : String
      fp16 ? "simd_mm_f16" : (type.q5_k? ? "simd_mm_q5k" : "simd_mm_q6k")
    end

    private def matmul_kernel_moe(type : TensorType) : String
      type.q5_k? ? "simd_gemm_q5k_moe" : "simd_gemm_q6k_moe"
    end

    private def matmul_kernel_mm_moe(type : TensorType, fp16 : Bool = false) : String
      fp16 ? "simd_mm_f16_moe" : (type.q5_k? ? "simd_mm_q5k_moe" : "simd_mm_q6k_moe")
    end

    MM_BATCH_THRESHOLD = 8  # Switch to simdgroup_matrix GEMM above this batch size
    MM_NR0 = 64             # Output rows per threadgroup for mm kernel
    MM_NR1 = 32             # Batch elements per threadgroup for mm kernel
    MM_SHMEM = 16384        # Double-buffered threadgroup memory (2×6KB tiles + output temp)

    # Dispatch matmul — auto-selects mm vs mv based on batch size
    private def matmul_dispatch(enc, type : TensorType,
                                 out_dim : Int32, batch : Int32)
      if batch > MM_BATCH_THRESHOLD
        enc.set_pipeline(pipe(matmul_kernel_mm(type)))
        enc.set_threadgroup_memory(MM_SHMEM, 0)
        grid = {(batch + MM_NR1 - 1) // MM_NR1, (out_dim + MM_NR0 - 1) // MM_NR0, 1}
        enc.dispatch_threadgroups(grid, {128, 1, 1})
      else
        enc.set_pipeline(pipe(matmul_kernel(type)))
        grid, tg = matmul_dispatch_tg(out_dim, batch)
        enc.dispatch_threadgroups(grid, tg)
      end
    end

    # Returns {threadgroup_count, threadgroup_size} for scalar SIMD dispatch
    private def matmul_dispatch_tg(out_dim : Int32, batch : Int32) : { {Int32, Int32, Int32}, {Int32, Int32, Int32} }
      rows_per_tg = SIMD_N_ROWS * SIMD_NR0
      tg_count = {(out_dim + rows_per_tg - 1) // rows_per_tg, {batch, 1}.max, 1}
      tg_size = {32, SIMD_N_ROWS, 1}
      {tg_count, tg_size}
    end

    # Add a matmul op to the compute graph (auto-selects mm vs mv)
    private def graph_matmul(g : ML::Metal::ComputeGraph,
                              gw : GPUWeight, x_buf : ML::MetalBuffer, out_buf : ML::MetalBuffer,
                              in_dim : UInt32, out_dim_val : UInt32, batch_val : UInt32, gelu : UInt32,
                              w_offset : Int64 = 0)
      batch_i = batch_val.to_i32; out_dim_i = out_dim_val.to_i32
      if batch_i > MM_BATCH_THRESHOLD
        grid = {(batch_i + MM_NR1 - 1) // MM_NR1, (out_dim_i + MM_NR0 - 1) // MM_NR0, 1}
        g.add_op(pipe(matmul_kernel_mm(gw.type))) do |op|
          op.buffer(gw.buffer, 0, :read, offset: w_offset)
          op.buffer(x_buf, 1, :read)
          op.buffer(gw.bias_buffer, 2, :read)
          op.buffer(out_buf, 3, :write)
          op.value(in_dim, 4); op.value(out_dim_val, 5)
          op.value(batch_val, 6); op.value(gelu, 7)
          op.threadgroup_memory(MM_SHMEM)
          op.dispatch_threadgroups(grid, {128, 1, 1})
        end
      else
        grid, tg = matmul_dispatch_tg(out_dim_i, batch_i)
        g.add_op(pipe(matmul_kernel(gw.type))) do |op|
          op.buffer(gw.buffer, 0, :read, offset: w_offset)
          op.buffer(x_buf, 1, :read)
          op.buffer(gw.bias_buffer, 2, :read)
          op.buffer(out_buf, 3, :write)
          op.value(in_dim, 4); op.value(out_dim_val, 5)
          op.value(batch_val, 6); op.value(gelu, 7)
          op.dispatch_threadgroups(grid, tg)
        end
      end
    end

    def encode_layers(
      hidden_data : Array(Float32), seq_len : Int32,
      layers : Array(NomicBertMoE::LayerWeights),
      dim : Int32, n_heads : Int32, head_dim : Int32, ffn_dim : Int32,
      n_experts : Int32, n_experts_used : Int32, moe_every_n : Int32,
      cpu_ref_per_layer : Array(Array(Float32))? = nil,  # per-layer comparison
    ) : Array(Float32)
      ws = @workspace.not_nil!
      # Write hidden data as FP16
      h_ptr = ws.hidden.contents.as(Pointer(UInt16))
      (seq_len * dim).times { |i| h_ptr[i] = ML::GGUF.f32_to_f16(hidden_data[i]) }

      scale = 1.0_f32 / Math.sqrt(head_dim.to_f32)
      batch = seq_len.to_u32; dim_u = dim.to_u32
      n_heads_u = n_heads.to_u32; head_dim_u = head_dim.to_u32; ffn_dim_u = ffn_dim.to_u32
      no_gelu = 0_u32; yes_gelu = 1_u32

      norm_bufs = @norm_bufs_cache ||= layers.map do |lw|
        n1w = ML::MetalBuffer.new(dim.to_i64 * 4); n1w.write(lw.norm1_w)
        n1b = ML::MetalBuffer.new(dim.to_i64 * 4); n1b.write(lw.norm1_b)
        n2w = ML::MetalBuffer.new(dim.to_i64 * 4); n2w.write(lw.norm2_w)
        n2b = ML::MetalBuffer.new(dim.to_i64 * 4); n2b.write(lw.norm2_b)
        {n1w, n1b, n2w, n2b}
      end

      cmd = ML::Metal::CommandBuffer.new(fast: true)
      cmd.enqueue
      h_bufs = [ws.hidden, ws.hidden2]

      # Build compute graph
      graph = ML::Metal::ComputeGraph.new
      enc = ML::Metal::GraphEncoder.new(graph)
      _t_encode_start = Time.instant

      layers.each_with_index do |lw, layer_idx|
        is_moe = (layer_idx % moe_every_n == 1) && n_experts > 0
        n1w_buf, n1b_buf, n2w_buf, n2b_buf = norm_bufs[layer_idx]
        h_in = h_bufs[layer_idx % 2]
        h_out = h_bufs[(layer_idx + 1) % 2]
        w = ML::Metal::BufferAccess::Write
        rw = ML::Metal::BufferAccess::ReadWrite

        # === Attention ===
        # QKV matmul: h_in → qkv
        qkv_gw = gw(lw.attn_qkv_w, lw.attn_qkv_b)
        enc.set_buffer(qkv_gw.buffer, 0); enc.set_buffer(h_in, 1)
        enc.set_buffer(qkv_gw.bias_buffer, 2); enc.set_buffer(ws.qkv, 3, w)
        enc.set_value(dim_u, 4); enc.set_value(3_u32 * dim_u, 5)
        enc.set_value(batch, 6); enc.set_value(no_gelu, 7)
        matmul_dispatch(enc, qkv_gw.type, 3 * dim, seq_len)
        enc.memory_barrier

        # Fused QKV split + RoPE: 1 dispatch replaces split + 2×rope + 2 barriers
        enc.set_pipeline(pipe("qkv_split_rope"))
        enc.set_buffer(ws.qkv, 0); enc.set_buffer(ws.q, 1, w); enc.set_buffer(ws.k, 2, w)
        enc.set_buffer(ws.v, 3, w); enc.set_buffer(ws.v_t, 4, w)
        enc.set_buffer(ws.cos_cache, 5); enc.set_buffer(ws.sin_cache, 6)
        enc.set_value(batch, 7); enc.set_value(dim_u, 8); enc.set_value(n_heads_u, 9); enc.set_value(head_dim_u, 10)
        enc.dispatch_1d(seq_len * dim, 256)
        enc.memory_barrier

        # Attention
        if seq_len > 32
          q_total = 8 * 2; n_sg = 2
          sh_q = q_total * head_dim * 2; sh_o = q_total * head_dim * 4; sh_s = q_total * 64 * 4
          enc.set_pipeline(pipe("attention_matmul"))
          enc.set_buffer(ws.q, 0); enc.set_buffer(ws.k, 1); enc.set_buffer(ws.v, 2); enc.set_buffer(ws.attn_out, 3, w)
          enc.set_value(batch, 4); enc.set_value(n_heads_u, 5); enc.set_value(head_dim_u, 6); enc.set_value(scale, 7)
          enc.set_threadgroup_memory(sh_q + sh_o + sh_s, 0)
          enc.dispatch_threadgroups({(seq_len + q_total - 1) // q_total, n_heads, 1}, {32, n_sg, 1})
        else
          n_qr = 8
          enc.set_pipeline(pipe("attention_forward"))
          enc.set_buffer(ws.q, 0); enc.set_buffer(ws.k, 1); enc.set_buffer(ws.v_t, 2); enc.set_buffer(ws.attn_out, 3, w)
          enc.set_value(batch, 4); enc.set_value(n_heads_u, 5); enc.set_value(head_dim_u, 6); enc.set_value(scale, 7)
          enc.set_threadgroup_memory(n_qr * seq_len * 4, 0)
          enc.dispatch_threadgroups({n_heads, (seq_len + n_qr - 1) // n_qr, 1}, {32, n_qr, 1})
        end
        enc.memory_barrier

        # Output projection: attn_out → ffn_out
        out_gw = gw(lw.attn_out_w, lw.attn_out_b)
        enc.set_buffer(out_gw.buffer, 0); enc.set_buffer(ws.attn_out, 1)
        enc.set_buffer(out_gw.bias_buffer, 2); enc.set_buffer(ws.ffn_out, 3, w)
        enc.set_value(dim_u, 4); enc.set_value(dim_u, 5); enc.set_value(batch, 6); enc.set_value(no_gelu, 7)
        matmul_dispatch(enc, out_gw.type, dim, seq_len)
        enc.memory_barrier

        # Norm1: h_out = layernorm(h_in + ffn_out)
        enc.set_pipeline(pipe("residual_layernorm_copy"))
        enc.set_buffer(h_in, 0); enc.set_buffer(ws.ffn_out, 1); enc.set_buffer(h_out, 2, w)
        enc.set_buffer(n1w_buf, 3); enc.set_buffer(n1b_buf, 4)
        enc.set_value(dim_u, 5); enc.dispatch_threadgroups({seq_len, 1, 1}, {32, 1, 1})
        enc.memory_barrier

        # === FFN (reads from h_out) ===
        if is_moe
          gate_w_arr = lw.gate_w.not_nil!
          exp_up_qw = lw.expert_up_w.not_nil!
          exp_down_qw = lw.expert_down_w.not_nil!
          up_row_bytes = (dim // 256) * exp_up_qw.type.block_bytes
          up_expert_bytes = ffn_dim * up_row_bytes
          down_row_bytes = (ffn_dim // 256) * exp_down_qw.type.block_bytes
          down_expert_bytes = dim * down_row_bytes
          up_gw_full = gw(exp_up_qw, Array(Float32).new(ffn_dim, 0.0_f32))
          down_gw_full = gw(exp_down_qw, Array(Float32).new(dim, 0.0_f32))

          # Fused SIMD gate + softmax + top-k + expert count (1 simdgroup per token)
          gate_w_buf = upload_f32(gate_w_arr)
          enc.set_pipeline(pipe("zero_int"))
          enc.set_buffer(ws.expert_counts, 0, w)
          enc.dispatch_1d(n_experts, 256)
          enc.memory_barrier

          enc.set_pipeline(pipe("gate_softmax_topk_count"))
          enc.set_buffer(h_out, 0); enc.set_buffer(gate_w_buf, 1)
          enc.set_buffer(ws.routing_ids, 2, w); enc.set_buffer(ws.routing_wts, 3, w)
          enc.set_buffer(ws.expert_counts, 4, rw)
          enc.set_value(dim_u, 5); enc.set_value(n_experts.to_u32, 6)
          enc.set_value(n_experts_used.to_u32, 7)
          enc.dispatch_threadgroups({seq_len, 1, 1}, {32, 1, 1})
          enc.memory_barrier

          if seq_len <= 64
            # === SYNC-FREE PATH: all experts × all tokens ===
            expert_stride_mid = seq_len.to_i64 * ffn_dim * 2
            expert_stride_out = seq_len.to_i64 * dim * 2
            # All expert UPs — INDEPENDENT (no barrier between experts)
            n_experts.times do |ei|
              up_offset = ei.to_i64 * up_expert_bytes
              enc.set_pipeline(pipe(matmul_kernel(exp_up_qw.type)))
              enc.set_buffer(up_gw_full.buffer, 0, offset: up_offset)
              enc.set_buffer(h_out, 1)
              enc.set_buffer(up_gw_full.bias_buffer, 2)
              enc.set_buffer(ws.expert_mid, 3, w, offset: ei.to_i64 * expert_stride_mid, length: expert_stride_mid)
              enc.set_value(dim_u, 4); enc.set_value(ffn_dim_u, 5)
              enc.set_value(batch, 6); enc.set_value(yes_gelu, 7)
              grid, tg = matmul_dispatch_tg(ffn_dim, seq_len); enc.dispatch_threadgroups(grid, tg)
            end
            enc.memory_barrier  # all ups done → downs read expert_mid
            # All expert DOWNs — INDEPENDENT
            n_experts.times do |ei|
              down_offset = ei.to_i64 * down_expert_bytes
              enc.set_pipeline(pipe(matmul_kernel(exp_down_qw.type)))
              enc.set_buffer(down_gw_full.buffer, 0, offset: down_offset)
              enc.set_buffer(ws.expert_mid, 1, offset: ei.to_i64 * expert_stride_mid, length: expert_stride_mid)
              enc.set_buffer(down_gw_full.bias_buffer, 2)
              enc.set_buffer(ws.expert_out, 3, w, offset: ei.to_i64 * expert_stride_out, length: expert_stride_out)
              enc.set_value(ffn_dim_u, 4); enc.set_value(dim_u, 5)
              enc.set_value(batch, 6); enc.set_value(no_gelu, 7)
              grid, tg = matmul_dispatch_tg(dim, seq_len); enc.dispatch_threadgroups(grid, tg)
            end
            enc.memory_barrier  # all downs done → scatter reads expert_out

            enc.set_pipeline(pipe("moe_weighted_scatter"))
            enc.set_buffer(ws.ffn_out, 0, rw); enc.set_buffer(ws.expert_out, 1)
            enc.set_buffer(ws.routing_ids, 2); enc.set_buffer(ws.routing_wts, 3)
            enc.set_value(dim_u, 4); enc.set_value(batch, 5)
            enc.set_value(n_experts_used.to_u32, 6); enc.set_value(n_experts.to_u32, 7)
            enc.dispatch_1d(seq_len * dim, 256)
            enc.memory_barrier  # ffn_out ready → norm2
          else
            # === GPU-NATIVE ROUTING — zero CPU sync ===
            # expert_counts already populated by fused gate_softmax_topk_count
            total_routing = seq_len * n_experts_used

            # Fused: prefix_sum + zero_counts + build_routing + write_dispatch_args
            enc.set_pipeline(pipe("moe_route_and_dispatch"))
            enc.set_buffer(ws.routing_ids, 0); enc.set_buffer(ws.routing_wts, 1)
            enc.set_buffer(ws.gather_map, 2, w); enc.set_buffer(ws.scatter_wts, 3, w)
            enc.set_buffer(ws.expert_counts, 4, rw); enc.set_buffer(ws.expert_offsets, 5, w)
            enc.set_buffer(ws.dispatch_args, 6, w)
            enc.set_value(n_experts_used.to_u32, 7); enc.set_value(n_experts.to_u32, 8)
            enc.set_value(batch, 9)
            enc.set_value(ffn_dim_u, 10); enc.set_value(dim_u, 11); enc.set_value(dim_u, 12)
            tg_sz = {seq_len, 1024}.min
            enc.dispatch_threadgroups({1, 1, 1}, {tg_sz, 1, 1})
            enc.memory_barrier

            # Gather routed tokens to packed buffer
            enc.set_pipeline(pipe("moe_gather"))
            enc.set_buffer(h_out, 0); enc.set_buffer(ws.moe_input, 1, w); enc.set_buffer(ws.gather_map, 2)
            enc.set_value(dim_u, 3)
            enc.dispatch_1d(total_routing * dim, 256)
            enc.memory_barrier

            # Compute batched dispatch args on GPU (expert_tg_offsets + indirect grids)
            enc.set_pipeline(pipe("moe_write_batched_args"))
            enc.set_buffer(ws.expert_offsets, 0)
            enc.set_buffer(ws.expert_tg_offs, 1, w)
            enc.set_buffer(ws.batched_up_grid, 2, w)
            enc.set_buffer(ws.batched_down_grid, 3, w)
            enc.set_value(n_experts.to_u32, 4)
            enc.set_value(ffn_dim_u, 5); enc.set_value(dim_u, 6)
            enc.dispatch_1d(1, 1)
            enc.memory_barrier

            # LTP Diamond: ALL 8 expert UPs in ONE batched dispatch
            batched_up_kernel = exp_up_qw.type.q5_k? ? "batched_mm_q5k" : "batched_mm_q6k"
            enc.set_pipeline(pipe(batched_up_kernel))
            enc.set_buffer(up_gw_full.buffer, 0)       # all expert weights concatenated
            enc.set_buffer(ws.moe_input, 1)
            enc.set_buffer(up_gw_full.bias_buffer, 2)
            enc.set_buffer(ws.moe_mid, 3, w)
            enc.set_buffer(ws.expert_offsets, 4)
            enc.set_buffer(ws.expert_tg_offs, 5)
            enc.set_value(dim_u, 6); enc.set_value(ffn_dim_u, 7)
            enc.set_value(yes_gelu, 8); enc.set_value(n_experts.to_u32, 9)
            enc.set_value(up_expert_bytes.to_u32, 10)
            enc.set_threadgroup_memory(MM_SHMEM, 0)
            enc.dispatch_threadgroups_indirect(ws.batched_up_grid, 0_i64, {128, 1, 1})
            enc.memory_barrier

            # LTP Diamond: ALL 8 expert DOWNs in ONE batched dispatch
            batched_down_kernel = exp_down_qw.type.q5_k? ? "batched_mm_q5k" : "batched_mm_q6k"
            enc.set_pipeline(pipe(batched_down_kernel))
            enc.set_buffer(down_gw_full.buffer, 0)
            enc.set_buffer(ws.moe_mid, 1)
            enc.set_buffer(down_gw_full.bias_buffer, 2)
            enc.set_buffer(ws.moe_output, 3, w)
            enc.set_buffer(ws.expert_offsets, 4)
            enc.set_buffer(ws.expert_tg_offs, 5)
            enc.set_value(ffn_dim_u, 6); enc.set_value(dim_u, 7)
            enc.set_value(no_gelu, 8); enc.set_value(n_experts.to_u32, 9)
            enc.set_value(down_expert_bytes.to_u32, 10)
            enc.set_threadgroup_memory(MM_SHMEM, 0)
            enc.dispatch_threadgroups_indirect(ws.batched_down_grid, 0_i64, {128, 1, 1})
            enc.memory_barrier

            # Atomic scatter: ALL routing slots in ONE dispatch (no sequential barriers)
            # Zero the f32 accumulator (use zero_int for 4-byte elements)
            enc.set_pipeline(pipe("zero_int"))
            enc.set_buffer(ws.ffn_out_f32, 0, w)
            enc.dispatch_1d(seq_len * dim, 256)  # zero_int writes int32 = same size as float
            enc.memory_barrier

            # Single atomic scatter dispatch
            enc.set_pipeline(pipe("moe_scatter_atomic"))
            enc.set_buffer(ws.ffn_out_f32, 0, rw)
            enc.set_buffer(ws.moe_output, 1)
            enc.set_buffer(ws.gather_map, 2)
            enc.set_buffer(ws.scatter_wts, 3)
            enc.set_value(dim_u, 4)
            enc.set_value(total_routing.to_u32, 5)
            enc.dispatch_1d(total_routing * dim, 256)
            enc.memory_barrier

            # norm2 reads f32 scatter output directly (fused f32→f16 + residual + layernorm)
            enc.set_pipeline(pipe("residual_layernorm_f32"))
            enc.set_buffer(h_out, 0, rw); enc.set_buffer(ws.ffn_out_f32, 1)
            enc.set_buffer(n2w_buf, 2); enc.set_buffer(n2b_buf, 3)
            enc.set_value(dim_u, 4); enc.dispatch_threadgroups({seq_len, 1, 1}, {32, 1, 1})
            enc.memory_barrier
          end

          # norm2 for sync-free MoE path (half input from ffn_out)
          if seq_len <= 64
            enc.set_pipeline(pipe("residual_layernorm"))
            enc.set_buffer(h_out, 0, rw); enc.set_buffer(ws.ffn_out, 1)
            enc.set_buffer(n2w_buf, 2); enc.set_buffer(n2b_buf, 3)
            enc.set_value(dim_u, 4); enc.dispatch_threadgroups({seq_len, 1, 1}, {32, 1, 1})
            enc.memory_barrier
          end
        else
          # Dense FFN
          up_gw = gw(lw.ffn_up_w.not_nil!, lw.ffn_up_b.not_nil!)
          enc.set_buffer(up_gw.buffer, 0); enc.set_buffer(h_out, 1)
          enc.set_buffer(up_gw.bias_buffer, 2); enc.set_buffer(ws.ffn_mid, 3, w)
          enc.set_value(dim_u, 4); enc.set_value(ffn_dim_u, 5)
          enc.set_value(batch, 6); enc.set_value(yes_gelu, 7)
          matmul_dispatch(enc, up_gw.type, ffn_dim, seq_len)
          enc.memory_barrier

          down_gw = gw(lw.ffn_down_w.not_nil!, lw.ffn_down_b.not_nil!)
          enc.set_buffer(down_gw.buffer, 0); enc.set_buffer(ws.ffn_mid, 1)
          enc.set_buffer(down_gw.bias_buffer, 2); enc.set_buffer(ws.ffn_out, 3, w)
          enc.set_value(ffn_dim_u, 4); enc.set_value(dim_u, 5)
          enc.set_value(batch, 6); enc.set_value(no_gelu, 7)
          matmul_dispatch(enc, down_gw.type, dim, seq_len)
          enc.memory_barrier

          enc.set_pipeline(pipe("residual_layernorm"))
          enc.set_buffer(h_out, 0, rw); enc.set_buffer(ws.ffn_out, 1)
          enc.set_buffer(n2w_buf, 2); enc.set_buffer(n2b_buf, 3)
          enc.set_value(dim_u, 4); enc.dispatch_threadgroups({seq_len, 1, 1}, {32, 1, 1})
          enc.memory_barrier
        end

        # Per-layer comparison with CPU reference (if provided)
        if (refs = cpu_ref_per_layer) && layer_idx < refs.size
          cr = refs[layer_idx]
          enc.end_encoding; cmd.commit_and_wait
          hp16 = h_out.contents.as(Pointer(UInt16))
          n = seq_len * dim
          dot = 0.0_f64; ng = 0.0_f64; nc = 0.0_f64
          max_err = 0.0_f64
          n.times do |i|
            g = Dequant.fp16_to_f32(hp16[i]).to_f64; c = cr[i].to_f64
            dot += g * c; ng += g * g; nc += c * c
            e = (g - c).abs; max_err = e if e > max_err
          end
          cmd = ML::Metal::CommandBuffer.new(fast: true); cmd.enqueue
          enc = ML::Metal::ComputeEncoder.new(cmd, concurrent: true)
          cos = dot / (Math.sqrt(ng) * Math.sqrt(nc))
          moe = (layer_idx % moe_every_n == 1) ? " (MoE)" : " (dense)"
          STDERR.puts "L#{layer_idx}#{moe} cos=#{cos.round(6)} max_err=#{max_err.round(6)}"
        end
      end

      # Mean pool
      final_hidden = h_bufs[layers.size % 2]
      enc.set_pipeline(pipe("mean_pool_l2"))
      enc.set_buffer(final_hidden, 0); enc.set_buffer(ws.output, 1, ML::Metal::BufferAccess::Write)
      enc.set_value(seq_len.to_u32, 2); enc.set_value(dim_u, 3)
      enc.dispatch_1d(1, 1)

      _t_encode_done = Time.instant

      # Compile and encode with wave-optimized barriers
      graph.compile!
      if ENV["PROFILE_MOE"]?
        st = graph.stats
        STDERR.puts "  graph: ops=#{st.n_ops} waves=#{st.n_waves} barriers=#{st.n_barriers} max_width=#{st.max_wave_width}"
      end
      graph.compile!
      if ENV["PROFILE_MOE"]?
        st = graph.stats
        STDERR.puts "  graph: ops=#{st.n_ops} waves=#{st.n_waves} barriers=#{st.n_barriers} max_width=#{st.max_wave_width}"
      end
      graph.encode(cmd)

      cmd.commit
      cmd.wait

      o_ptr = ws.output.contents.as(Pointer(Float32))
      Array(Float32).new(dim) { |i| o_ptr[i] }
    end


    # ComputeBackend interface fallbacks
    def matmul(x : Array(Float32), rows : Int32, qw : QuantWeight, bias : Array(Float32)) : Array(Float32)
      F32Backend.new.matmul(x, rows, qw, bias)
    end
    def layer_norm!(x : Array(Float32), n_pos : Int32, dim : Int32, w : Array(Float32), b : Array(Float32)) : Nil
      F32Backend.new.layer_norm!(x, n_pos, dim, w, b)
    end
    def softmax_row!(scores : Array(Float32), offset : Int32, len : Int32) : Nil
      F32Backend.new.softmax_row!(scores, offset, len)
    end
    def gelu(x : Float32) : Float32
      0.5_f32 * x * (1.0_f32 + Math.tanh(0.7978845608_f32 * (x + 0.044715_f32 * x * x * x)))
    end
    def dot(a : Array(Float32), a_off : Int32, b : Array(Float32), b_off : Int32, len : Int32) : Float32
      sum = 0.0_f32; len.times { |i| sum += a[a_off + i] * b[b_off + i] }; sum
    end
  end
end

{% end %}
