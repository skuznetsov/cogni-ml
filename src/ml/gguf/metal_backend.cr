# Metal GPU backend — all BERT ops on GPU, minimal CPU↔GPU syncs.
# MoE: CPU gating (tiny) + GPU expert matmuls (per-expert dispatch).

{% unless flag?(:cpu_only) %}

require "./compute"

module ML::GGUF
  BERT_FUSED_SOURCE = {{ read_file("#{__DIR__}/kernels/bert_fused.metal") }}
  GEMM_SOURCE = {{ read_file("#{__DIR__}/kernels/gemm_q5k.metal") }}
  SIMD_GEMM_SOURCE = {{ read_file("#{__DIR__}/kernels/gemm_simd.metal") }}

  class GPUWeight
    getter buffer : ML::MetalBuffer
    getter bias_buffer : ML::MetalBuffer
    getter type : TensorType
    getter out_dim : Int32
    getter in_dim : Int32
    def initialize(qw : QuantWeight, bias : Array(Float32))
      @type = qw.type; @out_dim = qw.out_dim; @in_dim = qw.in_dim
      @buffer = ML::MetalBuffer.new(qw.raw.size.to_i64)
      @buffer.write_bytes(qw.raw.to_unsafe, qw.raw.size)
      @bias_buffer = ML::MetalBuffer.new(bias.size.to_i64 * 4)
      @bias_buffer.write(bias)
    end
  end

  class GPUWorkspace
    getter hidden : ML::MetalBuffer
    getter qkv : ML::MetalBuffer
    getter q : ML::MetalBuffer
    getter k : ML::MetalBuffer
    getter v : ML::MetalBuffer
    getter attn_out : ML::MetalBuffer
    getter ffn_mid : ML::MetalBuffer
    getter ffn_out : ML::MetalBuffer
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
    getter expert_mid : ML::MetalBuffer   # [n_experts, max_seq, ffn_dim]
    getter expert_out : ML::MetalBuffer  # [n_experts, max_seq, dim]

    def initialize(max_seq : Int32, dim : Int32, ffn_dim : Int32, n_heads : Int32, head_dim : Int32,
                   n_experts_used : Int32, rope_cos : Array(Float32), rope_sin : Array(Float32))
      @hidden   = ML::MetalBuffer.new(max_seq.to_i64 * dim * 4)
      @qkv      = ML::MetalBuffer.new(max_seq.to_i64 * 3 * dim * 4)
      @q        = ML::MetalBuffer.new(n_heads.to_i64 * max_seq * head_dim * 4)
      @k        = ML::MetalBuffer.new(n_heads.to_i64 * max_seq * head_dim * 4)
      @v        = ML::MetalBuffer.new(n_heads.to_i64 * max_seq * head_dim * 4)
      @attn_out = ML::MetalBuffer.new(max_seq.to_i64 * dim * 4)
      @ffn_mid  = ML::MetalBuffer.new(max_seq.to_i64 * ffn_dim * 4)
      @ffn_out  = ML::MetalBuffer.new(max_seq.to_i64 * dim * 4)
      @output   = ML::MetalBuffer.new(dim.to_i64 * 4)
      @cos_cache = ML::MetalBuffer.new(rope_cos.size.to_i64 * 4); @cos_cache.write(rope_cos)
      @sin_cache = ML::MetalBuffer.new(rope_sin.size.to_i64 * 4); @sin_cache.write(rope_sin)
      # Pre-allocated MoE batched buffers: one contiguous buffer per data type
      # Max tokens routed = max_seq * n_experts_used (each token → n_experts_used experts)
      max_routing = max_seq * n_experts_used
      @moe_input  = ML::MetalBuffer.new(max_routing.to_i64 * dim * 4)      # [max_routing, dim]
      @moe_mid    = ML::MetalBuffer.new(max_routing.to_i64 * ffn_dim * 4)  # [max_routing, ffn_dim]
      @moe_output = ML::MetalBuffer.new(max_routing.to_i64 * dim * 4)      # [max_routing, dim]
      # Gate + routing buffers
      @gate_logits = ML::MetalBuffer.new(max_seq.to_i64 * 8 * 4)
      @routing_ids = ML::MetalBuffer.new(max_routing.to_i64 * 4)  # int32
      @routing_wts = ML::MetalBuffer.new(max_routing.to_i64 * 4)  # float32
      @gather_map  = ML::MetalBuffer.new(max_routing.to_i64 * 4)  # int32
      @scatter_wts = ML::MetalBuffer.new(max_routing.to_i64 * 4)  # float32
      # All-experts buffers: [8, max_seq, dim/ffn_dim] for sync-free MoE
      @expert_mid  = ML::MetalBuffer.new(8_i64 * max_seq * ffn_dim * 4)
      @expert_out  = ML::MetalBuffer.new(8_i64 * max_seq * dim * 4)
    end
  end

  class MetalBackend
    include ComputeBackend

    @gpu_weights : Hash(UInt64, GPUWeight)
    @gpu_f32_bufs : Hash(UInt64, ML::MetalBuffer)
    @pipelines : Hash(String, ML::Metal::ComputePipeline)
    @workspace : GPUWorkspace?

    def initialize
      raise "Metal not available" unless ML::Metal::Device.available?
      @gpu_weights = {} of UInt64 => GPUWeight
      @gpu_f32_bufs = {} of UInt64 => ML::MetalBuffer
      @pipelines = {} of String => ML::Metal::ComputePipeline
      %w[fused_q5k_matmul_gelu fused_q6k_matmul_gelu
         qkv_split rope_neox_inplace attention_forward
         layernorm_inplace residual_add weighted_add zero_region mean_pool_l2
         gelu_inplace gate_matmul softmax_topk moe_gather scatter_weighted_add
         moe_weighted_scatter].each do |name|
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
      # SIMD GEMM kernels (primary — better precision via simd_sum)
      %w[simd_gemm_q5k simd_gemm_q6k].each do |name|
        @pipelines[name] = ML::Metal::PipelineCache.get(name) {
          ML::Metal::ComputePipeline.new(name, SIMD_GEMM_SOURCE)
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

    private def matmul_kernel(type : TensorType) : String
      type.q5_k? ? "simd_gemm_q5k" : "simd_gemm_q6k"
    end

    # Returns {threadgroup_count, threadgroup_size} for dispatchThreadgroups
    private def matmul_dispatch_tg(out_dim : Int32, batch : Int32) : { {Int32, Int32, Int32}, {Int32, Int32, Int32} }
      tg_count = {(out_dim + SIMD_N_ROWS - 1) // SIMD_N_ROWS, {batch, 1}.max, 1}
      tg_size = {32, SIMD_N_ROWS, 1}
      {tg_count, tg_size}
    end

    def encode_layers(
      hidden_data : Array(Float32), seq_len : Int32,
      layers : Array(NomicBertMoE::LayerWeights),
      dim : Int32, n_heads : Int32, head_dim : Int32, ffn_dim : Int32,
      n_experts : Int32, n_experts_used : Int32, moe_every_n : Int32,
      cpu_ref_per_layer : Array(Array(Float32))? = nil,  # per-layer comparison
    ) : Array(Float32)
      ws = @workspace.not_nil!
      h_ptr = ws.hidden.contents.as(Pointer(Float32))
      (seq_len * dim).times { |i| h_ptr[i] = hidden_data[i] }

      scale = 1.0_f32 / Math.sqrt(head_dim.to_f32)
      batch = seq_len.to_u32; dim_u = dim.to_u32
      n_heads_u = n_heads.to_u32; head_dim_u = head_dim.to_u32; ffn_dim_u = ffn_dim.to_u32
      no_gelu = 0_u32; yes_gelu = 1_u32

      norm_bufs = layers.map do |lw|
        n1w = ML::MetalBuffer.new(dim.to_i64 * 4); n1w.write(lw.norm1_w)
        n1b = ML::MetalBuffer.new(dim.to_i64 * 4); n1b.write(lw.norm1_b)
        n2w = ML::MetalBuffer.new(dim.to_i64 * 4); n2w.write(lw.norm2_w)
        n2b = ML::MetalBuffer.new(dim.to_i64 * 4); n2b.write(lw.norm2_b)
        {n1w, n1b, n2w, n2b}
      end

      cmd = ML::Metal::CommandBuffer.new

      layers.each_with_index do |lw, layer_idx|
        is_moe = (layer_idx % moe_every_n == 1) && n_experts > 0
        n1w_buf, n1b_buf, n2w_buf, n2b_buf = norm_bufs[layer_idx]

        # === Attention (always GPU) ===
        qkv_gw = gw(lw.attn_qkv_w, lw.attn_qkv_b)
        enc = ML::Metal::ComputeEncoder.new(cmd)
        enc.set_pipeline(pipe(matmul_kernel(qkv_gw.type)))
        enc.set_buffer(qkv_gw.buffer, 0); enc.set_buffer(ws.hidden, 1)
        enc.set_buffer(qkv_gw.bias_buffer, 2); enc.set_buffer(ws.qkv, 3)
        enc.set_value(dim_u, 4); enc.set_value(3_u32 * dim_u, 5)
        enc.set_value(batch, 6); enc.set_value(no_gelu, 7)
        grid, tg = matmul_dispatch_tg(3 * dim, seq_len); enc.dispatch_threadgroups(grid, tg); enc.end_encoding

        enc = ML::Metal::ComputeEncoder.new(cmd)
        enc.set_pipeline(pipe("qkv_split"))
        enc.set_buffer(ws.qkv, 0); enc.set_buffer(ws.q, 1); enc.set_buffer(ws.k, 2); enc.set_buffer(ws.v, 3)
        enc.set_value(batch, 4); enc.set_value(dim_u, 5); enc.set_value(n_heads_u, 6); enc.set_value(head_dim_u, 7)
        enc.dispatch_1d(seq_len * dim, 256); enc.end_encoding

        [ws.q, ws.k].each do |buf|
          enc = ML::Metal::ComputeEncoder.new(cmd)
          enc.set_pipeline(pipe("rope_neox_inplace"))
          enc.set_buffer(buf, 0); enc.set_buffer(ws.cos_cache, 1); enc.set_buffer(ws.sin_cache, 2)
          enc.set_value(batch, 3); enc.set_value(n_heads_u, 4); enc.set_value(head_dim_u, 5)
          enc.dispatch_1d(seq_len * n_heads, 256); enc.end_encoding
        end

        enc = ML::Metal::ComputeEncoder.new(cmd)
        enc.set_pipeline(pipe("attention_forward"))
        enc.set_buffer(ws.q, 0); enc.set_buffer(ws.k, 1); enc.set_buffer(ws.v, 2); enc.set_buffer(ws.attn_out, 3)
        enc.set_value(batch, 4); enc.set_value(n_heads_u, 5); enc.set_value(head_dim_u, 6); enc.set_value(scale, 7)
        enc.set_threadgroup_memory(seq_len * 4, 0)  # shared scores[seq_len]
        enc.dispatch_threadgroups({n_heads, seq_len, 1}, {32, 1, 1}); enc.end_encoding

        out_gw = gw(lw.attn_out_w, lw.attn_out_b)
        enc = ML::Metal::ComputeEncoder.new(cmd)
        enc.set_pipeline(pipe(matmul_kernel(out_gw.type)))
        enc.set_buffer(out_gw.buffer, 0); enc.set_buffer(ws.attn_out, 1)
        enc.set_buffer(out_gw.bias_buffer, 2); enc.set_buffer(ws.ffn_out, 3)
        enc.set_value(dim_u, 4); enc.set_value(dim_u, 5); enc.set_value(batch, 6); enc.set_value(no_gelu, 7)
        grid, tg = matmul_dispatch_tg(dim, seq_len); enc.dispatch_threadgroups(grid, tg); enc.end_encoding

        enc = ML::Metal::ComputeEncoder.new(cmd); enc.set_pipeline(pipe("residual_add"))
        enc.set_buffer(ws.hidden, 0); enc.set_buffer(ws.ffn_out, 1)
        enc.dispatch_1d(seq_len * dim, 256); enc.end_encoding

        enc = ML::Metal::ComputeEncoder.new(cmd); enc.set_pipeline(pipe("layernorm_inplace"))
        enc.set_buffer(ws.hidden, 0); enc.set_buffer(n1w_buf, 1); enc.set_buffer(n1b_buf, 2)
        enc.set_value(dim_u, 3); enc.dispatch_threadgroups({seq_len, 1, 1}, {32, 1, 1}); enc.end_encoding

        # === FFN ===
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

          # GPU gate + softmax + top-k
          gate_w_buf = upload_f32(gate_w_arr)
          enc = ML::Metal::ComputeEncoder.new(cmd); enc.set_pipeline(pipe("gate_matmul"))
          enc.set_buffer(ws.hidden, 0); enc.set_buffer(gate_w_buf, 1); enc.set_buffer(ws.gate_logits, 2)
          enc.set_value(dim_u, 3); enc.set_value(n_experts.to_u32, 4)
          enc.dispatch({n_experts, seq_len, 1}, {n_experts, 1, 1}); enc.end_encoding

          enc = ML::Metal::ComputeEncoder.new(cmd); enc.set_pipeline(pipe("softmax_topk"))
          enc.set_buffer(ws.gate_logits, 0); enc.set_buffer(ws.routing_ids, 1); enc.set_buffer(ws.routing_wts, 2)
          enc.set_value(n_experts.to_u32, 3); enc.set_value(n_experts_used.to_u32, 4)
          enc.dispatch_1d(seq_len, 1); enc.end_encoding

          if seq_len <= 64
            # === SYNC-FREE PATH: all experts × all tokens ===
            expert_stride_mid = seq_len.to_i64 * ffn_dim * 4
            expert_stride_out = seq_len.to_i64 * dim * 4
            n_experts.times do |ei|
              up_offset = ei.to_i64 * up_expert_bytes
              enc = ML::Metal::ComputeEncoder.new(cmd)
              enc.set_pipeline(pipe(matmul_kernel(exp_up_qw.type)))
              enc.set_buffer(up_gw_full.buffer, 0, offset: up_offset)
              enc.set_buffer(ws.hidden, 1)
              enc.set_buffer(up_gw_full.bias_buffer, 2)
              enc.set_buffer(ws.expert_mid, 3, offset: ei.to_i64 * expert_stride_mid)
              enc.set_value(dim_u, 4); enc.set_value(ffn_dim_u, 5)
              enc.set_value(batch, 6); enc.set_value(yes_gelu, 7)
              grid, tg = matmul_dispatch_tg(ffn_dim, seq_len); enc.dispatch_threadgroups(grid, tg); enc.end_encoding

              down_offset = ei.to_i64 * down_expert_bytes
              enc = ML::Metal::ComputeEncoder.new(cmd)
              enc.set_pipeline(pipe(matmul_kernel(exp_down_qw.type)))
              enc.set_buffer(down_gw_full.buffer, 0, offset: down_offset)
              enc.set_buffer(ws.expert_mid, 1, offset: ei.to_i64 * expert_stride_mid)
              enc.set_buffer(down_gw_full.bias_buffer, 2)
              enc.set_buffer(ws.expert_out, 3, offset: ei.to_i64 * expert_stride_out)
              enc.set_value(ffn_dim_u, 4); enc.set_value(dim_u, 5)
              enc.set_value(batch, 6); enc.set_value(no_gelu, 7)
              grid, tg = matmul_dispatch_tg(dim, seq_len); enc.dispatch_threadgroups(grid, tg); enc.end_encoding
            end

            enc = ML::Metal::ComputeEncoder.new(cmd); enc.set_pipeline(pipe("moe_weighted_scatter"))
            enc.set_buffer(ws.ffn_out, 0); enc.set_buffer(ws.expert_out, 1)
            enc.set_buffer(ws.routing_ids, 2); enc.set_buffer(ws.routing_wts, 3)
            enc.set_value(dim_u, 4); enc.set_value(batch, 5)
            enc.set_value(n_experts_used.to_u32, 6); enc.set_value(n_experts.to_u32, 7)
            enc.dispatch_1d(seq_len * dim, 256); enc.end_encoding
          else
            # === SYNC PATH: only routed experts (4x less compute) ===
            cmd.commit_and_wait
            rids = ws.routing_ids.contents.as(Pointer(Int32))
            rwts = ws.routing_wts.contents.as(Pointer(Float32))
            hp = ws.hidden.contents.as(Pointer(Float32))

            expert_groups = Hash(Int32, Array({Int32, Float32})).new
            seq_len.times do |pos|
              n_experts_used.times do |ki|
                idx = pos * n_experts_used + ki
                ei = rids[idx]; w = rwts[idx]
                (expert_groups[ei] ||= [] of {Int32, Float32}) << {pos, w}
              end
            end

            gmap = ws.gather_map.contents.as(Pointer(Int32))
            swts = ws.scatter_wts.contents.as(Pointer(Float32))
            total_routing = seq_len * n_experts_used
            ri_base = 0
            expert_groups.each do |_, entries|
              entries.each_with_index do |(pos, weight), li|
                gmap[ri_base + li] = pos; swts[ri_base + li] = weight
              end
              ri_base += entries.size
            end

            cmd = ML::Metal::CommandBuffer.new
            enc = ML::Metal::ComputeEncoder.new(cmd); enc.set_pipeline(pipe("moe_gather"))
            enc.set_buffer(ws.hidden, 0); enc.set_buffer(ws.moe_input, 1); enc.set_buffer(ws.gather_map, 2)
            enc.set_value(dim_u, 3)
            enc.dispatch_1d(total_routing * dim, 256); enc.end_encoding

            enc = ML::Metal::ComputeEncoder.new(cmd); enc.set_pipeline(pipe("zero_region"))
            enc.set_buffer(ws.ffn_out, 0); enc.set_value(0_u32, 1)
            enc.dispatch_1d(seq_len * dim, 256); enc.end_encoding

            ri_base = 0
            expert_groups.each do |ei, entries|
              eb = entries.size
              up_offset = ei.to_i64 * up_expert_bytes
              input_off = ri_base.to_i64 * dim * 4
              mid_off = ri_base.to_i64 * ffn_dim * 4
              enc = ML::Metal::ComputeEncoder.new(cmd)
              enc.set_pipeline(pipe(matmul_kernel(exp_up_qw.type)))
              enc.set_buffer(up_gw_full.buffer, 0, offset: up_offset)
              enc.set_buffer(ws.moe_input, 1, offset: input_off)
              enc.set_buffer(up_gw_full.bias_buffer, 2)
              enc.set_buffer(ws.moe_mid, 3, offset: mid_off)
              enc.set_value(dim_u, 4); enc.set_value(ffn_dim_u, 5)
              enc.set_value(eb.to_u32, 6); enc.set_value(yes_gelu, 7)
              grid, tg = matmul_dispatch_tg(ffn_dim, eb); enc.dispatch_threadgroups(grid, tg); enc.end_encoding

              down_offset = ei.to_i64 * down_expert_bytes
              out_off = ri_base.to_i64 * dim * 4
              enc = ML::Metal::ComputeEncoder.new(cmd)
              enc.set_pipeline(pipe(matmul_kernel(exp_down_qw.type)))
              enc.set_buffer(down_gw_full.buffer, 0, offset: down_offset)
              enc.set_buffer(ws.moe_mid, 1, offset: mid_off)
              enc.set_buffer(down_gw_full.bias_buffer, 2)
              enc.set_buffer(ws.moe_output, 3, offset: out_off)
              enc.set_value(ffn_dim_u, 4); enc.set_value(dim_u, 5)
              enc.set_value(eb.to_u32, 6); enc.set_value(no_gelu, 7)
              grid, tg = matmul_dispatch_tg(dim, eb); enc.dispatch_threadgroups(grid, tg); enc.end_encoding

              enc = ML::Metal::ComputeEncoder.new(cmd); enc.set_pipeline(pipe("scatter_weighted_add"))
              enc.set_buffer(ws.ffn_out, 0)
              enc.set_buffer(ws.moe_output, 1, offset: ri_base.to_i64 * dim * 4)
              enc.set_buffer(ws.gather_map, 2, offset: ri_base.to_i64 * 4)
              enc.set_buffer(ws.scatter_wts, 3, offset: ri_base.to_i64 * 4)
              enc.set_value(dim_u, 4)
              enc.dispatch_1d(eb * dim, 256); enc.end_encoding
              ri_base += entries.size
            end
          end

          # Residual + norm2
          enc = ML::Metal::ComputeEncoder.new(cmd); enc.set_pipeline(pipe("residual_add"))
          enc.set_buffer(ws.hidden, 0); enc.set_buffer(ws.ffn_out, 1)
          enc.dispatch_1d(seq_len * dim, 256); enc.end_encoding

          enc = ML::Metal::ComputeEncoder.new(cmd); enc.set_pipeline(pipe("layernorm_inplace"))
          enc.set_buffer(ws.hidden, 0); enc.set_buffer(n2w_buf, 1); enc.set_buffer(n2b_buf, 2)
          enc.set_value(dim_u, 3); enc.dispatch_threadgroups({seq_len, 1, 1}, {32, 1, 1}); enc.end_encoding
        else
          # Dense FFN
          up_gw = gw(lw.ffn_up_w.not_nil!, lw.ffn_up_b.not_nil!)
          enc = ML::Metal::ComputeEncoder.new(cmd)
          enc.set_pipeline(pipe(matmul_kernel(up_gw.type)))
          enc.set_buffer(up_gw.buffer, 0); enc.set_buffer(ws.hidden, 1)
          enc.set_buffer(up_gw.bias_buffer, 2); enc.set_buffer(ws.ffn_mid, 3)
          enc.set_value(dim_u, 4); enc.set_value(ffn_dim_u, 5)
          enc.set_value(batch, 6); enc.set_value(yes_gelu, 7)
          grid, tg = matmul_dispatch_tg(ffn_dim, seq_len); enc.dispatch_threadgroups(grid, tg); enc.end_encoding

          down_gw = gw(lw.ffn_down_w.not_nil!, lw.ffn_down_b.not_nil!)
          enc = ML::Metal::ComputeEncoder.new(cmd)
          enc.set_pipeline(pipe(matmul_kernel(down_gw.type)))
          enc.set_buffer(down_gw.buffer, 0); enc.set_buffer(ws.ffn_mid, 1)
          enc.set_buffer(down_gw.bias_buffer, 2); enc.set_buffer(ws.ffn_out, 3)
          enc.set_value(ffn_dim_u, 4); enc.set_value(dim_u, 5)
          enc.set_value(batch, 6); enc.set_value(no_gelu, 7)
          grid, tg = matmul_dispatch_tg(dim, seq_len); enc.dispatch_threadgroups(grid, tg); enc.end_encoding

          enc = ML::Metal::ComputeEncoder.new(cmd); enc.set_pipeline(pipe("residual_add"))
          enc.set_buffer(ws.hidden, 0); enc.set_buffer(ws.ffn_out, 1)
          enc.dispatch_1d(seq_len * dim, 256); enc.end_encoding

          enc = ML::Metal::ComputeEncoder.new(cmd); enc.set_pipeline(pipe("layernorm_inplace"))
          enc.set_buffer(ws.hidden, 0); enc.set_buffer(n2w_buf, 1); enc.set_buffer(n2b_buf, 2)
          enc.set_value(dim_u, 3); enc.dispatch_threadgroups({seq_len, 1, 1}, {32, 1, 1}); enc.end_encoding
          # No commit — dense layers continue in same cmd buffer
        end

        # Per-layer comparison with CPU reference (if provided)
        if (refs = cpu_ref_per_layer) && layer_idx < refs.size
          cr = refs[layer_idx]
          hp = ws.hidden.contents.as(Pointer(Float32))
          n = seq_len * dim
          dot = 0.0_f64; ng = 0.0_f64; nc = 0.0_f64
          max_err = 0.0_f64
          n.times do |i|
            g = hp[i].to_f64; c = cr[i].to_f64
            dot += g * c; ng += g * g; nc += c * c
            e = (g - c).abs; max_err = e if e > max_err
          end
          cos = dot / (Math.sqrt(ng) * Math.sqrt(nc))
          moe = (layer_idx % moe_every_n == 1) ? " (MoE)" : " (dense)"
          STDERR.puts "L#{layer_idx}#{moe} cos=#{cos.round(6)} max_err=#{max_err.round(6)}"
        end
      end

      # Mean pool + normalize
      enc = ML::Metal::ComputeEncoder.new(cmd)
      enc.set_pipeline(pipe("mean_pool_l2"))
      enc.set_buffer(ws.hidden, 0); enc.set_buffer(ws.output, 1)
      enc.set_value(seq_len.to_u32, 2); enc.set_value(dim_u, 3)
      enc.dispatch_1d(1, 1); enc.end_encoding

      cmd.commit_and_wait

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
