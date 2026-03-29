# Metal GPU backend — fused dequant-matmul kernels, unified memory.
# ONE command buffer per dense layer, CPU fallback for MoE.

{% unless flag?(:cpu_only) %}

require "./compute"

module ML::GGUF
  BERT_FUSED_SOURCE = {{ read_file("#{__DIR__}/kernels/bert_fused.metal") }}

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

    def initialize(max_seq : Int32, dim : Int32, ffn_dim : Int32, n_heads : Int32, head_dim : Int32,
                   rope_cos : Array(Float32), rope_sin : Array(Float32))
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
    end
  end

  class MetalBackend
    include ComputeBackend

    @gpu_weights : Hash(UInt64, GPUWeight)
    @pipelines : Hash(String, ML::Metal::ComputePipeline)
    @workspace : GPUWorkspace?

    def initialize
      raise "Metal not available" unless ML::Metal::Device.available?
      @gpu_weights = {} of UInt64 => GPUWeight
      @pipelines = {} of String => ML::Metal::ComputePipeline
      %w[fused_q5k_matmul_gelu fused_q6k_matmul_gelu
         qkv_split rope_neox_inplace attention_forward
         layernorm_inplace residual_add mean_pool_l2].each do |name|
        @pipelines[name] = ML::Metal::PipelineCache.get(name) {
          ML::Metal::ComputePipeline.new(name, BERT_FUSED_SOURCE)
        }
      end
    end

    def upload_weight(qw : QuantWeight, bias : Array(Float32)) : Nil
      key = qw.raw.to_unsafe.address
      @gpu_weights[key] ||= GPUWeight.new(qw, bias)
    end

    def init_workspace(max_seq : Int32, dim : Int32, ffn_dim : Int32, n_heads : Int32,
                       head_dim : Int32, rope_cos : Array(Float32), rope_sin : Array(Float32))
      @workspace = GPUWorkspace.new(max_seq, dim, ffn_dim, n_heads, head_dim, rope_cos, rope_sin)
    end

    private def gw(qw : QuantWeight, bias : Array(Float32)) : GPUWeight
      key = qw.raw.to_unsafe.address
      @gpu_weights[key] ||= GPUWeight.new(qw, bias)
    end

    private def pipe(name : String) : ML::Metal::ComputePipeline
      @pipelines[name]
    end

    def encode_layers(
      hidden_data : Array(Float32), seq_len : Int32,
      layers : Array(NomicBertMoE::LayerWeights),
      dim : Int32, n_heads : Int32, head_dim : Int32, ffn_dim : Int32,
      n_experts : Int32, n_experts_used : Int32, moe_every_n : Int32,
    ) : Array(Float32)
      ws = @workspace.not_nil!
      h_ptr = ws.hidden.contents.as(Pointer(Float32))
      (seq_len * dim).times { |i| h_ptr[i] = hidden_data[i] }

      scale = 1.0_f32 / Math.sqrt(head_dim.to_f32)
      batch = seq_len.to_u32; dim_u = dim.to_u32
      n_heads_u = n_heads.to_u32; head_dim_u = head_dim.to_u32; ffn_dim_u = ffn_dim.to_u32
      no_gelu = 0_u32; yes_gelu = 1_u32

      # Pre-upload norm weights
      norm_bufs = layers.map do |lw|
        n1w = ML::MetalBuffer.new(dim.to_i64 * 4); n1w.write(lw.norm1_w)
        n1b = ML::MetalBuffer.new(dim.to_i64 * 4); n1b.write(lw.norm1_b)
        n2w = ML::MetalBuffer.new(dim.to_i64 * 4); n2w.write(lw.norm2_w)
        n2b = ML::MetalBuffer.new(dim.to_i64 * 4); n2b.write(lw.norm2_b)
        {n1w, n1b, n2w, n2b}
      end

      layers.each_with_index do |lw, layer_idx|
        is_moe = (layer_idx % moe_every_n == 1) && n_experts > 0
        n1w_buf, n1b_buf, n2w_buf, n2b_buf = norm_bufs[layer_idx]

        # === Attention (GPU) ===
        cmd = ML::Metal::CommandBuffer.new

        qkv_gw = gw(lw.attn_qkv_w, lw.attn_qkv_b)
        kernel = qkv_gw.type.q5_k? ? "fused_q5k_matmul_gelu" : "fused_q6k_matmul_gelu"
        enc = ML::Metal::ComputeEncoder.new(cmd)
        enc.set_pipeline(pipe(kernel))
        enc.set_buffer(qkv_gw.buffer, 0); enc.set_buffer(ws.hidden, 1)
        enc.set_buffer(qkv_gw.bias_buffer, 2); enc.set_buffer(ws.qkv, 3)
        enc.set_value(dim_u, 4); enc.set_value(3_u32 * dim_u, 5)
        enc.set_value(batch, 6); enc.set_value(no_gelu, 7)
        enc.dispatch({3 * dim, seq_len, 1}, {Math.min(256, 3 * dim), 1, 1})
        enc.end_encoding

        enc = ML::Metal::ComputeEncoder.new(cmd)
        enc.set_pipeline(pipe("qkv_split"))
        enc.set_buffer(ws.qkv, 0); enc.set_buffer(ws.q, 1)
        enc.set_buffer(ws.k, 2); enc.set_buffer(ws.v, 3)
        enc.set_value(batch, 4); enc.set_value(dim_u, 5)
        enc.set_value(n_heads_u, 6); enc.set_value(head_dim_u, 7)
        enc.dispatch_1d(seq_len * dim, 256)
        enc.end_encoding

        # RoPE Q
        enc = ML::Metal::ComputeEncoder.new(cmd)
        enc.set_pipeline(pipe("rope_neox_inplace"))
        enc.set_buffer(ws.q, 0); enc.set_buffer(ws.cos_cache, 1); enc.set_buffer(ws.sin_cache, 2)
        enc.set_value(batch, 3); enc.set_value(n_heads_u, 4); enc.set_value(head_dim_u, 5)
        enc.dispatch_1d(seq_len * n_heads, 256)
        enc.end_encoding

        # RoPE K
        enc = ML::Metal::ComputeEncoder.new(cmd)
        enc.set_pipeline(pipe("rope_neox_inplace"))
        enc.set_buffer(ws.k, 0); enc.set_buffer(ws.cos_cache, 1); enc.set_buffer(ws.sin_cache, 2)
        enc.set_value(batch, 3); enc.set_value(n_heads_u, 4); enc.set_value(head_dim_u, 5)
        enc.dispatch_1d(seq_len * n_heads, 256)
        enc.end_encoding

        # Attention
        enc = ML::Metal::ComputeEncoder.new(cmd)
        enc.set_pipeline(pipe("attention_forward"))
        enc.set_buffer(ws.q, 0); enc.set_buffer(ws.k, 1)
        enc.set_buffer(ws.v, 2); enc.set_buffer(ws.attn_out, 3)
        enc.set_value(batch, 4); enc.set_value(n_heads_u, 5)
        enc.set_value(head_dim_u, 6); enc.set_value(scale, 7)
        enc.dispatch_1d(n_heads * seq_len, 1)
        enc.end_encoding

        # Output projection
        out_gw = gw(lw.attn_out_w, lw.attn_out_b)
        kernel = out_gw.type.q5_k? ? "fused_q5k_matmul_gelu" : "fused_q6k_matmul_gelu"
        enc = ML::Metal::ComputeEncoder.new(cmd)
        enc.set_pipeline(pipe(kernel))
        enc.set_buffer(out_gw.buffer, 0); enc.set_buffer(ws.attn_out, 1)
        enc.set_buffer(out_gw.bias_buffer, 2); enc.set_buffer(ws.ffn_out, 3)
        enc.set_value(dim_u, 4); enc.set_value(dim_u, 5)
        enc.set_value(batch, 6); enc.set_value(no_gelu, 7)
        enc.dispatch({dim.to_i32, seq_len, 1}, {Math.min(256, dim), 1, 1})
        enc.end_encoding

        # Residual + norm1
        enc = ML::Metal::ComputeEncoder.new(cmd)
        enc.set_pipeline(pipe("residual_add"))
        enc.set_buffer(ws.hidden, 0); enc.set_buffer(ws.ffn_out, 1)
        enc.dispatch_1d(seq_len * dim, 256)
        enc.end_encoding

        enc = ML::Metal::ComputeEncoder.new(cmd)
        enc.set_pipeline(pipe("layernorm_inplace"))
        enc.set_buffer(ws.hidden, 0); enc.set_buffer(n1w_buf, 1); enc.set_buffer(n1b_buf, 2)
        enc.set_value(dim_u, 3)
        enc.dispatch_1d(seq_len, 1)
        enc.end_encoding

        # === FFN ===
        if is_moe
          cmd.commit_and_wait  # Must sync before CPU MoE
          # CPU fallback for MoE
          hp = ws.hidden.contents.as(Pointer(Float32))
          h_arr = Array(Float32).new(seq_len * dim) { |i| hp[i] }

          gate_w = lw.gate_w.not_nil!
          exp_up_qw = lw.expert_up_w.not_nil!
          exp_down_qw = lw.expert_down_w.not_nil!
          moe_result = Array(Float32).new(seq_len * dim, 0.0_f32)
          zero_ffn = Array(Float32).new(ffn_dim, 0.0_f32)
          zero_dim = Array(Float32).new(dim, 0.0_f32)
          up_row_bytes = (dim // 256) * exp_up_qw.type.block_bytes
          up_expert_bytes = ffn_dim * up_row_bytes
          down_row_bytes = (ffn_dim // 256) * exp_down_qw.type.block_bytes
          down_expert_bytes = dim * down_row_bytes

          seq_len.times do |pos|
            x_off = pos * dim; x_pos = h_arr[x_off, dim]
            gate_logits = Array(Float32).new(n_experts, 0.0_f32)
            n_experts.times { |e| d = 0.0_f32; dim.times { |j| d += h_arr[x_off + j] * gate_w[e * dim + j] }; gate_logits[e] = d }
            max_g = gate_logits.max
            probs = gate_logits.map { |v| Math.exp(v - max_g).to_f32 }; sum_p = probs.sum; probs.map! { |v| v / sum_p }
            top = probs.each_with_index.to_a.sort_by { |v, _| -v }.first(n_experts_used)
            top.each do |prob, ei|
              up_s = Bytes.new(exp_up_qw.raw.to_unsafe + ei * up_expert_bytes, up_expert_bytes, read_only: true)
              h = QuantMatmul.matmul_add(x_pos, 1, dim, up_s, exp_up_qw.type, ffn_dim, zero_ffn)
              h.map! { |v| 0.5_f32 * v * (1.0_f32 + Math.tanh(0.7978845608_f32 * (v + 0.044715_f32 * v * v * v))) }
              dn_s = Bytes.new(exp_down_qw.raw.to_unsafe + ei * down_expert_bytes, down_expert_bytes, read_only: true)
              d = QuantMatmul.matmul_add(h, 1, ffn_dim, dn_s, exp_down_qw.type, dim, zero_dim)
              dim.times { |j| moe_result[pos * dim + j] += prob * d[j] }
            end
          end

          (seq_len * dim).times { |i| h_arr[i] += moe_result[i] }
          F32Backend.new.layer_norm!(h_arr, seq_len, dim, lw.norm2_w, lw.norm2_b)
          ws.hidden.write(h_arr)
        else
          up_gw = gw(lw.ffn_up_w.not_nil!, lw.ffn_up_b.not_nil!)
          kernel = up_gw.type.q5_k? ? "fused_q5k_matmul_gelu" : "fused_q6k_matmul_gelu"
          enc = ML::Metal::ComputeEncoder.new(cmd)
          enc.set_pipeline(pipe(kernel))
          enc.set_buffer(up_gw.buffer, 0); enc.set_buffer(ws.hidden, 1)
          enc.set_buffer(up_gw.bias_buffer, 2); enc.set_buffer(ws.ffn_mid, 3)
          enc.set_value(dim_u, 4); enc.set_value(ffn_dim_u, 5)
          enc.set_value(batch, 6); enc.set_value(yes_gelu, 7)
          enc.dispatch({ffn_dim, seq_len, 1}, {Math.min(256, ffn_dim), 1, 1})
          enc.end_encoding

          down_gw = gw(lw.ffn_down_w.not_nil!, lw.ffn_down_b.not_nil!)
          kernel = down_gw.type.q5_k? ? "fused_q5k_matmul_gelu" : "fused_q6k_matmul_gelu"
          enc = ML::Metal::ComputeEncoder.new(cmd)
          enc.set_pipeline(pipe(kernel))
          enc.set_buffer(down_gw.buffer, 0); enc.set_buffer(ws.ffn_mid, 1)
          enc.set_buffer(down_gw.bias_buffer, 2); enc.set_buffer(ws.ffn_out, 3)
          enc.set_value(ffn_dim_u, 4); enc.set_value(dim_u, 5)
          enc.set_value(batch, 6); enc.set_value(no_gelu, 7)
          enc.dispatch({dim.to_i32, seq_len, 1}, {Math.min(256, dim), 1, 1})
          enc.end_encoding

          enc = ML::Metal::ComputeEncoder.new(cmd)
          enc.set_pipeline(pipe("residual_add"))
          enc.set_buffer(ws.hidden, 0); enc.set_buffer(ws.ffn_out, 1)
          enc.dispatch_1d(seq_len * dim, 256)
          enc.end_encoding

          enc = ML::Metal::ComputeEncoder.new(cmd)
          enc.set_pipeline(pipe("layernorm_inplace"))
          enc.set_buffer(ws.hidden, 0); enc.set_buffer(n2w_buf, 1); enc.set_buffer(n2b_buf, 2)
          enc.set_value(dim_u, 3)
          enc.dispatch_1d(seq_len, 1)
          enc.end_encoding

          cmd.commit_and_wait
        end
      end

      # Mean pool + normalize
      cmd = ML::Metal::CommandBuffer.new
      enc = ML::Metal::ComputeEncoder.new(cmd)
      enc.set_pipeline(pipe("mean_pool_l2"))
      enc.set_buffer(ws.hidden, 0); enc.set_buffer(ws.output, 1)
      enc.set_value(seq_len.to_u32, 2); enc.set_value(dim_u, 3)
      enc.dispatch_1d(1, 1)
      enc.end_encoding
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
      sum = 0.0_f32
      len.times { |i| sum += a[a_off + i] * b[b_off + i] }
      sum
    end
  end
end

{% end %}
