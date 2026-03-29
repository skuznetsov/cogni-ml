# Metal GPU backend — all BERT ops on GPU, minimal CPU↔GPU syncs.
# MoE: CPU gating (tiny) + GPU expert matmuls (per-expert dispatch).

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
    # MoE scratch: per-token input + per-expert output
    getter moe_tok : ML::MetalBuffer   # [dim] single token input
    getter moe_exp_out : ML::MetalBuffer  # [dim] single expert output

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
      @moe_tok = ML::MetalBuffer.new(dim.to_i64 * 4)
      @moe_exp_out = ML::MetalBuffer.new(dim.to_i64 * 4)
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
         layernorm_inplace residual_add weighted_add zero_region mean_pool_l2].each do |name|
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

    private def matmul_kernel(type : TensorType) : String
      type.q5_k? ? "fused_q5k_matmul_gelu" : "fused_q6k_matmul_gelu"
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
        enc.dispatch({3 * dim, seq_len, 1}, {Math.min(256, 3 * dim), 1, 1}); enc.end_encoding

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
        enc.dispatch_1d(n_heads * seq_len, 1); enc.end_encoding

        out_gw = gw(lw.attn_out_w, lw.attn_out_b)
        enc = ML::Metal::ComputeEncoder.new(cmd)
        enc.set_pipeline(pipe(matmul_kernel(out_gw.type)))
        enc.set_buffer(out_gw.buffer, 0); enc.set_buffer(ws.attn_out, 1)
        enc.set_buffer(out_gw.bias_buffer, 2); enc.set_buffer(ws.ffn_out, 3)
        enc.set_value(dim_u, 4); enc.set_value(dim_u, 5); enc.set_value(batch, 6); enc.set_value(no_gelu, 7)
        enc.dispatch({dim.to_i32, seq_len, 1}, {Math.min(256, dim), 1, 1}); enc.end_encoding

        enc = ML::Metal::ComputeEncoder.new(cmd); enc.set_pipeline(pipe("residual_add"))
        enc.set_buffer(ws.hidden, 0); enc.set_buffer(ws.ffn_out, 1)
        enc.dispatch_1d(seq_len * dim, 256); enc.end_encoding

        enc = ML::Metal::ComputeEncoder.new(cmd); enc.set_pipeline(pipe("layernorm_inplace"))
        enc.set_buffer(ws.hidden, 0); enc.set_buffer(n1w_buf, 1); enc.set_buffer(n1b_buf, 2)
        enc.set_value(dim_u, 3); enc.dispatch_1d(seq_len, 1); enc.end_encoding

        # === FFN ===
        if is_moe
          # Sync to read hidden for CPU gating
          cmd.commit_and_wait
          hp = ws.hidden.contents.as(Pointer(Float32))

          # Debug: dump hidden at pos=0 before gating
          STDERR.puts "DEBUG MoE L#{layer_idx}: hidden[0][0..4]=#{Array(Float32).new(5) { |i| hp[i] }.map(&.round(4))}"

          # CPU gating: ~5μs (768×8 matmul + softmax + top-2, per token)
          gate_w = lw.gate_w.not_nil!
          exp_up_qw = lw.expert_up_w.not_nil!
          exp_down_qw = lw.expert_down_w.not_nil!
          up_row_bytes = (dim // 256) * exp_up_qw.type.block_bytes
          up_expert_bytes = ffn_dim * up_row_bytes
          down_row_bytes = (ffn_dim // 256) * exp_down_qw.type.block_bytes
          down_expert_bytes = dim * down_row_bytes

          routing = [] of {Int32, Int32, Float32}
          seq_len.times do |pos|
            gate_logits = Array(Float32).new(n_experts, 0.0_f32)
            n_experts.times { |e| d = 0.0_f32; dim.times { |j| d += hp[pos * dim + j] * gate_w[e * dim + j] }; gate_logits[e] = d }
            max_g = gate_logits.max
            probs = gate_logits.map { |v| Math.exp(v - max_g).to_f32 }; sum_p = probs.sum; probs.map! { |v| v / sum_p }
            top = probs.each_with_index.to_a.sort_by { |v, _| -v }.first(n_experts_used)
            if pos == 0
              STDERR.puts "DEBUG MoE L#{layer_idx} pos=0: top=#{top.map { |p, e| {e, p.round(4)} }}"
            end
            top.each { |prob, ei| routing << {pos, ei, prob} }
          end

          # Debug: CPU expert 0 up for pos=0 comparison
          if layer_idx == 1
            pos0_input = Array(Float32).new(dim) { |j| hp[j] }
            up_slice = Bytes.new(exp_up_qw.raw.to_unsafe, ffn_dim * ((dim // 256) * exp_up_qw.type.block_bytes), read_only: true)
            zero_b = Array(Float32).new(ffn_dim, 0.0_f32)
            cpu_up0 = QuantMatmul.matmul_add(pos0_input, 1, dim, up_slice, exp_up_qw.type, ffn_dim, zero_b)
            STDERR.puts "DEBUG CPU expert0 up pos=0: first 5=#{cpu_up0[0, 5].map(&.round(4))}"
          end

          # GPU expert matmuls — all in one cmd, no intermediate sync
          cmd = ML::Metal::CommandBuffer.new
          up_gw_full = gw(exp_up_qw, Array(Float32).new(ffn_dim, 0.0_f32))
          down_gw_full = gw(exp_down_qw, Array(Float32).new(dim, 0.0_f32))

          # Zero ffn_out (accumulator for weighted expert outputs)
          enc = ML::Metal::ComputeEncoder.new(cmd); enc.set_pipeline(pipe("zero_region"))
          enc.set_buffer(ws.ffn_out, 0); enc.set_value(0_u32, 1)
          enc.dispatch_1d(seq_len * dim, 256); enc.end_encoding

          # Pre-allocate per-routing-entry input buffers (can't share — GPU dispatches are batched)
          tok_bufs = routing.map do |pos, _, _|
            b = ML::MetalBuffer.new(dim.to_i64 * 4)
            bp = b.contents.as(Pointer(Float32))
            dim.times { |j| bp[j] = hp[pos * dim + j] }
            b
          end
          exp_out_bufs = routing.map { ML::MetalBuffer.new(dim.to_i64 * 4) }
          exp_mid_bufs = routing.map { ML::MetalBuffer.new(ffn_dim.to_i64 * 4) }

          routing.each_with_index do |(pos, ei, weight), ri|
            # Expert up + GELU
            up_offset = ei.to_i64 * up_expert_bytes
            enc = ML::Metal::ComputeEncoder.new(cmd)
            enc.set_pipeline(pipe(matmul_kernel(exp_up_qw.type)))
            enc.set_buffer(up_gw_full.buffer, 0, offset: up_offset)
            enc.set_buffer(tok_bufs[ri], 1)
            enc.set_buffer(up_gw_full.bias_buffer, 2)
            enc.set_buffer(exp_mid_bufs[ri], 3)
            enc.set_value(dim_u, 4); enc.set_value(ffn_dim_u, 5)
            enc.set_value(1_u32, 6); enc.set_value(yes_gelu, 7)
            enc.dispatch({ffn_dim, 1, 1}, {Math.min(256, ffn_dim), 1, 1}); enc.end_encoding

            # Expert down
            down_offset = ei.to_i64 * down_expert_bytes
            enc = ML::Metal::ComputeEncoder.new(cmd)
            enc.set_pipeline(pipe(matmul_kernel(exp_down_qw.type)))
            enc.set_buffer(down_gw_full.buffer, 0, offset: down_offset)
            enc.set_buffer(exp_mid_bufs[ri], 1)
            enc.set_buffer(down_gw_full.bias_buffer, 2)
            enc.set_buffer(exp_out_bufs[ri], 3)
            enc.set_value(ffn_dim_u, 4); enc.set_value(dim_u, 5)
            enc.set_value(1_u32, 6); enc.set_value(no_gelu, 7)
            enc.dispatch({dim.to_i32, 1, 1}, {Math.min(256, dim), 1, 1}); enc.end_encoding

            # Weighted accumulate: ffn_out[pos*dim..] += weight * exp_out
            pos_offset = (pos * dim).to_u32
            enc = ML::Metal::ComputeEncoder.new(cmd); enc.set_pipeline(pipe("weighted_add"))
            enc.set_buffer(ws.ffn_out, 0); enc.set_buffer(exp_out_bufs[ri], 1)
            enc.set_value(pos_offset, 2); enc.set_value(weight, 3)
            enc.dispatch_1d(dim, 256); enc.end_encoding
          end

          # Residual + norm2
          enc = ML::Metal::ComputeEncoder.new(cmd); enc.set_pipeline(pipe("residual_add"))
          enc.set_buffer(ws.hidden, 0); enc.set_buffer(ws.ffn_out, 1)
          enc.dispatch_1d(seq_len * dim, 256); enc.end_encoding

          enc = ML::Metal::ComputeEncoder.new(cmd); enc.set_pipeline(pipe("layernorm_inplace"))
          enc.set_buffer(ws.hidden, 0); enc.set_buffer(n2w_buf, 1); enc.set_buffer(n2b_buf, 2)
          enc.set_value(dim_u, 3); enc.dispatch_1d(seq_len, 1); enc.end_encoding

          # Must commit and wait BEFORE buffers go out of scope (GC safety)
          cmd.commit_and_wait

          # Debug: check GPU expert outputs match CPU (with GELU)
          if layer_idx == 1
            tb0 = tok_bufs[0].contents.as(Pointer(Float32))
            tok_arr = Array(Float32).new(dim) { |j| tb0[j] }
            up_slice = Bytes.new(exp_up_qw.raw.to_unsafe, ffn_dim * ((dim // 256) * exp_up_qw.type.block_bytes), read_only: true)
            cpu_up0 = QuantMatmul.matmul_add(tok_arr, 1, dim, up_slice, exp_up_qw.type, ffn_dim, Array(Float32).new(ffn_dim, 0.0_f32))
            cpu_up0.map! { |v| 0.5_f32 * v * (1.0_f32 + Math.tanh(0.7978845608_f32 * (v + 0.044715_f32 * v * v * v))) }
            # CPU expert 0 down
            dn_slice = Bytes.new(exp_down_qw.raw.to_unsafe, dim * ((ffn_dim // 256) * exp_down_qw.type.block_bytes), read_only: true)
            cpu_dn0 = QuantMatmul.matmul_add(cpu_up0, 1, ffn_dim, dn_slice, exp_down_qw.type, dim, Array(Float32).new(dim, 0.0_f32))
            STDERR.puts "DEBUG CPU expert0 down (with GELU): [0..4]=#{cpu_dn0[0, 5].map(&.round(4))}"
            STDERR.puts "DEBUG CPU expert0 up+GELU: [0..4]=#{cpu_up0[0, 5].map(&.round(6))}"
          end

          # Debug: check GPU expert 0 up output for pos=0
          if layer_idx == 1
            mid0 = exp_mid_bufs[0].contents.as(Pointer(Float32))
            STDERR.puts "DEBUG GPU expert0 up pos=0: first 5=#{Array(Float32).new(5) { |i| mid0[i] }.map(&.round(4))}"
            out0 = exp_out_bufs[0].contents.as(Pointer(Float32))
            STDERR.puts "DEBUG GPU expert0 down pos=0: first 5=#{Array(Float32).new(5) { |i| out0[i] }.map(&.round(4))}"

            # Check ffn_out accumulation
            fo = ws.ffn_out.contents.as(Pointer(Float32))
            STDERR.puts "DEBUG GPU ffn_out pos=0 [0..4]: #{Array(Float32).new(5) { |i| fo[i] }.map(&.round(4))}"
          end

          cmd = ML::Metal::CommandBuffer.new
        else
          # Dense FFN
          up_gw = gw(lw.ffn_up_w.not_nil!, lw.ffn_up_b.not_nil!)
          enc = ML::Metal::ComputeEncoder.new(cmd)
          enc.set_pipeline(pipe(matmul_kernel(up_gw.type)))
          enc.set_buffer(up_gw.buffer, 0); enc.set_buffer(ws.hidden, 1)
          enc.set_buffer(up_gw.bias_buffer, 2); enc.set_buffer(ws.ffn_mid, 3)
          enc.set_value(dim_u, 4); enc.set_value(ffn_dim_u, 5)
          enc.set_value(batch, 6); enc.set_value(yes_gelu, 7)
          enc.dispatch({ffn_dim, seq_len, 1}, {Math.min(256, ffn_dim), 1, 1}); enc.end_encoding

          down_gw = gw(lw.ffn_down_w.not_nil!, lw.ffn_down_b.not_nil!)
          enc = ML::Metal::ComputeEncoder.new(cmd)
          enc.set_pipeline(pipe(matmul_kernel(down_gw.type)))
          enc.set_buffer(down_gw.buffer, 0); enc.set_buffer(ws.ffn_mid, 1)
          enc.set_buffer(down_gw.bias_buffer, 2); enc.set_buffer(ws.ffn_out, 3)
          enc.set_value(ffn_dim_u, 4); enc.set_value(dim_u, 5)
          enc.set_value(batch, 6); enc.set_value(no_gelu, 7)
          enc.dispatch({dim.to_i32, seq_len, 1}, {Math.min(256, dim), 1, 1}); enc.end_encoding

          enc = ML::Metal::ComputeEncoder.new(cmd); enc.set_pipeline(pipe("residual_add"))
          enc.set_buffer(ws.hidden, 0); enc.set_buffer(ws.ffn_out, 1)
          enc.dispatch_1d(seq_len * dim, 256); enc.end_encoding

          enc = ML::Metal::ComputeEncoder.new(cmd); enc.set_pipeline(pipe("layernorm_inplace"))
          enc.set_buffer(ws.hidden, 0); enc.set_buffer(n2w_buf, 1); enc.set_buffer(n2b_buf, 2)
          enc.set_value(dim_u, 3); enc.dispatch_1d(seq_len, 1); enc.end_encoding

          # Per-layer commit for dense too (correctness first, speed later)
          cmd.commit_and_wait
          cmd = ML::Metal::CommandBuffer.new
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
