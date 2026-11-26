# Full-attention kernels, KV updates, and layer dispatch for the Qwen35 Metal engine.

module ML
  module GGUF
    module Qwen35Metal
      # The Flash-MMA kernel has a deliberately exact ABI. Automatic admission
      # stays limited to measured model/device shapes. Explicit "1" admits only
      # bounded ABI-compatible prompt sizes for experiments; "0" is rollback.
      def self.prefill_attn_flash_d256_policy?(device_name : String,
                                               start_pos : Int32,
                                               n_tokens : Int32,
                                               n_head : Int32,
                                               n_head_kv : Int32,
                                               head_dim : Int32,
                                               kv_cache_f16 : Bool,
                                               adaptive : Bool,
                                               override : String?) : Bool
        enabled = case override
                  when nil, "1" then true
                  when "0"      then false
                  else
                    raise ArgumentError.new("QWEN35_PREFILL_ATTN_FLASH_D256 must be 0 or 1")
                  end
        return false unless enabled
        return false unless device_name == "Apple M2 Max"
        return false unless kv_cache_f16 && !adaptive
        return false unless head_dim == 256 && n_head_kv == 4

        if override == "1"
          return false unless n_head == 16 || n_head == 24
          # Prefix/tail support is an operator-validated experiment, not an
          # automatic continuation-speed certificate. Subtraction avoids
          # overflowing an externally supplied start position.
          return start_pos >= 0 && n_tokens >= 1 && n_tokens <= 2048 &&
            start_pos <= 8192 - n_tokens
        end

        return false unless start_pos == 0 && n_tokens % 64 == 0
        # Admit only measured model/batch points. The kernel ABI can execute
        # other multiples of 64, but numerical validity alone is not a speed
        # certificate and the shorter 27B GQA6 points remain below the current
        # production admission threshold.
        (n_head == 16 && (n_tokens == 1024 || n_tokens == 2048)) ||
          (n_head == 24 && n_tokens == 2048)
      end

      {% unless flag?(:cpu_only) %}
        ATTN_DECODE_SOURCE = {{ read_file("#{__DIR__}/../../../metal/kernels/qwen35/attn_decode_qwen35.metal") }}
        ATTN_DECODE_H16_SOURCE = "#define QWEN35_KV_CACHE_F16 1\n" + ATTN_DECODE_SOURCE

        FULLATTN_SOURCE = {{ read_file("#{__DIR__}/../../../metal/kernels/qwen35/fullattn_qwen35.metal") }}
        FULLATTN_H16_SOURCE = "#define QWEN35_KV_CACHE_F16 1\n" + FULLATTN_SOURCE
        ATTN_FLASH_D256_SOURCE = {{ read_file("#{__DIR__}/../../../metal/kernels/qwen35/qwen35_attn_flash_d256.metal") }}

        @@attn_pipeline : ML::Metal::ComputePipeline?
        @@attn_h16_pipeline : ML::Metal::ComputePipeline?
        @@attn_gqa4_pipeline : ML::Metal::ComputePipeline?
        @@attn_gqa4_h16_pipeline : ML::Metal::ComputePipeline?
        @@attn_splitk_stage1_pipeline : ML::Metal::ComputePipeline?
        @@attn_splitk_stage1_h16_pipeline : ML::Metal::ComputePipeline?
        @@attn_splitk_stage2_pipeline : ML::Metal::ComputePipeline?

        @@split_qgate_pipeline : ML::Metal::ComputePipeline?
        @@split_qgate_rows_pipeline : ML::Metal::ComputePipeline?
        @@rmsnorm_heads_pipeline : ML::Metal::ComputePipeline?
        @@rmsnorm_heads_rows_pipeline : ML::Metal::ComputePipeline?
        @@rope_partial_pipeline : ML::Metal::ComputePipeline?
        @@rope_partial_rows_pipeline : ML::Metal::ComputePipeline?
        @@kv_write_pipeline : ML::Metal::ComputePipeline?
        @@kv_write_h16_pipeline : ML::Metal::ComputePipeline?
        @@kv_write_rows_pipeline : ML::Metal::ComputePipeline?
        @@kv_write_rows_h16_pipeline : ML::Metal::ComputePipeline?
        @@attn_rows_pipeline : ML::Metal::ComputePipeline?
        @@attn_rows_h16_pipeline : ML::Metal::ComputePipeline?
        @@attn_rows_sg4_pipeline : ML::Metal::ComputePipeline?
        @@attn_rows_sg4_h16_pipeline : ML::Metal::ComputePipeline?
        @@attn_rows_sg4_pregate_pipeline : ML::Metal::ComputePipeline?
        @@attn_rows_sg4_pregate_h16_pipeline : ML::Metal::ComputePipeline?
        @@attn_flash_d256_pipeline : ML::Metal::ComputePipeline?

        private def self.attn_pipeline : ML::Metal::ComputePipeline
          @@attn_pipeline ||= ML::Metal::PipelineCache.get("qwen35_attn_decode") {
            ML::Metal::ComputePipeline.new("qwen35_attn_decode", ATTN_DECODE_SOURCE)
          }
        end

        private def self.attn_h16_pipeline : ML::Metal::ComputePipeline
          @@attn_h16_pipeline ||= ML::Metal::PipelineCache.get("qwen35_attn_decode_kv_h16") {
            ML::Metal::ComputePipeline.new("qwen35_attn_decode", ATTN_DECODE_H16_SOURCE)
          }
        end

        private def self.attn_gqa4_pipeline : ML::Metal::ComputePipeline
          @@attn_gqa4_pipeline ||= ML::Metal::PipelineCache.get("qwen35_attn_decode_gqa4") {
            ML::Metal::ComputePipeline.new("qwen35_attn_decode_gqa4", ATTN_DECODE_SOURCE)
          }
        end

        private def self.attn_gqa4_h16_pipeline : ML::Metal::ComputePipeline
          @@attn_gqa4_h16_pipeline ||= ML::Metal::PipelineCache.get("qwen35_attn_decode_gqa4_kv_h16") {
            ML::Metal::ComputePipeline.new("qwen35_attn_decode_gqa4", ATTN_DECODE_H16_SOURCE)
          }
        end

        private def self.attn_splitk_stage1_pipeline : ML::Metal::ComputePipeline
          @@attn_splitk_stage1_pipeline ||= ML::Metal::PipelineCache.get("qwen35_attn_decode_splitk_stage1") {
            ML::Metal::ComputePipeline.new("qwen35_attn_decode_splitk_stage1", ATTN_DECODE_SOURCE)
          }
        end

        private def self.attn_splitk_stage1_h16_pipeline : ML::Metal::ComputePipeline
          @@attn_splitk_stage1_h16_pipeline ||= ML::Metal::PipelineCache.get("qwen35_attn_decode_splitk_stage1_kv_h16") {
            ML::Metal::ComputePipeline.new("qwen35_attn_decode_splitk_stage1", ATTN_DECODE_H16_SOURCE)
          }
        end

        private def self.attn_splitk_stage2_pipeline : ML::Metal::ComputePipeline
          @@attn_splitk_stage2_pipeline ||= ML::Metal::PipelineCache.get("qwen35_attn_decode_splitk_stage2") {
            ML::Metal::ComputePipeline.new("qwen35_attn_decode_splitk_stage2", ATTN_DECODE_SOURCE)
          }
        end

        private def self.split_qgate_pipeline : ML::Metal::ComputePipeline
          @@split_qgate_pipeline ||= ML::Metal::PipelineCache.get("qwen35_split_qgate") {
            ML::Metal::ComputePipeline.new("qwen35_split_qgate", FULLATTN_SOURCE)
          }
        end

        private def self.split_qgate_rows_pipeline : ML::Metal::ComputePipeline
          @@split_qgate_rows_pipeline ||= ML::Metal::PipelineCache.get("qwen35_split_qgate_rows") {
            ML::Metal::ComputePipeline.new("qwen35_split_qgate_rows", FULLATTN_SOURCE)
          }
        end

        private def self.rmsnorm_heads_pipeline : ML::Metal::ComputePipeline
          @@rmsnorm_heads_pipeline ||= ML::Metal::PipelineCache.get("qwen35_rmsnorm_heads") {
            ML::Metal::ComputePipeline.new("qwen35_rmsnorm_heads", FULLATTN_SOURCE)
          }
        end

        private def self.rmsnorm_heads_rows_pipeline : ML::Metal::ComputePipeline
          @@rmsnorm_heads_rows_pipeline ||= ML::Metal::PipelineCache.get("qwen35_rmsnorm_heads_rows") {
            ML::Metal::ComputePipeline.new("qwen35_rmsnorm_heads_rows", FULLATTN_SOURCE)
          }
        end

        private def self.rope_partial_pipeline : ML::Metal::ComputePipeline
          @@rope_partial_pipeline ||= ML::Metal::PipelineCache.get("qwen35_rope_partial") {
            ML::Metal::ComputePipeline.new("qwen35_rope_partial", FULLATTN_SOURCE)
          }
        end

        private def self.rope_partial_rows_pipeline : ML::Metal::ComputePipeline
          @@rope_partial_rows_pipeline ||= ML::Metal::PipelineCache.get("qwen35_rope_partial_rows") {
            ML::Metal::ComputePipeline.new("qwen35_rope_partial_rows", FULLATTN_SOURCE)
          }
        end

        private def self.kv_write_pipeline : ML::Metal::ComputePipeline
          @@kv_write_pipeline ||= ML::Metal::PipelineCache.get("qwen35_kv_write") {
            ML::Metal::ComputePipeline.new("qwen35_kv_write", FULLATTN_SOURCE)
          }
        end

        private def self.kv_write_h16_pipeline : ML::Metal::ComputePipeline
          @@kv_write_h16_pipeline ||= ML::Metal::PipelineCache.get("qwen35_kv_write_h16") {
            ML::Metal::ComputePipeline.new("qwen35_kv_write", FULLATTN_H16_SOURCE)
          }
        end

        private def self.kv_write_rows_pipeline : ML::Metal::ComputePipeline
          @@kv_write_rows_pipeline ||= ML::Metal::PipelineCache.get("qwen35_kv_write_rows") {
            ML::Metal::ComputePipeline.new("qwen35_kv_write_rows", FULLATTN_SOURCE)
          }
        end

        private def self.kv_write_rows_h16_pipeline : ML::Metal::ComputePipeline
          @@kv_write_rows_h16_pipeline ||= ML::Metal::PipelineCache.get("qwen35_kv_write_rows_h16") {
            ML::Metal::ComputePipeline.new("qwen35_kv_write_rows", FULLATTN_H16_SOURCE)
          }
        end

        private def self.attn_rows_pipeline : ML::Metal::ComputePipeline
          @@attn_rows_pipeline ||= ML::Metal::PipelineCache.get("qwen35_attn_decode_rows") {
            ML::Metal::ComputePipeline.new("qwen35_attn_decode_rows", FULLATTN_SOURCE)
          }
        end

        private def self.attn_rows_h16_pipeline : ML::Metal::ComputePipeline
          @@attn_rows_h16_pipeline ||= ML::Metal::PipelineCache.get("qwen35_attn_decode_rows_kv_h16") {
            ML::Metal::ComputePipeline.new("qwen35_attn_decode_rows", FULLATTN_H16_SOURCE)
          }
        end

        private def self.attn_rows_sg4_pipeline : ML::Metal::ComputePipeline
          @@attn_rows_sg4_pipeline ||= ML::Metal::PipelineCache.get("qwen35_attn_decode_rows_sg4") {
            ML::Metal::ComputePipeline.new("qwen35_attn_decode_rows_sg4", FULLATTN_SOURCE)
          }
        end

        private def self.attn_rows_sg4_h16_pipeline : ML::Metal::ComputePipeline
          @@attn_rows_sg4_h16_pipeline ||= ML::Metal::PipelineCache.get("qwen35_attn_decode_rows_sg4_kv_h16") {
            ML::Metal::ComputePipeline.new("qwen35_attn_decode_rows_sg4", FULLATTN_H16_SOURCE)
          }
        end

        private def self.attn_rows_sg4_pregate_pipeline : ML::Metal::ComputePipeline
          @@attn_rows_sg4_pregate_pipeline ||= ML::Metal::PipelineCache.get("qwen35_attn_decode_rows_sg4_pregate") {
            ML::Metal::ComputePipeline.new("qwen35_attn_decode_rows_sg4_pregate", FULLATTN_SOURCE)
          }
        end

        private def self.attn_rows_sg4_pregate_h16_pipeline : ML::Metal::ComputePipeline
          @@attn_rows_sg4_pregate_h16_pipeline ||= ML::Metal::PipelineCache.get("qwen35_attn_decode_rows_sg4_pregate_kv_h16") {
            ML::Metal::ComputePipeline.new("qwen35_attn_decode_rows_sg4_pregate", FULLATTN_H16_SOURCE)
          }
        end

        private def self.attn_flash_d256_pipeline : ML::Metal::ComputePipeline
          @@attn_flash_d256_pipeline ||= ML::Metal::PipelineCache.get("qwen35_attn_flash_d256") {
            ML::Metal::ComputePipeline.new("qwen35_attn_flash_d256", ATTN_FLASH_D256_SOURCE)
          }
        end

        private def self.prefill_attn_rows_sg4_enabled? : Bool
          ENV["QWEN35_PREFILL_ATTN_ROWS_SG4_OFF"]? != "1"
        end

        private def self.prefill_attn_rows_sg4_pregate_enabled? : Bool
          ENV["QWEN35_PREFILL_ATTN_ROWS_SG4_PREGATE"]? == "1"
        end

        private def self.prefill_attn_rows_sg4_direct_gate_min_tokens : Int32
          (ENV["QWEN35_PREFILL_ATTN_ROWS_SG4_DIRECT_GATE_MIN"]? || "1024").to_i32
        end

        private def self.prefill_attn_rows_sg4_direct_gate_enabled?(n_tokens : Int32) : Bool
          ENV["QWEN35_PREFILL_ATTN_ROWS_SG4_DIRECT_GATE_OFF"]? != "1" &&
            n_tokens >= prefill_attn_rows_sg4_direct_gate_min_tokens
        end

        private def self.prefill_attn_flash_d256_enabled?(start_pos : Int32,
                                                          n_tokens : Int32,
                                                          n_head : Int32,
                                                          n_head_kv : Int32,
                                                          head_dim : Int32,
                                                          kv_cache_f16 : Bool) : Bool
          override = ENV["QWEN35_PREFILL_ATTN_FLASH_D256"]?
          device_name = override == "0" ? "" : ML::Metal::Device.instance.name
          prefill_attn_flash_d256_policy?(
            device_name, start_pos, n_tokens, n_head, n_head_kv, head_dim,
            kv_cache_f16, false, override,
          )
        end

        private def self.q8_kv_dual_gemv_enabled? : Bool
          q8_dual_gemv_enabled? && ENV["QWEN35_Q8_KV_DUAL_GEMV_OFF"]? != "1"
        end

        private def self.q8_kv_dual_gemv_candidate?(k_qw : QuantWeight,
                                                    v_qw : QuantWeight,
                                                    batch : Int32 = 1) : Bool
          q8_kv_dual_gemv_enabled? && batch == 1 &&
            k_qw.type.q8_0? && v_qw.type.q8_0? &&
            k_qw.in_dim == v_qw.in_dim &&
            k_qw.out_dim == v_qw.out_dim
        end

        private def self.attn_splitk_min_context : Int32
          (ENV["QWEN35_ATTN_SPLITK_MIN_CTX"]? || "128").to_i32
        end

        private def self.attn_splitk_chunk_size : Int32
          value = (ENV["QWEN35_ATTN_SPLITK_CHUNK"]? || "64").to_i32
          value > 0 ? value : 64
        end

        private def self.attn_gqa4_enabled? : Bool
          ENV["QWEN35_ATTN_GQA4"]? == "1" && ENV["QWEN35_ATTN_GQA4_OFF"]? != "1"
        end

        # Gated attention decode on Metal. CPU side prepares Q (post-rmsnorm,
        # post-RoPE) and gate (raw, kernel applies sigmoid); K/V are already
        # appended to the per-layer k_cache_buf/v_cache_buf at row `pos`.
        # Returns the gated attention output `attn_o` as `Array(Float32)`
        # of length `n_head * head_dim`, ready for the output projection.
        def self.attn_decode(q : Array(Float32),
                             gate : Array(Float32),
                             k_cache_buf : ML::MetalBuffer,
                             v_cache_buf : ML::MetalBuffer,
                             pos : Int32, n_head : Int32, n_head_kv : Int32,
                             head_dim : Int32, heads_per_group : Int32,
                             scale : Float32) : Array(Float32)
          ML::Metal::Device.init!

          t0 = Time.instant if Profile.enabled?
          q_dim    = n_head * head_dim
          q_bytes  = q_dim.to_i64 * sizeof(Float32)
          q_buf    = Scratch.get(:attn_q,    q_bytes)
          gate_buf = Scratch.get(:attn_gate, q_bytes)
          out_buf  = Scratch.get(:attn_out,  q_bytes)
          q_buf.write(q)
          gate_buf.write(gate)

          cache_len = pos + 1

          cmd = ML::Metal::CommandBuffer.new
          enc = ML::Metal::ComputeEncoder.new(cmd)
          enc.set_pipeline(attn_pipeline)
          enc.set_buffer(q_buf,         0)
          enc.set_buffer(gate_buf,      1)
          enc.set_buffer(k_cache_buf,   2)
          enc.set_buffer(v_cache_buf,   3)
          enc.set_buffer(out_buf,       4, ML::Metal::BufferAccess::Write)
          enc.set_value(cache_len.to_u32,       5)
          enc.set_value(n_head.to_u32,          6)
          enc.set_value(n_head_kv.to_u32,       7)
          enc.set_value(head_dim.to_u32,        8)
          enc.set_value(heads_per_group.to_u32, 9)
          enc.set_value(scale,                 10)
          enc.dispatch_threadgroups({n_head, 1, 1}, {32, 1, 1})
          enc.end_encoding
          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_f32(out_buf, q_dim)
          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_attn(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          result
        end

        # Fused attention decode + output projection in a single command
        # buffer. This removes one CPU round-trip from the full-attention
        # layers: we keep attn_o GPU-resident and immediately feed it into
        # the quantized output projection.
        #
        # Returns nil when `out_qw` is not one of the Metal-routable K-quants.
        def self.attn_decode_project(q : Array(Float32),
                                     gate : Array(Float32),
                                     k_cache_buf : ML::MetalBuffer,
                                     v_cache_buf : ML::MetalBuffer,
                                     out_qw : QuantWeight,
                                     pos : Int32, n_head : Int32, n_head_kv : Int32,
                                     head_dim : Int32, heads_per_group : Int32,
                                     scale : Float32) : Array(Float32)?
          pipeline = gemv_pipeline_for(out_qw)
          return nil if pipeline.nil?

          ML::Metal::Device.init!

          q_dim = n_head * head_dim
          raise "attn output projection in_dim mismatch: expected #{q_dim}, got #{out_qw.in_dim}" unless out_qw.in_dim == q_dim

          t0 = Time.instant if Profile.enabled?
          q_bytes   = q_dim.to_i64 * sizeof(Float32)
          out_bytes = out_qw.out_dim.to_i64 * sizeof(Float32)
          q_buf     = Scratch.get(:attn_q,         q_bytes)
          gate_buf  = Scratch.get(:attn_gate,      q_bytes)
          attn_buf  = Scratch.get(:attn_proj_mid,  q_bytes)
          proj_buf  = Scratch.get(:attn_proj_out,  out_bytes)
          q_buf.write(q)
          gate_buf.write(gate)

          cache_len = pos + 1
          w_buf, w_off = if slot = mmap_slot_for(out_qw.raw)
                           slot
                         else
                           {out_qw.fallback_metal_buffer, 0_i64}
                         end

          cmd = ML::Metal::CommandBuffer.new

          attn_enc = ML::Metal::ComputeEncoder.new(cmd)
          attn_enc.set_pipeline(attn_pipeline)
          attn_enc.set_buffer(q_buf,         0)
          attn_enc.set_buffer(gate_buf,      1)
          attn_enc.set_buffer(k_cache_buf,   2)
          attn_enc.set_buffer(v_cache_buf,   3)
          attn_enc.set_buffer(attn_buf,      4, ML::Metal::BufferAccess::Write)
          attn_enc.set_value(cache_len.to_u32,       5)
          attn_enc.set_value(n_head.to_u32,          6)
          attn_enc.set_value(n_head_kv.to_u32,       7)
          attn_enc.set_value(head_dim.to_u32,        8)
          attn_enc.set_value(heads_per_group.to_u32, 9)
          attn_enc.set_value(scale,                 10)
          attn_enc.dispatch_threadgroups({n_head, 1, 1}, {32, 1, 1})
          attn_enc.end_encoding

          proj_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(proj_enc, pipeline.not_nil!, attn_buf, proj_buf, w_buf, w_off, out_qw.in_dim, out_qw.out_dim)
          proj_enc.end_encoding

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_f32(proj_buf, out_qw.out_dim)
          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_attn(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          result
        end

        # Full-attention route with GPU prep:
        #   q/k/v projections -> split q+gate -> q/k RMSNorm -> RoPE ->
        #   KV write -> attention -> output projection, all on GPU.
        def self.full_attn_project(x : Array(Float32),
                                   q_qw : QuantWeight,
                                   k_qw : QuantWeight,
                                   v_qw : QuantWeight,
                                   q_norm : Array(Float32),
                                   k_norm : Array(Float32),
                                   out_qw : QuantWeight,
                                   k_cache_buf : ML::MetalBuffer,
                                   v_cache_buf : ML::MetalBuffer,
                                   pos : Int32,
                                   n_head : Int32,
                                   n_head_kv : Int32,
                                   head_dim : Int32,
                                   rope_dim_count : Int32,
                                   heads_per_group : Int32,
                                   rope_freq_base : Float32,
                                   scale : Float32) : Array(Float32)?
          q_pipe = gemv_pipeline_for(q_qw)
          k_pipe = gemv_pipeline_for(k_qw)
          v_pipe = gemv_pipeline_for(v_qw)
          out_pipe = gemv_pipeline_for(out_qw)
          return nil if q_pipe.nil? || k_pipe.nil? || v_pipe.nil? || out_pipe.nil?

          ML::Metal::Device.init!

          q_dim = n_head * head_dim
          kv_dim = n_head_kv * head_dim
          x_buf      = Scratch.get(:fattn_x,      x.size.to_i64 * sizeof(Float32))
          qfull_buf  = Scratch.get(:fattn_qfull,  q_qw.out_dim.to_i64 * sizeof(Float32))
          q_buf      = Scratch.get(:fattn_q,      q_dim.to_i64 * sizeof(Float32))
          gate_buf   = Scratch.get(:fattn_gate,   q_dim.to_i64 * sizeof(Float32))
          k_buf      = Scratch.get(:fattn_k,      kv_dim.to_i64 * sizeof(Float32))
          v_buf      = Scratch.get(:fattn_v,      kv_dim.to_i64 * sizeof(Float32))
          attn_buf   = Scratch.get(:fattn_attn,   q_dim.to_i64 * sizeof(Float32))
          out_buf    = Scratch.get(:fattn_out,    out_qw.out_dim.to_i64 * sizeof(Float32))
          qnorm_buf  = Scratch.get(:fattn_qnorm,  q_norm.size.to_i64 * sizeof(Float32))
          knorm_buf  = Scratch.get(:fattn_knorm,  k_norm.size.to_i64 * sizeof(Float32))
          x_buf.write(x)
          qnorm_buf.write(q_norm)
          knorm_buf.write(k_norm)

          q_w_buf, q_w_off = if slot = mmap_slot_for(q_qw.raw)
                               slot
                             else
                               {q_qw.fallback_metal_buffer, 0_i64}
                             end
          k_w_buf, k_w_off = if slot = mmap_slot_for(k_qw.raw)
                               slot
                             else
                               {k_qw.fallback_metal_buffer, 0_i64}
                             end
          v_w_buf, v_w_off = if slot = mmap_slot_for(v_qw.raw)
                               slot
                             else
                               {v_qw.fallback_metal_buffer, 0_i64}
                             end
          out_w_buf, out_w_off = if slot = mmap_slot_for(out_qw.raw)
                                   slot
                                 else
                                   {out_qw.fallback_metal_buffer, 0_i64}
                                 end

          t0 = Time.instant if Profile.enabled?
          cmd = ML::Metal::CommandBuffer.new

          proj_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(proj_enc, q_pipe.not_nil!, x_buf, qfull_buf, q_w_buf, q_w_off, q_qw.in_dim, q_qw.out_dim)
          if q8_kv_dual_gemv_candidate?(k_qw, v_qw)
            encode_gemv_q8_dual(proj_enc, x_buf, k_buf, v_buf,
              k_w_buf, k_w_off, v_w_buf, v_w_off, k_qw.in_dim, k_qw.out_dim)
          else
            encode_gemv(proj_enc, k_pipe.not_nil!, x_buf, k_buf,     k_w_buf, k_w_off, k_qw.in_dim, k_qw.out_dim)
            encode_gemv(proj_enc, v_pipe.not_nil!, x_buf, v_buf,     v_w_buf, v_w_off, v_qw.in_dim, v_qw.out_dim)
          end
          proj_enc.end_encoding

          split_enc = ML::Metal::ComputeEncoder.new(cmd)
          split_enc.set_pipeline(split_qgate_pipeline)
          split_enc.set_buffer(qfull_buf, 0)
          split_enc.set_buffer(q_buf,     1, ML::Metal::BufferAccess::Write)
          split_enc.set_buffer(gate_buf,  2, ML::Metal::BufferAccess::Write)
          split_enc.set_value(n_head.to_u32,   3)
          split_enc.set_value(head_dim.to_u32, 4)
          split_enc.dispatch_1d(q_dim, 256)
          split_enc.end_encoding

          qnorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          qnorm_enc.set_pipeline(rmsnorm_heads_pipeline)
          qnorm_enc.set_buffer(q_buf,     0, ML::Metal::BufferAccess::ReadWrite)
          qnorm_enc.set_buffer(qnorm_buf, 1)
          qnorm_enc.set_value(head_dim.to_u32, 2)
          qnorm_enc.set_value(1.0e-6_f32, 3)
          qnorm_enc.dispatch_threadgroups({n_head, 1, 1}, {32, 1, 1})
          qnorm_enc.end_encoding

          knorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          knorm_enc.set_pipeline(rmsnorm_heads_pipeline)
          knorm_enc.set_buffer(k_buf,     0, ML::Metal::BufferAccess::ReadWrite)
          knorm_enc.set_buffer(knorm_buf, 1)
          knorm_enc.set_value(head_dim.to_u32, 2)
          knorm_enc.set_value(1.0e-6_f32, 3)
          knorm_enc.dispatch_threadgroups({n_head_kv, 1, 1}, {32, 1, 1})
          knorm_enc.end_encoding

          qrope_enc = ML::Metal::ComputeEncoder.new(cmd)
          qrope_enc.set_pipeline(rope_partial_pipeline)
          qrope_enc.set_buffer(q_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          qrope_enc.set_value(head_dim.to_u32,       1)
          qrope_enc.set_value(rope_dim_count.to_u32, 2)
          qrope_enc.set_value(pos.to_u32,            3)
          qrope_enc.set_value(rope_freq_base,        4)
          qrope_enc.dispatch_threadgroups({n_head, 1, 1}, {32, 1, 1})
          qrope_enc.end_encoding

          krope_enc = ML::Metal::ComputeEncoder.new(cmd)
          krope_enc.set_pipeline(rope_partial_pipeline)
          krope_enc.set_buffer(k_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          krope_enc.set_value(head_dim.to_u32,       1)
          krope_enc.set_value(rope_dim_count.to_u32, 2)
          krope_enc.set_value(pos.to_u32,            3)
          krope_enc.set_value(rope_freq_base,        4)
          krope_enc.dispatch_threadgroups({n_head_kv, 1, 1}, {32, 1, 1})
          krope_enc.end_encoding

          kvwrite_enc = ML::Metal::ComputeEncoder.new(cmd)
          kvwrite_enc.set_pipeline(kv_write_pipeline)
          kvwrite_enc.set_buffer(k_buf,       0)
          kvwrite_enc.set_buffer(v_buf,       1)
          kvwrite_enc.set_buffer(k_cache_buf, 2, ML::Metal::BufferAccess::ReadWrite)
          kvwrite_enc.set_buffer(v_cache_buf, 3, ML::Metal::BufferAccess::ReadWrite)
          kvwrite_enc.set_value((pos * kv_dim).to_u32, 4)
          kvwrite_enc.set_value(kv_dim.to_u32,         5)
          kvwrite_enc.dispatch_1d(kv_dim, 256)
          kvwrite_enc.end_encoding

          attn_enc = ML::Metal::ComputeEncoder.new(cmd)
          attn_enc.set_pipeline(attn_pipeline)
          attn_enc.set_buffer(q_buf,         0)
          attn_enc.set_buffer(gate_buf,      1)
          attn_enc.set_buffer(k_cache_buf,   2)
          attn_enc.set_buffer(v_cache_buf,   3)
          attn_enc.set_buffer(attn_buf,      4, ML::Metal::BufferAccess::Write)
          attn_enc.set_value((pos + 1).to_u32,       5)
          attn_enc.set_value(n_head.to_u32,          6)
          attn_enc.set_value(n_head_kv.to_u32,       7)
          attn_enc.set_value(head_dim.to_u32,        8)
          attn_enc.set_value(heads_per_group.to_u32, 9)
          attn_enc.set_value(scale,                 10)
          attn_enc.dispatch_threadgroups({n_head, 1, 1}, {32, 1, 1})
          attn_enc.end_encoding

          outproj_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(outproj_enc, out_pipe.not_nil!, attn_buf, out_buf, out_w_buf, out_w_off, out_qw.in_dim, out_qw.out_dim)
          outproj_enc.end_encoding

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_f32(out_buf, out_qw.out_dim)
          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_attn(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          result
        end

        # Fused full-attention layer route:
        #   q/k/v projections -> split/norm/rope -> kv write -> attention ->
        #   output projection -> residual add + post-attn RMSNorm ->
        #   SwiGLU FFN -> final residual add
        # in one command buffer, with only the final layer output read back.
        def self.full_attn_layer_project(inp : Array(Float32),
                                         cur : Array(Float32),
                                         q_qw : QuantWeight,
                                         k_qw : QuantWeight,
                                         v_qw : QuantWeight,
                                         q_norm : Array(Float32),
                                         k_norm : Array(Float32),
                                         out_qw : QuantWeight,
                                         k_cache_buf : ML::MetalBuffer,
                                         v_cache_buf : ML::MetalBuffer,
                                         post_attention_norm : Array(Float32),
                                         ffn_gate_qw : QuantWeight,
                                         ffn_up_qw : QuantWeight,
                                         ffn_down_qw : QuantWeight,
                                         pos : Int32,
                                         n_head : Int32,
                                         n_head_kv : Int32,
                                         head_dim : Int32,
                                         rope_dim_count : Int32,
                                         heads_per_group : Int32,
                                         rope_freq_base : Float32,
                                         eps : Float32,
                                         scale : Float32) : Array(Float32)?
          q_pipe = gemv_pipeline_for(q_qw)
          k_pipe = gemv_pipeline_for(k_qw)
          v_pipe = gemv_pipeline_for(v_qw)
          out_pipe = gemv_pipeline_for(out_qw)
          ffn_gate_pipe = gemv_pipeline_for(ffn_gate_qw)
          ffn_up_pipe = gemv_pipeline_for(ffn_up_qw)
          ffn_down_pipe = gemv_pipeline_for(ffn_down_qw)
          return nil if q_pipe.nil? || k_pipe.nil? || v_pipe.nil? || out_pipe.nil? ||
                        ffn_gate_pipe.nil? || ffn_up_pipe.nil? || ffn_down_pipe.nil?

          ML::Metal::Device.init!

          hidden_dim = inp.size
          q_dim = n_head * head_dim
          kv_dim = n_head_kv * head_dim
          ffn_dim = ffn_gate_qw.out_dim

          inp_buf = Scratch.get(:fsl_inp, hidden_dim.to_i64 * sizeof(Float32))
          cur_buf = Scratch.get(:fsl_cur, cur.size.to_i64 * sizeof(Float32))
          qfull_buf = Scratch.get(:fsl_qfull, q_qw.out_dim.to_i64 * sizeof(Float32))
          q_buf = Scratch.get(:fsl_q, q_dim.to_i64 * sizeof(Float32))
          gate_buf = Scratch.get(:fsl_gate, q_dim.to_i64 * sizeof(Float32))
          k_buf = Scratch.get(:fsl_k, kv_dim.to_i64 * sizeof(Float32))
          v_buf = Scratch.get(:fsl_v, kv_dim.to_i64 * sizeof(Float32))
          attn_buf = Scratch.get(:fsl_attn, q_dim.to_i64 * sizeof(Float32))
          attn_out_buf = Scratch.get(:fsl_attn_out, out_qw.out_dim.to_i64 * sizeof(Float32))
          qnorm_buf = Scratch.get(:fsl_qnorm, q_norm.size.to_i64 * sizeof(Float32))
          knorm_buf = Scratch.get(:fsl_knorm, k_norm.size.to_i64 * sizeof(Float32))
          post_norm_buf = Scratch.get(:fsl_postnorm_w, post_attention_norm.size.to_i64 * sizeof(Float32))
          residual_buf = Scratch.get(:fsl_residual, hidden_dim.to_i64 * sizeof(Float32))
          normed_buf = Scratch.get(:fsl_normed, hidden_dim.to_i64 * sizeof(Float32))
          ffn_gate_buf = Scratch.get(:fsl_ffn_gate, ffn_dim.to_i64 * sizeof(Float32))
          ffn_up_buf = Scratch.get(:fsl_ffn_up, ffn_dim.to_i64 * sizeof(Float32))
          ffn_comb_buf = Scratch.get(:fsl_ffn_comb, ffn_dim.to_i64 * sizeof(Float32))
          ffn_out_buf = Scratch.get(:fsl_ffn_out, ffn_down_qw.out_dim.to_i64 * sizeof(Float32))
          out_buf = Scratch.get(:fsl_out, hidden_dim.to_i64 * sizeof(Float32))

          inp_buf.write(inp)
          cur_buf.write(cur)
          qnorm_buf.write(q_norm)
          knorm_buf.write(k_norm)
          post_norm_buf.write(post_attention_norm)

          q_w_buf, q_w_off = if slot = mmap_slot_for(q_qw.raw)
                               slot
                             else
                               {q_qw.fallback_metal_buffer, 0_i64}
                             end
          k_w_buf, k_w_off = if slot = mmap_slot_for(k_qw.raw)
                               slot
                             else
                               {k_qw.fallback_metal_buffer, 0_i64}
                             end
          v_w_buf, v_w_off = if slot = mmap_slot_for(v_qw.raw)
                               slot
                             else
                               {v_qw.fallback_metal_buffer, 0_i64}
                             end
          out_w_buf, out_w_off = if slot = mmap_slot_for(out_qw.raw)
                                   slot
                                 else
                                   {out_qw.fallback_metal_buffer, 0_i64}
                                 end
          ffn_gate_w_buf, ffn_gate_w_off = if slot = mmap_slot_for(ffn_gate_qw.raw)
                                             slot
                                           else
                                             {ffn_gate_qw.fallback_metal_buffer, 0_i64}
                                           end
          ffn_up_w_buf, ffn_up_w_off = if slot = mmap_slot_for(ffn_up_qw.raw)
                                         slot
                                       else
                                         {ffn_up_qw.fallback_metal_buffer, 0_i64}
                                       end
          ffn_down_w_buf, ffn_down_w_off = if slot = mmap_slot_for(ffn_down_qw.raw)
                                             slot
                                           else
                                             {ffn_down_qw.fallback_metal_buffer, 0_i64}
                                           end

          t0 = Time.instant if Profile.enabled?
          cmd = ML::Metal::CommandBuffer.new

          proj_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(proj_enc, q_pipe.not_nil!, cur_buf, qfull_buf, q_w_buf, q_w_off, q_qw.in_dim, q_qw.out_dim)
          if q8_kv_dual_gemv_candidate?(k_qw, v_qw)
            encode_gemv_q8_dual(proj_enc, cur_buf, k_buf, v_buf,
              k_w_buf, k_w_off, v_w_buf, v_w_off, k_qw.in_dim, k_qw.out_dim)
          else
            encode_gemv(proj_enc, k_pipe.not_nil!, cur_buf, k_buf, k_w_buf, k_w_off, k_qw.in_dim, k_qw.out_dim)
            encode_gemv(proj_enc, v_pipe.not_nil!, cur_buf, v_buf, v_w_buf, v_w_off, v_qw.in_dim, v_qw.out_dim)
          end
          proj_enc.end_encoding

          split_enc = ML::Metal::ComputeEncoder.new(cmd)
          split_enc.set_pipeline(split_qgate_pipeline)
          split_enc.set_buffer(qfull_buf, 0)
          split_enc.set_buffer(q_buf, 1, ML::Metal::BufferAccess::Write)
          split_enc.set_buffer(gate_buf, 2, ML::Metal::BufferAccess::Write)
          split_enc.set_value(n_head.to_u32, 3)
          split_enc.set_value(head_dim.to_u32, 4)
          split_enc.dispatch_1d(q_dim, 256)
          split_enc.end_encoding

          qnorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          qnorm_enc.set_pipeline(rmsnorm_heads_pipeline)
          qnorm_enc.set_buffer(q_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          qnorm_enc.set_buffer(qnorm_buf, 1)
          qnorm_enc.set_value(head_dim.to_u32, 2)
          qnorm_enc.set_value(eps, 3)
          qnorm_enc.dispatch_threadgroups({n_head, 1, 1}, {32, 1, 1})
          qnorm_enc.end_encoding

          knorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          knorm_enc.set_pipeline(rmsnorm_heads_pipeline)
          knorm_enc.set_buffer(k_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          knorm_enc.set_buffer(knorm_buf, 1)
          knorm_enc.set_value(head_dim.to_u32, 2)
          knorm_enc.set_value(eps, 3)
          knorm_enc.dispatch_threadgroups({n_head_kv, 1, 1}, {32, 1, 1})
          knorm_enc.end_encoding

          qrope_enc = ML::Metal::ComputeEncoder.new(cmd)
          qrope_enc.set_pipeline(rope_partial_pipeline)
          qrope_enc.set_buffer(q_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          qrope_enc.set_value(head_dim.to_u32, 1)
          qrope_enc.set_value(rope_dim_count.to_u32, 2)
          qrope_enc.set_value(pos.to_u32, 3)
          qrope_enc.set_value(rope_freq_base, 4)
          qrope_enc.dispatch_threadgroups({n_head, 1, 1}, {32, 1, 1})
          qrope_enc.end_encoding

          krope_enc = ML::Metal::ComputeEncoder.new(cmd)
          krope_enc.set_pipeline(rope_partial_pipeline)
          krope_enc.set_buffer(k_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          krope_enc.set_value(head_dim.to_u32, 1)
          krope_enc.set_value(rope_dim_count.to_u32, 2)
          krope_enc.set_value(pos.to_u32, 3)
          krope_enc.set_value(rope_freq_base, 4)
          krope_enc.dispatch_threadgroups({n_head_kv, 1, 1}, {32, 1, 1})
          krope_enc.end_encoding

          kvwrite_enc = ML::Metal::ComputeEncoder.new(cmd)
          kvwrite_enc.set_pipeline(kv_write_pipeline)
          kvwrite_enc.set_buffer(k_buf, 0)
          kvwrite_enc.set_buffer(v_buf, 1)
          kvwrite_enc.set_buffer(k_cache_buf, 2, ML::Metal::BufferAccess::ReadWrite)
          kvwrite_enc.set_buffer(v_cache_buf, 3, ML::Metal::BufferAccess::ReadWrite)
          kvwrite_enc.set_value((pos * kv_dim).to_u32, 4)
          kvwrite_enc.set_value(kv_dim.to_u32, 5)
          kvwrite_enc.dispatch_1d(kv_dim, 256)
          kvwrite_enc.end_encoding

          attn_enc = ML::Metal::ComputeEncoder.new(cmd)
          attn_enc.set_pipeline(attn_pipeline)
          attn_enc.set_buffer(q_buf, 0)
          attn_enc.set_buffer(gate_buf, 1)
          attn_enc.set_buffer(k_cache_buf, 2)
          attn_enc.set_buffer(v_cache_buf, 3)
          attn_enc.set_buffer(attn_buf, 4, ML::Metal::BufferAccess::Write)
          attn_enc.set_value((pos + 1).to_u32, 5)
          attn_enc.set_value(n_head.to_u32, 6)
          attn_enc.set_value(n_head_kv.to_u32, 7)
          attn_enc.set_value(head_dim.to_u32, 8)
          attn_enc.set_value(heads_per_group.to_u32, 9)
          attn_enc.set_value(scale, 10)
          attn_enc.dispatch_threadgroups({n_head, 1, 1}, {32, 1, 1})
          attn_enc.end_encoding

          outproj_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(outproj_enc, out_pipe.not_nil!, attn_buf, attn_out_buf, out_w_buf, out_w_off, out_qw.in_dim, out_qw.out_dim)
          outproj_enc.end_encoding

          addnorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          addnorm_enc.set_pipeline(add_rmsnorm_pipeline)
          addnorm_enc.set_buffer(inp_buf, 0)
          addnorm_enc.set_buffer(attn_out_buf, 1)
          addnorm_enc.set_buffer(post_norm_buf, 2)
          addnorm_enc.set_buffer(residual_buf, 3, ML::Metal::BufferAccess::Write)
          addnorm_enc.set_buffer(normed_buf, 4, ML::Metal::BufferAccess::Write)
          addnorm_enc.set_value(hidden_dim.to_u32, 5)
          addnorm_enc.set_value(eps, 6)
          addnorm_enc.dispatch_threadgroups({1, 1, 1}, {256, 1, 1})
          addnorm_enc.end_encoding

          ffn_proj_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(ffn_proj_enc, ffn_gate_pipe.not_nil!, normed_buf, ffn_gate_buf, ffn_gate_w_buf, ffn_gate_w_off, ffn_gate_qw.in_dim, ffn_gate_qw.out_dim)
          encode_gemv(ffn_proj_enc, ffn_up_pipe.not_nil!, normed_buf, ffn_up_buf, ffn_up_w_buf, ffn_up_w_off, ffn_up_qw.in_dim, ffn_up_qw.out_dim)
          ffn_proj_enc.end_encoding

          swiglu_enc = ML::Metal::ComputeEncoder.new(cmd)
          swiglu_enc.set_pipeline(ffn_swiglu_pipeline)
          swiglu_enc.set_buffer(ffn_gate_buf, 0)
          swiglu_enc.set_buffer(ffn_up_buf, 1)
          swiglu_enc.set_buffer(ffn_comb_buf, 2, ML::Metal::BufferAccess::Write)
          swiglu_enc.set_value(ffn_dim.to_u32, 3)
          swiglu_enc.dispatch_1d(ffn_dim, 256)
          swiglu_enc.end_encoding

          ffn_down_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(ffn_down_enc, ffn_down_pipe.not_nil!, ffn_comb_buf, ffn_out_buf, ffn_down_w_buf, ffn_down_w_off, ffn_down_qw.in_dim, ffn_down_qw.out_dim)
          ffn_down_enc.end_encoding

          add_enc = ML::Metal::ComputeEncoder.new(cmd)
          add_enc.set_pipeline(add_vec_pipeline)
          add_enc.set_buffer(residual_buf, 0)
          add_enc.set_buffer(ffn_out_buf, 1)
          add_enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          add_enc.set_value(hidden_dim.to_u32, 3)
          add_enc.dispatch_1d(hidden_dim, 256)
          add_enc.end_encoding

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_f32(out_buf, hidden_dim)
          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_attn(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          result
        end

        # Compile the complete ordinary F16 KV ABI before a caller allocates
        # or mutates typed cache state. A single missing pipeline must reject
        # the route instead of falling back to an F32 cache kernel.
        def self.kv_cache_f16_pipelines_supported? : Bool
          ML::Metal::Device.init!
          kv_write_h16_pipeline
          kv_write_rows_h16_pipeline
          attn_h16_pipeline
          attn_gqa4_h16_pipeline
          attn_splitk_stage1_h16_pipeline
          attn_rows_h16_pipeline
          attn_rows_sg4_h16_pipeline
          attn_rows_sg4_pregate_h16_pipeline
          case override = ENV["QWEN35_PREFILL_ATTN_FLASH_D256"]?
          when nil, "1"
            # The exact-shape policy is automatic only on M2 Max. Compile the
            # optional pipeline here, before typed F16 KV state is mutated, so
            # a Metal compiler/runtime rejection becomes ordinary route
            # rejection rather than a late prefill failure.
            if ML::Metal::Device.instance.name == "Apple M2 Max"
              attn_flash_d256_pipeline
            end
          when "0"
          else
            raise ArgumentError.new("QWEN35_PREFILL_ATTN_FLASH_D256 must be 0 or 1")
          end
          true
        rescue
          false
        end

        # Compile every pipeline selected by the exact terminal-layer route
        # before its caller mutates recurrent or KV state. Pipeline creation is
        # lazy and raises on failure, so this admission query must be total.
        def self.full_attn_layer_chunk_project_last_supported?(q_qw : QuantWeight,
                                                                k_qw : QuantWeight,
                                                                v_qw : QuantWeight,
                                                                out_qw : QuantWeight,
                                                                ffn_gate_qw : QuantWeight,
                                                                ffn_up_qw : QuantWeight,
                                                                ffn_down_qw : QuantWeight,
                                                                n_tokens : Int32,
                                                                kv_cache_f16 : Bool = false) : Bool
          return false unless n_tokens > 0

          ML::Metal::Device.init!
          return false unless gemv_pipeline_for(q_qw)
          return false unless matmul_pipeline_supported?(k_qw, n_tokens)
          return false unless matmul_pipeline_supported?(v_qw, n_tokens)
          return false unless gemv_route_pipeline_supported?(out_qw, 1, nil)
          return false unless gemv_route_pipeline_supported?(ffn_gate_qw, 1, nil)
          return false unless gemv_route_pipeline_supported?(ffn_up_qw, 1, nil)
          return false unless gemv_route_pipeline_supported?(ffn_down_qw, 1, nil)

          rmsnorm_rows_pipeline
          split_qgate_pipeline
          rmsnorm_heads_pipeline
          rmsnorm_heads_rows_pipeline
          rope_partial_pipeline
          rope_partial_rows_pipeline
          if kv_cache_f16
            kv_write_rows_h16_pipeline
            attn_h16_pipeline
            attn_rows_h16_pipeline
            attn_rows_sg4_h16_pipeline
            attn_rows_sg4_pregate_h16_pipeline
          else
            kv_write_rows_pipeline
            attn_pipeline
          end
          add_rmsnorm_pipeline
          ffn_swiglu_pipeline
          add_vec_pipeline
          true
        rescue
          false
        end

        # Final full-attention prefill specialization.
        #
        # The final decoder layer only needs K/V cache updates for every prompt
        # row; only the last row's hidden state is needed for next-token logits.
        # This route therefore projects K/V for the whole chunk, but computes
        # Q/attention/FFN output only for the final row.
        def self.full_attn_layer_chunk_project_last(inp : Array(Float32),
                                                    q_qw : QuantWeight,
                                                    k_qw : QuantWeight,
                                                    v_qw : QuantWeight,
                                                    attn_norm : Array(Float32),
                                                    q_norm : Array(Float32),
                                                    k_norm : Array(Float32),
                                                    out_qw : QuantWeight,
                                                    k_cache_buf : ML::MetalBuffer,
                                                    v_cache_buf : ML::MetalBuffer,
                                                    post_attention_norm : Array(Float32),
                                                    ffn_gate_qw : QuantWeight,
                                                    ffn_up_qw : QuantWeight,
                                                    ffn_down_qw : QuantWeight,
                                                    start_pos : Int32,
                                                    n_tokens : Int32,
                                                    n_head : Int32,
                                                    n_head_kv : Int32,
                                                    head_dim : Int32,
                                                    rope_dim_count : Int32,
                                                    heads_per_group : Int32,
                                                    rope_freq_base : Float32,
                                                    eps : Float32,
                                                    scale : Float32,
                                                    input_buf : ML::MetalBuffer? = nil,
                                                    kv_cache_f16 : Bool = false) : Array(Float32)?
          return nil unless full_attn_layer_chunk_project_last_supported?(
            q_qw, k_qw, v_qw, out_qw, ffn_gate_qw, ffn_up_qw, ffn_down_qw, n_tokens,
            kv_cache_f16,
          )
          q_pipe = gemv_pipeline_for(q_qw)
          k_pipe = gemv_pipeline_for(k_qw)
          v_pipe = gemv_pipeline_for(v_qw)
          out_pipe = gemv_pipeline_for(out_qw)
          ffn_gate_pipe = gemv_pipeline_for(ffn_gate_qw)
          ffn_up_pipe = gemv_pipeline_for(ffn_up_qw)
          ffn_down_pipe = gemv_pipeline_for(ffn_down_qw)
          return nil unless n_tokens > 0

          ML::Metal::Device.init!

          hidden_dim = q_qw.in_dim
          q_dim = n_head * head_dim
          kv_dim = n_head_kv * head_dim
          ffn_dim = ffn_gate_qw.out_dim
          hidden_bytes = (n_tokens * hidden_dim).to_i64 * sizeof(Float32)
          if ib = input_buf
            raise "final full-attn input buffer too small" if ib.size < hidden_bytes
          else
            raise "final full-attn input size mismatch" unless inp.size == n_tokens * hidden_dim
          end

          final_pos = start_pos + n_tokens - 1
          last_offset = (n_tokens - 1) * hidden_dim
          last_byte_offset = last_offset.to_i64 * sizeof(Float32)
          last_x = input_buf ? nil : inp[last_offset, hidden_dim]

          inp_buf = input_buf || Scratch.get(:full_last_inp, hidden_bytes)
          last_inp_buf = Scratch.get(:full_last_last_inp, hidden_dim.to_i64 * sizeof(Float32))
          norm_w_buf = Scratch.get(:full_last_norm_w, attn_norm.size.to_i64 * sizeof(Float32))
          cur_buf = Scratch.get(:full_last_cur, hidden_bytes)
          qfull_buf = Scratch.get(:full_last_qfull, q_qw.out_dim.to_i64 * sizeof(Float32))
          q_buf = Scratch.get(:full_last_q, q_dim.to_i64 * sizeof(Float32))
          gate_buf = Scratch.get(:full_last_gate, q_dim.to_i64 * sizeof(Float32))
          k_buf = Scratch.get(:full_last_k, (n_tokens * kv_dim).to_i64 * sizeof(Float32))
          v_buf = Scratch.get(:full_last_v, (n_tokens * kv_dim).to_i64 * sizeof(Float32))
          attn_buf = Scratch.get(:full_last_attn, q_dim.to_i64 * sizeof(Float32))
          attn_out_buf = Scratch.get(:full_last_attn_out, out_qw.out_dim.to_i64 * sizeof(Float32))
          qnorm_buf = Scratch.get(:full_last_qnorm, q_norm.size.to_i64 * sizeof(Float32))
          knorm_buf = Scratch.get(:full_last_knorm, k_norm.size.to_i64 * sizeof(Float32))
          post_norm_buf = Scratch.get(:full_last_postnorm_w, post_attention_norm.size.to_i64 * sizeof(Float32))
          residual_buf = Scratch.get(:full_last_residual, hidden_dim.to_i64 * sizeof(Float32))
          normed_buf = Scratch.get(:full_last_normed, hidden_dim.to_i64 * sizeof(Float32))
          ffn_gate_buf = Scratch.get(:full_last_ffn_gate, ffn_dim.to_i64 * sizeof(Float32))
          ffn_up_buf = Scratch.get(:full_last_ffn_up, ffn_dim.to_i64 * sizeof(Float32))
          ffn_comb_buf = Scratch.get(:full_last_ffn_comb, ffn_dim.to_i64 * sizeof(Float32))
          ffn_out_buf = Scratch.get(:full_last_ffn_out, ffn_down_qw.out_dim.to_i64 * sizeof(Float32))
          out_buf = Scratch.get(:full_last_out, hidden_dim.to_i64 * sizeof(Float32))

          inp_buf.write(inp) unless input_buf
          last_inp_buf.write(last_x.not_nil!) unless input_buf
          norm_w_buf.write(attn_norm)
          qnorm_buf.write(q_norm)
          knorm_buf.write(k_norm)
          post_norm_buf.write(post_attention_norm)

          q_w_buf, q_w_off = weight_slot(q_qw)
          k_w_buf, k_w_off = weight_slot(k_qw)
          v_w_buf, v_w_off = weight_slot(v_qw)
          out_w_buf, out_w_off = weight_slot(out_qw)
          ffn_gate_w_buf, ffn_gate_w_off = weight_slot(ffn_gate_qw)
          ffn_up_w_buf, ffn_up_w_off = weight_slot(ffn_up_qw)
          ffn_down_w_buf, ffn_down_w_off = weight_slot(ffn_down_qw)

          t0 = Time.instant if Profile.enabled?
          cmd = ML::Metal::CommandBuffer.new

          if input_buf
            blit = ML::Metal::BlitEncoder.new(cmd)
            blit.copy_buffer(inp_buf, last_byte_offset.to_i32, last_inp_buf, 0, (hidden_dim * sizeof(Float32)).to_i32)
            blit.end_encoding
          end

          norm_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_rmsnorm_rows(norm_enc, inp_buf, norm_w_buf, cur_buf, hidden_dim, n_tokens, eps)
          norm_enc.end_encoding

          proj_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv_input_offset(proj_enc, q_pipe.not_nil!, cur_buf, last_byte_offset, qfull_buf, q_w_buf, q_w_off, q_qw.in_dim, q_qw.out_dim)
          encode_matmul(proj_enc, k_pipe.not_nil!, k_qw, cur_buf, k_buf, k_w_buf, k_w_off, k_qw.in_dim, k_qw.out_dim, n_tokens)
          encode_matmul(proj_enc, v_pipe.not_nil!, v_qw, cur_buf, v_buf, v_w_buf, v_w_off, v_qw.in_dim, v_qw.out_dim, n_tokens)
          proj_enc.end_encoding

          split_enc = ML::Metal::ComputeEncoder.new(cmd)
          split_enc.set_pipeline(split_qgate_pipeline)
          split_enc.set_buffer(qfull_buf, 0)
          split_enc.set_buffer(q_buf, 1, ML::Metal::BufferAccess::Write)
          split_enc.set_buffer(gate_buf, 2, ML::Metal::BufferAccess::Write)
          split_enc.set_value(n_head.to_u32, 3)
          split_enc.set_value(head_dim.to_u32, 4)
          split_enc.dispatch_1d(q_dim, 256)
          split_enc.end_encoding

          qnorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          qnorm_enc.set_pipeline(rmsnorm_heads_pipeline)
          qnorm_enc.set_buffer(q_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          qnorm_enc.set_buffer(qnorm_buf, 1)
          qnorm_enc.set_value(head_dim.to_u32, 2)
          qnorm_enc.set_value(eps, 3)
          qnorm_enc.dispatch_threadgroups({n_head, 1, 1}, {32, 1, 1})
          qnorm_enc.end_encoding

          knorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          knorm_enc.set_pipeline(rmsnorm_heads_rows_pipeline)
          knorm_enc.set_buffer(k_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          knorm_enc.set_buffer(knorm_buf, 1)
          knorm_enc.set_value(head_dim.to_u32, 2)
          knorm_enc.set_value(eps, 3)
          knorm_enc.set_value(n_head_kv.to_u32, 4)
          knorm_enc.set_value(n_tokens.to_u32, 5)
          knorm_enc.dispatch_threadgroups({n_head_kv, n_tokens, 1}, {32, 1, 1})
          knorm_enc.end_encoding

          qrope_enc = ML::Metal::ComputeEncoder.new(cmd)
          qrope_enc.set_pipeline(rope_partial_pipeline)
          qrope_enc.set_buffer(q_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          qrope_enc.set_value(head_dim.to_u32, 1)
          qrope_enc.set_value(rope_dim_count.to_u32, 2)
          qrope_enc.set_value(final_pos.to_u32, 3)
          qrope_enc.set_value(rope_freq_base, 4)
          qrope_enc.dispatch_threadgroups({n_head, 1, 1}, {32, 1, 1})
          qrope_enc.end_encoding

          krope_enc = ML::Metal::ComputeEncoder.new(cmd)
          krope_enc.set_pipeline(rope_partial_rows_pipeline)
          krope_enc.set_buffer(k_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          krope_enc.set_value(head_dim.to_u32, 1)
          krope_enc.set_value(rope_dim_count.to_u32, 2)
          krope_enc.set_value(start_pos.to_u32, 3)
          krope_enc.set_value(rope_freq_base, 4)
          krope_enc.set_value(n_head_kv.to_u32, 5)
          krope_enc.set_value(n_tokens.to_u32, 6)
          krope_enc.dispatch_threadgroups({n_head_kv, n_tokens, 1}, {32, 1, 1})
          krope_enc.end_encoding

          kvwrite_enc = ML::Metal::ComputeEncoder.new(cmd)
          kvwrite_enc.set_pipeline(kv_cache_f16 ? kv_write_rows_h16_pipeline : kv_write_rows_pipeline)
          kvwrite_enc.set_buffer(k_buf, 0)
          kvwrite_enc.set_buffer(v_buf, 1)
          kvwrite_enc.set_buffer(k_cache_buf, 2, ML::Metal::BufferAccess::ReadWrite)
          kvwrite_enc.set_buffer(v_cache_buf, 3, ML::Metal::BufferAccess::ReadWrite)
          kvwrite_enc.set_value(start_pos.to_u32, 4)
          kvwrite_enc.set_value(kv_dim.to_u32, 5)
          kvwrite_enc.set_value(n_tokens.to_u32, 6)
          kvwrite_enc.dispatch_1d(n_tokens * kv_dim, 256)
          kvwrite_enc.end_encoding

          attn_enc = ML::Metal::ComputeEncoder.new(cmd)
          attn_enc.set_pipeline(kv_cache_f16 ? attn_h16_pipeline : attn_pipeline)
          attn_enc.set_buffer(q_buf, 0)
          attn_enc.set_buffer(gate_buf, 1)
          attn_enc.set_buffer(k_cache_buf, 2)
          attn_enc.set_buffer(v_cache_buf, 3)
          attn_enc.set_buffer(attn_buf, 4, ML::Metal::BufferAccess::Write)
          attn_enc.set_value((final_pos + 1).to_u32, 5)
          attn_enc.set_value(n_head.to_u32, 6)
          attn_enc.set_value(n_head_kv.to_u32, 7)
          attn_enc.set_value(head_dim.to_u32, 8)
          attn_enc.set_value(heads_per_group.to_u32, 9)
          attn_enc.set_value(scale, 10)
          attn_enc.dispatch_threadgroups({n_head, 1, 1}, {32, 1, 1})
          attn_enc.end_encoding

          outproj_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(outproj_enc, out_pipe.not_nil!, attn_buf, attn_out_buf, out_w_buf, out_w_off, out_qw.in_dim, out_qw.out_dim)
          outproj_enc.end_encoding

          addnorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          addnorm_enc.set_pipeline(add_rmsnorm_pipeline)
          addnorm_enc.set_buffer(last_inp_buf, 0)
          addnorm_enc.set_buffer(attn_out_buf, 1)
          addnorm_enc.set_buffer(post_norm_buf, 2)
          addnorm_enc.set_buffer(residual_buf, 3, ML::Metal::BufferAccess::Write)
          addnorm_enc.set_buffer(normed_buf, 4, ML::Metal::BufferAccess::Write)
          addnorm_enc.set_value(hidden_dim.to_u32, 5)
          addnorm_enc.set_value(eps, 6)
          addnorm_enc.dispatch_threadgroups({1, 1, 1}, {256, 1, 1})
          addnorm_enc.end_encoding

          ffn_proj_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(ffn_proj_enc, ffn_gate_pipe.not_nil!, normed_buf, ffn_gate_buf, ffn_gate_w_buf, ffn_gate_w_off, ffn_gate_qw.in_dim, ffn_gate_qw.out_dim)
          encode_gemv(ffn_proj_enc, ffn_up_pipe.not_nil!, normed_buf, ffn_up_buf, ffn_up_w_buf, ffn_up_w_off, ffn_up_qw.in_dim, ffn_up_qw.out_dim)
          ffn_proj_enc.end_encoding

          swiglu_enc = ML::Metal::ComputeEncoder.new(cmd)
          ffn_act_buf = swiglu_inplace_enabled? ? ffn_up_buf : ffn_comb_buf
          swiglu_enc.set_pipeline(ffn_swiglu_pipeline)
          swiglu_enc.set_buffer(ffn_gate_buf, 0)
          swiglu_enc.set_buffer(ffn_up_buf, 1)
          swiglu_enc.set_buffer(ffn_act_buf, 2, ML::Metal::BufferAccess::Write)
          swiglu_enc.set_value(ffn_dim.to_u32, 3)
          swiglu_enc.dispatch_1d(ffn_dim, 256)
          swiglu_enc.end_encoding

          ffn_down_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(ffn_down_enc, ffn_down_pipe.not_nil!, ffn_act_buf, ffn_out_buf, ffn_down_w_buf, ffn_down_w_off, ffn_down_qw.in_dim, ffn_down_qw.out_dim)
          ffn_down_enc.end_encoding

          add_enc = ML::Metal::ComputeEncoder.new(cmd)
          add_enc.set_pipeline(add_vec_pipeline)
          add_enc.set_buffer(residual_buf, 0)
          add_enc.set_buffer(ffn_out_buf, 1)
          add_enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          add_enc.set_value(hidden_dim.to_u32, 3)
          add_enc.dispatch_1d(hidden_dim, 256)
          add_enc.end_encoding

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_f32(out_buf, hidden_dim)
          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_attn(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          result
        end
        def self.full_attn_layer_chunk_project_last_top1(inp : Array(Float32),
                                                    q_qw : QuantWeight,
                                                    k_qw : QuantWeight,
                                                    v_qw : QuantWeight,
                                                    attn_norm : Array(Float32),
                                                    q_norm : Array(Float32),
                                                    k_norm : Array(Float32),
                                                    out_qw : QuantWeight,
                                                    k_cache_buf : ML::MetalBuffer,
                                                    v_cache_buf : ML::MetalBuffer,
                                                    post_attention_norm : Array(Float32),
                                                    ffn_gate_qw : QuantWeight,
                                                    ffn_up_qw : QuantWeight,
                                                    ffn_down_qw : QuantWeight,
                                                    start_pos : Int32,
                                                    n_tokens : Int32,
                                                    n_head : Int32,
                                                    n_head_kv : Int32,
                                                    head_dim : Int32,
                                                    rope_dim_count : Int32,
                                                    heads_per_group : Int32,
                                                    rope_freq_base : Float32,
                                                    eps : Float32,
                                                    scale : Float32,
                                                    output_norm : Array(Float32),
                                                    output_qw : QuantWeight,
                                                    input_buf : ML::MetalBuffer? = nil,
                                                    kv_cache_f16 : Bool = false) : {Int32, Float32}?
          return nil unless full_attn_layer_chunk_project_last_supported?(
            q_qw, k_qw, v_qw, out_qw, ffn_gate_qw, ffn_up_qw, ffn_down_qw, n_tokens,
            kv_cache_f16,
          )
          q_pipe = gemv_pipeline_for(q_qw)
          k_pipe = gemv_pipeline_for(k_qw)
          v_pipe = gemv_pipeline_for(v_qw)
          out_pipe = gemv_pipeline_for(out_qw)
          ffn_gate_pipe = gemv_pipeline_for(ffn_gate_qw)
          ffn_up_pipe = gemv_pipeline_for(ffn_up_qw)
          ffn_down_pipe = gemv_pipeline_for(ffn_down_qw)
          return nil if q_pipe.nil? || k_pipe.nil? || v_pipe.nil? || out_pipe.nil? ||
                        ffn_gate_pipe.nil? || ffn_up_pipe.nil? || ffn_down_pipe.nil?
          return nil unless can_use_head_top1_fused?(output_qw)
          return nil unless n_tokens > 0

          ML::Metal::Device.init!

          hidden_dim = q_qw.in_dim
          q_dim = n_head * head_dim
          kv_dim = n_head_kv * head_dim
          ffn_dim = ffn_gate_qw.out_dim
          tile_count = (output_qw.out_dim + HEAD_TOP1_ROWS_PER_TG - 1) // HEAD_TOP1_ROWS_PER_TG
          raise "final full-attn top1 output norm size mismatch" unless output_norm.size == hidden_dim
          raise "final full-attn top1 head shape mismatch" unless output_qw.in_dim == hidden_dim
          hidden_bytes = (n_tokens * hidden_dim).to_i64 * sizeof(Float32)
          if ib = input_buf
            raise "final full-attn input buffer too small" if ib.size < hidden_bytes
          else
            raise "final full-attn input size mismatch" unless inp.size == n_tokens * hidden_dim
          end

          final_pos = start_pos + n_tokens - 1
          last_offset = (n_tokens - 1) * hidden_dim
          last_byte_offset = last_offset.to_i64 * sizeof(Float32)
          last_x = input_buf ? nil : inp[last_offset, hidden_dim]

          inp_buf = input_buf || Scratch.get(:full_last_inp, hidden_bytes)
          last_inp_buf = Scratch.get(:full_last_last_inp, hidden_dim.to_i64 * sizeof(Float32))
          norm_w_buf = Scratch.get(:full_last_norm_w, attn_norm.size.to_i64 * sizeof(Float32))
          cur_buf = Scratch.get(:full_last_cur, hidden_bytes)
          qfull_buf = Scratch.get(:full_last_qfull, q_qw.out_dim.to_i64 * sizeof(Float32))
          q_buf = Scratch.get(:full_last_q, q_dim.to_i64 * sizeof(Float32))
          gate_buf = Scratch.get(:full_last_gate, q_dim.to_i64 * sizeof(Float32))
          k_buf = Scratch.get(:full_last_k, (n_tokens * kv_dim).to_i64 * sizeof(Float32))
          v_buf = Scratch.get(:full_last_v, (n_tokens * kv_dim).to_i64 * sizeof(Float32))
          attn_buf = Scratch.get(:full_last_attn, q_dim.to_i64 * sizeof(Float32))
          attn_out_buf = Scratch.get(:full_last_attn_out, out_qw.out_dim.to_i64 * sizeof(Float32))
          qnorm_buf = Scratch.get(:full_last_qnorm, q_norm.size.to_i64 * sizeof(Float32))
          knorm_buf = Scratch.get(:full_last_knorm, k_norm.size.to_i64 * sizeof(Float32))
          post_norm_buf = Scratch.get(:full_last_postnorm_w, post_attention_norm.size.to_i64 * sizeof(Float32))
          residual_buf = Scratch.get(:full_last_residual, hidden_dim.to_i64 * sizeof(Float32))
          normed_buf = Scratch.get(:full_last_normed, hidden_dim.to_i64 * sizeof(Float32))
          ffn_gate_buf = Scratch.get(:full_last_ffn_gate, ffn_dim.to_i64 * sizeof(Float32))
          ffn_up_buf = Scratch.get(:full_last_ffn_up, ffn_dim.to_i64 * sizeof(Float32))
          ffn_comb_buf = Scratch.get(:full_last_ffn_comb, ffn_dim.to_i64 * sizeof(Float32))
          ffn_out_buf = Scratch.get(:full_last_ffn_out, ffn_down_qw.out_dim.to_i64 * sizeof(Float32))
          out_buf = Scratch.get(:full_last_out, hidden_dim.to_i64 * sizeof(Float32))
          head_norm_w_buf = Scratch.get(:full_last_head_norm_w, output_norm.size.to_i64 * sizeof(Float32))
          head_normed_buf = Scratch.get(:full_last_head_normed, hidden_dim.to_i64 * sizeof(Float32))
          tile_values_buf = Scratch.get(:full_last_head_tile_values, tile_count.to_i64 * sizeof(Float32))
          tile_ids_buf = Scratch.get(:full_last_head_tile_ids, tile_count.to_i64 * sizeof(UInt32))
          top1_id_buf = Scratch.get(:full_last_head_top1_id, sizeof(UInt32).to_i64)
          top1_value_buf = Scratch.get(:full_last_head_top1_value, sizeof(Float32).to_i64)

          inp_buf.write(inp) unless input_buf
          last_inp_buf.write(last_x.not_nil!) unless input_buf
          norm_w_buf.write(attn_norm)
          qnorm_buf.write(q_norm)
          knorm_buf.write(k_norm)
          post_norm_buf.write(post_attention_norm)
          head_norm_w_buf.write(output_norm)

          q_w_buf, q_w_off = weight_slot(q_qw)
          k_w_buf, k_w_off = weight_slot(k_qw)
          v_w_buf, v_w_off = weight_slot(v_qw)
          out_w_buf, out_w_off = weight_slot(out_qw)
          ffn_gate_w_buf, ffn_gate_w_off = weight_slot(ffn_gate_qw)
          ffn_up_w_buf, ffn_up_w_off = weight_slot(ffn_up_qw)
          ffn_down_w_buf, ffn_down_w_off = weight_slot(ffn_down_qw)
          output_w_buf, output_w_off = weight_slot(output_qw)

          t0 = Time.instant if Profile.enabled?
          cmd = ML::Metal::CommandBuffer.new

          if input_buf
            blit = ML::Metal::BlitEncoder.new(cmd)
            blit.copy_buffer(inp_buf, last_byte_offset.to_i32, last_inp_buf, 0, (hidden_dim * sizeof(Float32)).to_i32)
            blit.end_encoding
          end

          norm_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_rmsnorm_rows(norm_enc, inp_buf, norm_w_buf, cur_buf, hidden_dim, n_tokens, eps)
          norm_enc.end_encoding

          proj_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv_input_offset(proj_enc, q_pipe.not_nil!, cur_buf, last_byte_offset, qfull_buf, q_w_buf, q_w_off, q_qw.in_dim, q_qw.out_dim)
          encode_matmul(proj_enc, k_pipe.not_nil!, k_qw, cur_buf, k_buf, k_w_buf, k_w_off, k_qw.in_dim, k_qw.out_dim, n_tokens)
          encode_matmul(proj_enc, v_pipe.not_nil!, v_qw, cur_buf, v_buf, v_w_buf, v_w_off, v_qw.in_dim, v_qw.out_dim, n_tokens)
          proj_enc.end_encoding

          split_enc = ML::Metal::ComputeEncoder.new(cmd)
          split_enc.set_pipeline(split_qgate_pipeline)
          split_enc.set_buffer(qfull_buf, 0)
          split_enc.set_buffer(q_buf, 1, ML::Metal::BufferAccess::Write)
          split_enc.set_buffer(gate_buf, 2, ML::Metal::BufferAccess::Write)
          split_enc.set_value(n_head.to_u32, 3)
          split_enc.set_value(head_dim.to_u32, 4)
          split_enc.dispatch_1d(q_dim, 256)
          split_enc.end_encoding

          qnorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          qnorm_enc.set_pipeline(rmsnorm_heads_pipeline)
          qnorm_enc.set_buffer(q_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          qnorm_enc.set_buffer(qnorm_buf, 1)
          qnorm_enc.set_value(head_dim.to_u32, 2)
          qnorm_enc.set_value(eps, 3)
          qnorm_enc.dispatch_threadgroups({n_head, 1, 1}, {32, 1, 1})
          qnorm_enc.end_encoding

          knorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          knorm_enc.set_pipeline(rmsnorm_heads_rows_pipeline)
          knorm_enc.set_buffer(k_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          knorm_enc.set_buffer(knorm_buf, 1)
          knorm_enc.set_value(head_dim.to_u32, 2)
          knorm_enc.set_value(eps, 3)
          knorm_enc.set_value(n_head_kv.to_u32, 4)
          knorm_enc.set_value(n_tokens.to_u32, 5)
          knorm_enc.dispatch_threadgroups({n_head_kv, n_tokens, 1}, {32, 1, 1})
          knorm_enc.end_encoding

          qrope_enc = ML::Metal::ComputeEncoder.new(cmd)
          qrope_enc.set_pipeline(rope_partial_pipeline)
          qrope_enc.set_buffer(q_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          qrope_enc.set_value(head_dim.to_u32, 1)
          qrope_enc.set_value(rope_dim_count.to_u32, 2)
          qrope_enc.set_value(final_pos.to_u32, 3)
          qrope_enc.set_value(rope_freq_base, 4)
          qrope_enc.dispatch_threadgroups({n_head, 1, 1}, {32, 1, 1})
          qrope_enc.end_encoding

          krope_enc = ML::Metal::ComputeEncoder.new(cmd)
          krope_enc.set_pipeline(rope_partial_rows_pipeline)
          krope_enc.set_buffer(k_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          krope_enc.set_value(head_dim.to_u32, 1)
          krope_enc.set_value(rope_dim_count.to_u32, 2)
          krope_enc.set_value(start_pos.to_u32, 3)
          krope_enc.set_value(rope_freq_base, 4)
          krope_enc.set_value(n_head_kv.to_u32, 5)
          krope_enc.set_value(n_tokens.to_u32, 6)
          krope_enc.dispatch_threadgroups({n_head_kv, n_tokens, 1}, {32, 1, 1})
          krope_enc.end_encoding

          kvwrite_enc = ML::Metal::ComputeEncoder.new(cmd)
          kvwrite_enc.set_pipeline(kv_cache_f16 ? kv_write_rows_h16_pipeline : kv_write_rows_pipeline)
          kvwrite_enc.set_buffer(k_buf, 0)
          kvwrite_enc.set_buffer(v_buf, 1)
          kvwrite_enc.set_buffer(k_cache_buf, 2, ML::Metal::BufferAccess::ReadWrite)
          kvwrite_enc.set_buffer(v_cache_buf, 3, ML::Metal::BufferAccess::ReadWrite)
          kvwrite_enc.set_value(start_pos.to_u32, 4)
          kvwrite_enc.set_value(kv_dim.to_u32, 5)
          kvwrite_enc.set_value(n_tokens.to_u32, 6)
          kvwrite_enc.dispatch_1d(n_tokens * kv_dim, 256)
          kvwrite_enc.end_encoding

          attn_enc = ML::Metal::ComputeEncoder.new(cmd)
          attn_enc.set_pipeline(kv_cache_f16 ? attn_h16_pipeline : attn_pipeline)
          attn_enc.set_buffer(q_buf, 0)
          attn_enc.set_buffer(gate_buf, 1)
          attn_enc.set_buffer(k_cache_buf, 2)
          attn_enc.set_buffer(v_cache_buf, 3)
          attn_enc.set_buffer(attn_buf, 4, ML::Metal::BufferAccess::Write)
          attn_enc.set_value((final_pos + 1).to_u32, 5)
          attn_enc.set_value(n_head.to_u32, 6)
          attn_enc.set_value(n_head_kv.to_u32, 7)
          attn_enc.set_value(head_dim.to_u32, 8)
          attn_enc.set_value(heads_per_group.to_u32, 9)
          attn_enc.set_value(scale, 10)
          attn_enc.dispatch_threadgroups({n_head, 1, 1}, {32, 1, 1})
          attn_enc.end_encoding

          outproj_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(outproj_enc, out_pipe.not_nil!, attn_buf, attn_out_buf, out_w_buf, out_w_off, out_qw.in_dim, out_qw.out_dim)
          outproj_enc.end_encoding

          addnorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          addnorm_enc.set_pipeline(add_rmsnorm_pipeline)
          addnorm_enc.set_buffer(last_inp_buf, 0)
          addnorm_enc.set_buffer(attn_out_buf, 1)
          addnorm_enc.set_buffer(post_norm_buf, 2)
          addnorm_enc.set_buffer(residual_buf, 3, ML::Metal::BufferAccess::Write)
          addnorm_enc.set_buffer(normed_buf, 4, ML::Metal::BufferAccess::Write)
          addnorm_enc.set_value(hidden_dim.to_u32, 5)
          addnorm_enc.set_value(eps, 6)
          addnorm_enc.dispatch_threadgroups({1, 1, 1}, {256, 1, 1})
          addnorm_enc.end_encoding

          ffn_proj_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(ffn_proj_enc, ffn_gate_pipe.not_nil!, normed_buf, ffn_gate_buf, ffn_gate_w_buf, ffn_gate_w_off, ffn_gate_qw.in_dim, ffn_gate_qw.out_dim)
          encode_gemv(ffn_proj_enc, ffn_up_pipe.not_nil!, normed_buf, ffn_up_buf, ffn_up_w_buf, ffn_up_w_off, ffn_up_qw.in_dim, ffn_up_qw.out_dim)
          ffn_proj_enc.end_encoding

          swiglu_enc = ML::Metal::ComputeEncoder.new(cmd)
          ffn_act_buf = swiglu_inplace_enabled? ? ffn_up_buf : ffn_comb_buf
          swiglu_enc.set_pipeline(ffn_swiglu_pipeline)
          swiglu_enc.set_buffer(ffn_gate_buf, 0)
          swiglu_enc.set_buffer(ffn_up_buf, 1)
          swiglu_enc.set_buffer(ffn_act_buf, 2, ML::Metal::BufferAccess::Write)
          swiglu_enc.set_value(ffn_dim.to_u32, 3)
          swiglu_enc.dispatch_1d(ffn_dim, 256)
          swiglu_enc.end_encoding

          ffn_down_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(ffn_down_enc, ffn_down_pipe.not_nil!, ffn_act_buf, ffn_out_buf, ffn_down_w_buf, ffn_down_w_off, ffn_down_qw.in_dim, ffn_down_qw.out_dim)
          ffn_down_enc.end_encoding

          add_enc = ML::Metal::ComputeEncoder.new(cmd)
          add_enc.set_pipeline(add_vec_pipeline)
          add_enc.set_buffer(residual_buf, 0)
          add_enc.set_buffer(ffn_out_buf, 1)
          add_enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          add_enc.set_value(hidden_dim.to_u32, 3)
          add_enc.dispatch_1d(hidden_dim, 256)
          add_enc.end_encoding

          head_norm_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_rmsnorm_vec(head_norm_enc, out_buf, head_norm_w_buf, head_normed_buf, hidden_dim, eps)
          head_norm_enc.end_encoding

          head_top1_enc = ML::Metal::ComputeEncoder.new(cmd)
          head_top1_enc.set_pipeline(output_qw.type.q8_0? ? mv8_top1_tiles_pipeline : mv6_top1_tiles_pipeline)
          head_top1_enc.set_buffer(output_w_buf, 0, ML::Metal::BufferAccess::Read, offset: output_w_off)
          head_top1_enc.set_buffer(head_normed_buf, 1)
          head_top1_enc.set_buffer(tile_values_buf, 2, ML::Metal::BufferAccess::Write)
          head_top1_enc.set_buffer(tile_ids_buf, 3, ML::Metal::BufferAccess::Write)
          head_top1_enc.set_value(output_qw.in_dim.to_u32, 4)
          head_top1_enc.set_value(output_qw.out_dim.to_u32, 5)
          head_top1_enc.dispatch_threadgroups({tile_count, 1, 1}, {output_qw.type.q8_0? ? MV_Q8_NSG * 32 : 64, 1, 1})
          head_top1_enc.end_encoding

          reduce_top1_enc = ML::Metal::ComputeEncoder.new(cmd)
          reduce_top1_enc.set_pipeline(top1_reduce_tiles_pipeline)
          reduce_top1_enc.set_buffer(tile_values_buf, 0)
          reduce_top1_enc.set_buffer(tile_ids_buf, 1)
          reduce_top1_enc.set_buffer(top1_id_buf, 2, ML::Metal::BufferAccess::Write)
          reduce_top1_enc.set_buffer(top1_value_buf, 3, ML::Metal::BufferAccess::Write)
          reduce_top1_enc.set_value(tile_count.to_u32, 4)
          reduce_top1_enc.dispatch_threadgroups({1, 1, 1}, {256, 1, 1})
          reduce_top1_enc.end_encoding

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_top1(top1_id_buf, top1_value_buf)
          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_attn(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          {result[0].to_i32, result[1]}
        end

        def self.full_attn_layer_chunk_project(inp : Array(Float32),
                                               q_qw : QuantWeight,
                                               k_qw : QuantWeight,
                                               v_qw : QuantWeight,
                                               attn_norm : Array(Float32),
                                               q_norm : Array(Float32),
                                               k_norm : Array(Float32),
                                               out_qw : QuantWeight,
                                               k_cache_buf : ML::MetalBuffer?,
                                               v_cache_buf : ML::MetalBuffer?,
                                               post_attention_norm : Array(Float32),
                                               ffn_gate_qw : QuantWeight,
                                               ffn_up_qw : QuantWeight,
                                               ffn_down_qw : QuantWeight,
                                               start_pos : Int32,
                                               n_tokens : Int32,
                                               n_head : Int32,
                                               n_head_kv : Int32,
                                               head_dim : Int32,
                                               rope_dim_count : Int32,
                                               heads_per_group : Int32,
                                               rope_freq_base : Float32,
                                               eps : Float32,
                                               scale : Float32,
                                               read_output : Bool = true,
                                               output_buf : ML::MetalBuffer? = nil,
                                               input_buf : ML::MetalBuffer? = nil,
                                               append_command_buffer : ML::Metal::CommandBuffer? = nil,
                                               adaptive_prefill_encoder : AdaptiveDecodeEncoder? = nil,
                                               kv_cache_f16 : Bool = false) : Array(Float32)?
          q_pipe = gemv_pipeline_for(q_qw)
          k_pipe = gemv_pipeline_for(k_qw)
          v_pipe = gemv_pipeline_for(v_qw)
          out_pipe = gemv_pipeline_for(out_qw)
          ffn_gate_pipe = gemv_pipeline_for(ffn_gate_qw)
          ffn_up_pipe = gemv_pipeline_for(ffn_up_qw)
          ffn_down_pipe = gemv_pipeline_for(ffn_down_qw)
          return nil if q_pipe.nil? || k_pipe.nil? || v_pipe.nil? || out_pipe.nil? ||
                        ffn_gate_pipe.nil? || ffn_up_pipe.nil? || ffn_down_pipe.nil?
          return nil unless n_tokens > 0
          return nil if append_command_buffer && read_output
          if adaptive_prefill_encoder
            raise ArgumentError.new("adaptive resident QBit KV cannot coexist with F16 KV buffers") if kv_cache_f16
            raise ArgumentError.new("adaptive resident QBit KV cannot coexist with F32 KV buffers") if k_cache_buf || v_cache_buf
          else
            return nil unless k_cache_buf && v_cache_buf
          end

          ML::Metal::Device.init!

          hidden_dim = q_qw.in_dim
          q_dim = n_head * head_dim
          kv_dim = n_head_kv * head_dim
          ffn_dim = ffn_gate_qw.out_dim
          hidden_bytes = (n_tokens * hidden_dim).to_i64 * sizeof(Float32)
          if ib = input_buf
            raise "full_attn_layer_chunk input buffer too small" if ib.size < hidden_bytes
          else
            raise "full_attn_layer_chunk input size mismatch" unless inp.size == n_tokens * hidden_dim
          end

          inp_buf = input_buf || Scratch.get(:full_chunk_inp, hidden_bytes)
          norm_w_buf = Scratch.get(:full_chunk_norm_w, attn_norm.size.to_i64 * sizeof(Float32))
          cur_buf = Scratch.get(:full_chunk_cur, hidden_bytes)
          qfull_buf = Scratch.get(:full_chunk_qfull, (n_tokens * q_qw.out_dim).to_i64 * sizeof(Float32))
          q_buf = Scratch.get(:full_chunk_q, (n_tokens * q_dim).to_i64 * sizeof(Float32))
          gate_buf = Scratch.get(:full_chunk_gate, (n_tokens * q_dim).to_i64 * sizeof(Float32))
          k_buf = Scratch.get(:full_chunk_k, (n_tokens * kv_dim).to_i64 * sizeof(Float32))
          v_buf = Scratch.get(:full_chunk_v, (n_tokens * kv_dim).to_i64 * sizeof(Float32))
          attn_buf = Scratch.get(:full_chunk_attn, (n_tokens * q_dim).to_i64 * sizeof(Float32))
          attn_out_buf = Scratch.get(:full_chunk_attn_out, (n_tokens * out_qw.out_dim).to_i64 * sizeof(Float32))
          qnorm_buf = Scratch.get(:full_chunk_qnorm, q_norm.size.to_i64 * sizeof(Float32))
          knorm_buf = Scratch.get(:full_chunk_knorm, k_norm.size.to_i64 * sizeof(Float32))
          post_norm_buf = Scratch.get(:full_chunk_postnorm_w, post_attention_norm.size.to_i64 * sizeof(Float32))
          residual_buf = Scratch.get(:full_chunk_residual, hidden_bytes)
          normed_buf = Scratch.get(:full_chunk_normed, hidden_bytes)
          ffn_gate_buf = Scratch.get(:full_chunk_ffn_gate, (n_tokens * ffn_dim).to_i64 * sizeof(Float32))
          ffn_up_buf = Scratch.get(:full_chunk_ffn_up, (n_tokens * ffn_dim).to_i64 * sizeof(Float32))
          ffn_comb_buf = Scratch.get(:full_chunk_ffn_comb, (n_tokens * ffn_dim).to_i64 * sizeof(Float32))
          ffn_out_buf = Scratch.get(:full_chunk_ffn_out, (n_tokens * ffn_down_qw.out_dim).to_i64 * sizeof(Float32))
          out_buf = Scratch.get(:full_chunk_out, hidden_bytes)

          inp_buf.write(inp) unless input_buf
          norm_w_buf.write(attn_norm)
          qnorm_buf.write(q_norm)
          knorm_buf.write(k_norm)
          post_norm_buf.write(post_attention_norm)

          q_w_buf, q_w_off = weight_slot(q_qw)
          k_w_buf, k_w_off = weight_slot(k_qw)
          v_w_buf, v_w_off = weight_slot(v_qw)
          out_w_buf, out_w_off = weight_slot(out_qw)
          ffn_gate_w_buf, ffn_gate_w_off = weight_slot(ffn_gate_qw)
          ffn_up_w_buf, ffn_up_w_off = weight_slot(ffn_up_qw)
          ffn_down_w_buf, ffn_down_w_off = weight_slot(ffn_down_qw)

          t0 = Time.instant if Profile.enabled?
          cmd = append_command_buffer || ML::Metal::CommandBuffer.new
          appended = !append_command_buffer.nil?

          norm_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_rmsnorm_rows(norm_enc, inp_buf, norm_w_buf, cur_buf, hidden_dim, n_tokens, eps)
          norm_enc.end_encoding

          proj_enc = ML::Metal::ComputeEncoder.new(cmd)
          full_projection_qws = [q_qw, k_qw, v_qw]
          if prefill_shared_projection_h16_candidate?(full_projection_qws, n_tokens, false)
            encode_shared_projection_h16(proj_enc, full_projection_qws, cur_buf,
              [qfull_buf, k_buf, v_buf], n_tokens, "full_chunk_projection_x16")
          else
            encode_matmul(proj_enc, q_pipe.not_nil!, q_qw, cur_buf, qfull_buf, q_w_buf, q_w_off, q_qw.in_dim, q_qw.out_dim, n_tokens)
            encode_matmul(proj_enc, k_pipe.not_nil!, k_qw, cur_buf, k_buf, k_w_buf, k_w_off, k_qw.in_dim, k_qw.out_dim, n_tokens)
            encode_matmul(proj_enc, v_pipe.not_nil!, v_qw, cur_buf, v_buf, v_w_buf, v_w_off, v_qw.in_dim, v_qw.out_dim, n_tokens)
          end
          proj_enc.end_encoding

          split_enc = ML::Metal::ComputeEncoder.new(cmd)
          split_enc.set_pipeline(split_qgate_rows_pipeline)
          split_enc.set_buffer(qfull_buf, 0)
          split_enc.set_buffer(q_buf, 1, ML::Metal::BufferAccess::Write)
          split_enc.set_buffer(gate_buf, 2, ML::Metal::BufferAccess::Write)
          split_enc.set_value(n_head.to_u32, 3)
          split_enc.set_value(head_dim.to_u32, 4)
          split_enc.set_value(n_tokens.to_u32, 5)
          split_enc.dispatch_1d(n_tokens * q_dim, 256)
          split_enc.end_encoding

          qnorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          qnorm_enc.set_pipeline(rmsnorm_heads_rows_pipeline)
          qnorm_enc.set_buffer(q_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          qnorm_enc.set_buffer(qnorm_buf, 1)
          qnorm_enc.set_value(head_dim.to_u32, 2)
          qnorm_enc.set_value(eps, 3)
          qnorm_enc.set_value(n_head.to_u32, 4)
          qnorm_enc.set_value(n_tokens.to_u32, 5)
          qnorm_enc.dispatch_threadgroups({n_head, n_tokens, 1}, {32, 1, 1})
          qnorm_enc.end_encoding

          knorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          knorm_enc.set_pipeline(rmsnorm_heads_rows_pipeline)
          knorm_enc.set_buffer(k_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          knorm_enc.set_buffer(knorm_buf, 1)
          knorm_enc.set_value(head_dim.to_u32, 2)
          knorm_enc.set_value(eps, 3)
          knorm_enc.set_value(n_head_kv.to_u32, 4)
          knorm_enc.set_value(n_tokens.to_u32, 5)
          knorm_enc.dispatch_threadgroups({n_head_kv, n_tokens, 1}, {32, 1, 1})
          knorm_enc.end_encoding

          qrope_enc = ML::Metal::ComputeEncoder.new(cmd)
          qrope_enc.set_pipeline(rope_partial_rows_pipeline)
          qrope_enc.set_buffer(q_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          qrope_enc.set_value(head_dim.to_u32, 1)
          qrope_enc.set_value(rope_dim_count.to_u32, 2)
          qrope_enc.set_value(start_pos.to_u32, 3)
          qrope_enc.set_value(rope_freq_base, 4)
          qrope_enc.set_value(n_head.to_u32, 5)
          qrope_enc.set_value(n_tokens.to_u32, 6)
          qrope_enc.dispatch_threadgroups({n_head, n_tokens, 1}, {32, 1, 1})
          qrope_enc.end_encoding

          krope_enc = ML::Metal::ComputeEncoder.new(cmd)
          krope_enc.set_pipeline(rope_partial_rows_pipeline)
          krope_enc.set_buffer(k_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          krope_enc.set_value(head_dim.to_u32, 1)
          krope_enc.set_value(rope_dim_count.to_u32, 2)
          krope_enc.set_value(start_pos.to_u32, 3)
          krope_enc.set_value(rope_freq_base, 4)
          krope_enc.set_value(n_head_kv.to_u32, 5)
          krope_enc.set_value(n_tokens.to_u32, 6)
          krope_enc.dispatch_threadgroups({n_head_kv, n_tokens, 1}, {32, 1, 1})
          krope_enc.end_encoding

          if adaptive_encoder = adaptive_prefill_encoder
            adaptive_encoder.call(cmd, q_buf, gate_buf, k_buf, v_buf, attn_buf)
          else
            kvwrite_enc = ML::Metal::ComputeEncoder.new(cmd)
            kvwrite_enc.set_pipeline(kv_cache_f16 ? kv_write_rows_h16_pipeline : kv_write_rows_pipeline)
            kvwrite_enc.set_buffer(k_buf, 0)
            kvwrite_enc.set_buffer(v_buf, 1)
            kvwrite_enc.set_buffer(k_cache_buf.not_nil!, 2, ML::Metal::BufferAccess::ReadWrite)
            kvwrite_enc.set_buffer(v_cache_buf.not_nil!, 3, ML::Metal::BufferAccess::ReadWrite)
            kvwrite_enc.set_value(start_pos.to_u32, 4)
            kvwrite_enc.set_value(kv_dim.to_u32, 5)
            kvwrite_enc.set_value(n_tokens.to_u32, 6)
            kvwrite_enc.dispatch_1d(n_tokens * kv_dim, 256)
            kvwrite_enc.end_encoding

            attn_enc = ML::Metal::ComputeEncoder.new(cmd)
            use_flash_d256 = prefill_attn_flash_d256_enabled?(
              start_pos, n_tokens, n_head, n_head_kv, head_dim, kv_cache_f16,
            )
            Profile.bump_route_marker("prefill_attn_flash_d256") if use_flash_d256
            use_attn_sg4 = prefill_attn_rows_sg4_enabled? && n_tokens >= 4
            use_direct_gate = !prefill_attn_rows_sg4_pregate_enabled? && prefill_attn_rows_sg4_direct_gate_enabled?(n_tokens)
            attn_sg4_pipeline = if kv_cache_f16
                                  use_direct_gate ? attn_rows_sg4_h16_pipeline : attn_rows_sg4_pregate_h16_pipeline
                                else
                                  use_direct_gate ? attn_rows_sg4_pipeline : attn_rows_sg4_pregate_pipeline
                                end
            attn_rows_selected = kv_cache_f16 ? attn_rows_h16_pipeline : attn_rows_pipeline
            attn_enc.set_pipeline(use_flash_d256 ? attn_flash_d256_pipeline : (use_attn_sg4 ? attn_sg4_pipeline : attn_rows_selected))
            attn_enc.set_buffer(q_buf, 0)
            attn_enc.set_buffer(gate_buf, 1)
            attn_enc.set_buffer(k_cache_buf.not_nil!, 2)
            attn_enc.set_buffer(v_cache_buf.not_nil!, 3)
            attn_enc.set_buffer(attn_buf, 4, ML::Metal::BufferAccess::Write)
            attn_enc.set_value(start_pos.to_u32, 5)
            attn_enc.set_value(n_tokens.to_u32, 6)
            attn_enc.set_value(n_head.to_u32, 7)
            attn_enc.set_value(n_head_kv.to_u32, 8)
            attn_enc.set_value(head_dim.to_u32, 9)
            attn_enc.set_value(heads_per_group.to_u32, 10)
            attn_enc.set_value(scale, 11)
            if use_flash_d256
              attn_enc.set_threadgroup_memory(16 * 1024, 0)
              attn_enc.dispatch_threadgroups({(n_tokens + 7) // 8, n_head, 1}, {32, 4, 1})
            elsif use_attn_sg4
              attn_enc.dispatch_threadgroups({n_head, (n_tokens + 3) // 4, 1}, {128, 1, 1})
            else
              attn_enc.dispatch_threadgroups({n_head, n_tokens, 1}, {32, 1, 1})
            end
            attn_enc.end_encoding
          end

          outproj_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_matmul(outproj_enc, out_pipe.not_nil!, out_qw, attn_buf, attn_out_buf, out_w_buf, out_w_off, out_qw.in_dim, out_qw.out_dim, n_tokens)
          outproj_enc.end_encoding

          addnorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_add_rmsnorm_rows(addnorm_enc, inp_buf, attn_out_buf, post_norm_buf, residual_buf, normed_buf, hidden_dim, n_tokens, eps)
          addnorm_enc.end_encoding

          ffn_proj_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_matmul(ffn_proj_enc, ffn_gate_pipe.not_nil!, ffn_gate_qw, normed_buf, ffn_gate_buf, ffn_gate_w_buf, ffn_gate_w_off, ffn_gate_qw.in_dim, ffn_gate_qw.out_dim, n_tokens)
          encode_matmul(ffn_proj_enc, ffn_up_pipe.not_nil!, ffn_up_qw, normed_buf, ffn_up_buf, ffn_up_w_buf, ffn_up_w_off, ffn_up_qw.in_dim, ffn_up_qw.out_dim, n_tokens)
          ffn_proj_enc.end_encoding

          swiglu_enc = ML::Metal::ComputeEncoder.new(cmd)
          ffn_act_buf = swiglu_inplace_enabled? ? ffn_up_buf : ffn_comb_buf
          swiglu_enc.set_pipeline(ffn_swiglu_pipeline)
          swiglu_enc.set_buffer(ffn_gate_buf, 0)
          swiglu_enc.set_buffer(ffn_up_buf, 1)
          swiglu_enc.set_buffer(ffn_act_buf, 2, ML::Metal::BufferAccess::Write)
          swiglu_enc.set_value((n_tokens * ffn_dim).to_u32, 3)
          swiglu_enc.dispatch_1d(n_tokens * ffn_dim, 256)
          swiglu_enc.end_encoding

          ffn_down_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_matmul(ffn_down_enc, ffn_down_pipe.not_nil!, ffn_down_qw, ffn_act_buf, ffn_out_buf, ffn_down_w_buf, ffn_down_w_off, ffn_down_qw.in_dim, ffn_down_qw.out_dim, n_tokens)
          ffn_down_enc.end_encoding

          add_enc = ML::Metal::ComputeEncoder.new(cmd)
          add_enc.set_pipeline(add_vec_pipeline)
          add_enc.set_buffer(residual_buf, 0)
          add_enc.set_buffer(ffn_out_buf, 1)
          add_enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
          add_enc.set_value((n_tokens * hidden_dim).to_u32, 3)
          add_enc.dispatch_1d(n_tokens * hidden_dim, 256)
          add_enc.end_encoding

          if ob = output_buf
            blit = ML::Metal::BlitEncoder.new(cmd)
            blit.copy_buffer(out_buf, 0, ob, 0, ((n_tokens * hidden_dim).to_i64 * sizeof(Float32)).to_i32)
            blit.end_encoding
          end

          return [] of Float32 if appended

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_output ? read_shared_f32(out_buf, n_tokens * hidden_dim) : [] of Float32
          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_attn(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              read_output ? (t_read - t_wait.not_nil!).total_nanoseconds.to_i64 : 0_i64,
            )
          end
          result
        end

        # Final full-attention body-only prefill specialization.
        #
        # When no logits or downstream hidden rows are requested, the last
        # decoder layer only has to populate its K/V cache for future decode.
        # Q/attention/FFN output is semantically dead and can be skipped.
        def self.full_attn_layer_chunk_kv_cache_only(inp : Array(Float32),
                                                     k_qw : QuantWeight,
                                                     v_qw : QuantWeight,
                                                     attn_norm : Array(Float32),
                                                     k_norm : Array(Float32),
                                                     k_cache_buf : ML::MetalBuffer,
                                                     v_cache_buf : ML::MetalBuffer,
                                                     start_pos : Int32,
                                                     n_tokens : Int32,
                                                     n_head_kv : Int32,
                                                     head_dim : Int32,
                                                     rope_dim_count : Int32,
                                                     rope_freq_base : Float32,
                                                     eps : Float32,
                                                     input_buf : ML::MetalBuffer? = nil,
                                                     append_command_buffer : ML::Metal::CommandBuffer? = nil,
                                                     kv_cache_f16 : Bool = false) : Bool
          k_pipe = gemv_pipeline_for(k_qw)
          v_pipe = gemv_pipeline_for(v_qw)
          return false if k_pipe.nil? || v_pipe.nil?
          return false unless n_tokens > 0

          ML::Metal::Device.init!

          hidden_dim = k_qw.in_dim
          kv_dim = n_head_kv * head_dim
          hidden_elems = n_tokens * hidden_dim
          hidden_bytes = hidden_elems.to_i64 * sizeof(Float32)
          if ib = input_buf
            raise "full-attn kv-cache input buffer too small" if ib.size < hidden_bytes
          else
            raise "full-attn kv-cache input size mismatch" unless inp.size == hidden_elems
          end

          inp_buf = input_buf || Scratch.get(:full_kv_only_inp, hidden_bytes)
          norm_w_buf = Scratch.get(:full_kv_only_norm_w, attn_norm.size.to_i64 * sizeof(Float32))
          cur_buf = Scratch.get(:full_kv_only_cur, hidden_bytes)
          cur_h16_buf = Scratch.get(:full_kv_only_cur_h16, hidden_elems.to_i64 * 2_i64)
          k_buf = Scratch.get(:full_kv_only_k, (n_tokens * kv_dim).to_i64 * sizeof(Float32))
          v_buf = Scratch.get(:full_kv_only_v, (n_tokens * kv_dim).to_i64 * sizeof(Float32))
          knorm_buf = Scratch.get(:full_kv_only_knorm, k_norm.size.to_i64 * sizeof(Float32))

          unless input_buf
            inp_buf.write(inp)
            Profile.bump_group_transfer("full_kv_only.boundary", hidden_bytes, 0_i64) if Profile.enabled?
          end
          ConstCache.write_once("full_kv_only_norm_w_#{attn_norm.object_id}", norm_w_buf, attn_norm)
          ConstCache.write_once("full_kv_only_knorm_#{k_norm.object_id}", knorm_buf, k_norm)

          k_w_buf, k_w_off = weight_slot(k_qw)
          v_w_buf, v_w_off = weight_slot(v_qw)

          t0 = Time.instant if Profile.enabled?
          cmd = append_command_buffer || ML::Metal::CommandBuffer.new
          appended = !append_command_buffer.nil?

          norm_enc = ML::Metal::ComputeEncoder.new(cmd)
          norm_h16_proj = prefill_rmsnorm_h16_proj_enabled? && n_tokens > GEMM_BATCH_THRESHOLD
          if norm_h16_proj
            encode_rmsnorm_rows_f32_h16(norm_enc, inp_buf, norm_w_buf, cur_buf, cur_h16_buf, hidden_dim, n_tokens, eps)
          else
            encode_rmsnorm_rows(norm_enc, inp_buf, norm_w_buf, cur_buf, hidden_dim, n_tokens, eps)
          end
          norm_enc.end_encoding

          Profile.trace("prefill.full.kv_cache") do
            proj_enc = ML::Metal::ComputeEncoder.new(cmd)
            if norm_h16_proj && h16_batch_gemm_candidate?(k_qw, n_tokens)
              raise "unsupported h16 final k-cache route" unless encode_matmul_from_h16(proj_enc, k_qw, cur_h16_buf, k_buf, k_w_buf, k_w_off, k_qw.in_dim, k_qw.out_dim, n_tokens)
            else
              encode_matmul(proj_enc, k_pipe.not_nil!, k_qw, cur_buf, k_buf, k_w_buf, k_w_off, k_qw.in_dim, k_qw.out_dim, n_tokens)
            end
            if norm_h16_proj && h16_batch_gemm_candidate?(v_qw, n_tokens)
              raise "unsupported h16 final v-cache route" unless encode_matmul_from_h16(proj_enc, v_qw, cur_h16_buf, v_buf, v_w_buf, v_w_off, v_qw.in_dim, v_qw.out_dim, n_tokens)
            else
              encode_matmul(proj_enc, v_pipe.not_nil!, v_qw, cur_buf, v_buf, v_w_buf, v_w_off, v_qw.in_dim, v_qw.out_dim, n_tokens)
            end
            proj_enc.end_encoding
          end

          knorm_enc = ML::Metal::ComputeEncoder.new(cmd)
          knorm_enc.set_pipeline(rmsnorm_heads_rows_pipeline)
          knorm_enc.set_buffer(k_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          knorm_enc.set_buffer(knorm_buf, 1)
          knorm_enc.set_value(head_dim.to_u32, 2)
          knorm_enc.set_value(eps, 3)
          knorm_enc.set_value(n_head_kv.to_u32, 4)
          knorm_enc.set_value(n_tokens.to_u32, 5)
          knorm_enc.dispatch_threadgroups({n_head_kv, n_tokens, 1}, {32, 1, 1})
          knorm_enc.end_encoding

          krope_enc = ML::Metal::ComputeEncoder.new(cmd)
          krope_enc.set_pipeline(rope_partial_rows_pipeline)
          krope_enc.set_buffer(k_buf, 0, ML::Metal::BufferAccess::ReadWrite)
          krope_enc.set_value(head_dim.to_u32, 1)
          krope_enc.set_value(rope_dim_count.to_u32, 2)
          krope_enc.set_value(start_pos.to_u32, 3)
          krope_enc.set_value(rope_freq_base, 4)
          krope_enc.set_value(n_head_kv.to_u32, 5)
          krope_enc.set_value(n_tokens.to_u32, 6)
          krope_enc.dispatch_threadgroups({n_head_kv, n_tokens, 1}, {32, 1, 1})
          krope_enc.end_encoding

          kvwrite_enc = ML::Metal::ComputeEncoder.new(cmd)
          kvwrite_enc.set_pipeline(kv_cache_f16 ? kv_write_rows_h16_pipeline : kv_write_rows_pipeline)
          kvwrite_enc.set_buffer(k_buf, 0)
          kvwrite_enc.set_buffer(v_buf, 1)
          kvwrite_enc.set_buffer(k_cache_buf, 2, ML::Metal::BufferAccess::ReadWrite)
          kvwrite_enc.set_buffer(v_cache_buf, 3, ML::Metal::BufferAccess::ReadWrite)
          kvwrite_enc.set_value(start_pos.to_u32, 4)
          kvwrite_enc.set_value(kv_dim.to_u32, 5)
          kvwrite_enc.set_value(n_tokens.to_u32, 6)
          kvwrite_enc.dispatch_1d(n_tokens * kv_dim, 256)
          kvwrite_enc.end_encoding

          return true if appended

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          if Profile.enabled?
            Profile.bump_attn(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              0_i64,
            )
            Profile.bump_group("full_kv_only",
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              0_i64)
          end
          true
        end
      {% end %}
    end
  end
end
