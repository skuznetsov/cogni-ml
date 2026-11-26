# Vocabulary projection and token selection for the Qwen35 Metal engine.

module ML
  module GGUF
    module Qwen35Metal
      HEAD_TOP1_ROWS_PER_TG = 12

      {% if flag?(:cpu_only) %}
        def self.project_top1_no_norm(out_qw : QuantWeight,
                                      x : Array(Float32)) : Array(Float32)?
          nil
        end

        def self.project_top1_no_norm_graph(out_qw : QuantWeight,
                                            x : Array(Float32)) : Array(Float32)?
          nil
        end

        def self.project_top2_no_norm(out_qw : QuantWeight,
                                      x : Array(Float32)) : Array(Float32)?
          nil
        end

        def self.rmsnorm_project_top1_allowed_ids(x : Array(Float32),
                                                  norm_weight : Array(Float32),
                                                  out_qw : QuantWeight,
                                                  eps : Float32,
                                                  allowed_ids : Array(Int32)) : Array(Float32)?
          nil
        end

        def self.rmsnorm_project_top1_allowed_ids_supported?(out_qw : QuantWeight) : Bool
          false
        end
      {% else %}
        @@mv8_top1_tiles_pipeline : ML::Metal::ComputePipeline?
        @@mv8_top2_tiles_pipeline : ML::Metal::ComputePipeline?
        @@mv6_top1_tiles_pipeline : ML::Metal::ComputePipeline?
        @@mv6_top1_allowed_tiles_pipeline : ML::Metal::ComputePipeline?
        @@mv6_top2_tiles_pipeline : ML::Metal::ComputePipeline?
        @@mv6_top1_tiles_batch_pipeline : ML::Metal::ComputePipeline?
        @@top1_reduce_tiles_pipeline : ML::Metal::ComputePipeline?
        @@top2_reduce_tiles_pipeline : ML::Metal::ComputePipeline?
        @@top1_reduce_tiles_batch_pipeline : ML::Metal::ComputePipeline?
        @@top1_reduce_f16_rows_pipeline : ML::Metal::ComputePipeline?
        @@top2_reduce_f16_rows_pipeline : ML::Metal::ComputePipeline?
        @@store_top1_token_id_pipeline : ML::Metal::ComputePipeline?

        private def self.mv6_top1_tiles_pipeline : ML::Metal::ComputePipeline
          @@mv6_top1_tiles_pipeline ||= ML::Metal::PipelineCache.get("simd_mv_q6k_top1_tiles_f32") {
            ML::Metal::ComputePipeline.new("simd_mv_q6k_top1_tiles_f32", GEMM_Q56K_SOURCE)
          }
        end

        private def self.mv6_top1_tiles_rows_pipeline(rows_per_tg : Int32) : ML::Metal::ComputePipeline
          raise ArgumentError.new("Q6 top1 rows_per_tg must be positive") unless rows_per_tg > 0
          raise ArgumentError.new("Q6 top1 rows_per_tg must be divisible by #{MV_Q6_NSG}") unless rows_per_tg % MV_Q6_NSG == 0

          key = "simd_mv_q6k_top1_tiles_f32_rows#{rows_per_tg}"
          ML::Metal::PipelineCache.get(key) do
            source = GEMM_Q56K_SOURCE
              .gsub("constant short MV6_TOP_ROWS_PER_TG = 12;", "constant short MV6_TOP_ROWS_PER_TG = #{rows_per_tg};")
            ML::Metal::ComputePipeline.new("simd_mv_q6k_top1_tiles_f32", source)
          end
        end

        private def self.mv6_top1_allowed_tiles_pipeline : ML::Metal::ComputePipeline
          @@mv6_top1_allowed_tiles_pipeline ||= ML::Metal::PipelineCache.get("simd_mv_q6k_top1_allowed_tiles_f32") {
            ML::Metal::ComputePipeline.new("simd_mv_q6k_top1_allowed_tiles_f32", GEMM_Q56K_SOURCE)
          }
        end

        private def self.mv6_top2_tiles_pipeline : ML::Metal::ComputePipeline
          @@mv6_top2_tiles_pipeline ||= ML::Metal::PipelineCache.get("simd_mv_q6k_top2_tiles_f32") {
            ML::Metal::ComputePipeline.new("simd_mv_q6k_top2_tiles_f32", GEMM_Q56K_SOURCE)
          }
        end

        private def self.mv6_top1_tiles_batch_pipeline : ML::Metal::ComputePipeline
          @@mv6_top1_tiles_batch_pipeline ||= ML::Metal::PipelineCache.get("simd_mv_q6k_top1_tiles_batch_f32") {
            ML::Metal::ComputePipeline.new("simd_mv_q6k_top1_tiles_batch_f32", GEMM_Q56K_SOURCE)
          }
        end

        private def self.mv8_top1_tiles_pipeline : ML::Metal::ComputePipeline
          @@mv8_top1_tiles_pipeline ||= ML::Metal::PipelineCache.get("simd_mv_q8_0_top1_tiles_f32") {
            ML::Metal::ComputePipeline.new("simd_mv_q8_0_top1_tiles_f32", GEMM_Q56K_SOURCE)
          }
        end

        private def self.mv8_top2_tiles_pipeline : ML::Metal::ComputePipeline
          @@mv8_top2_tiles_pipeline ||= ML::Metal::PipelineCache.get("simd_mv_q8_0_top2_tiles_f32") {
            ML::Metal::ComputePipeline.new("simd_mv_q8_0_top2_tiles_f32", GEMM_Q56K_SOURCE)
          }
        end

        private def self.top1_reduce_tiles_pipeline : ML::Metal::ComputePipeline
          @@top1_reduce_tiles_pipeline ||= ML::Metal::PipelineCache.get("qwen35_top1_reduce_tiles") {
            ML::Metal::ComputePipeline.new("qwen35_top1_reduce_tiles", GEMM_Q56K_SOURCE)
          }
        end

        private def self.top2_reduce_tiles_pipeline : ML::Metal::ComputePipeline
          @@top2_reduce_tiles_pipeline ||= ML::Metal::PipelineCache.get("qwen35_top2_reduce_tiles") {
            ML::Metal::ComputePipeline.new("qwen35_top2_reduce_tiles", GEMM_Q56K_SOURCE)
          }
        end

        private def self.top1_reduce_tiles_batch_pipeline : ML::Metal::ComputePipeline
          @@top1_reduce_tiles_batch_pipeline ||= ML::Metal::PipelineCache.get("qwen35_top1_reduce_tiles_batch") {
            ML::Metal::ComputePipeline.new("qwen35_top1_reduce_tiles_batch", GEMM_Q56K_SOURCE)
          }
        end

        private def self.top1_reduce_f16_rows_pipeline : ML::Metal::ComputePipeline
          @@top1_reduce_f16_rows_pipeline ||= ML::Metal::PipelineCache.get("qwen35_top1_reduce_f16_rows") {
            ML::Metal::ComputePipeline.new("qwen35_top1_reduce_f16_rows", GEMM_Q56K_SOURCE)
          }
        end

        private def self.top2_reduce_f16_rows_pipeline : ML::Metal::ComputePipeline
          @@top2_reduce_f16_rows_pipeline ||= ML::Metal::PipelineCache.get("qwen35_top2_reduce_f16_rows") {
            ML::Metal::ComputePipeline.new("qwen35_top2_reduce_f16_rows", GEMM_Q56K_SOURCE)
          }
        end

        private def self.store_top1_token_id_pipeline : ML::Metal::ComputePipeline
          @@store_top1_token_id_pipeline ||= ML::Metal::PipelineCache.get("qwen35_store_top1_token_id") {
            ML::Metal::ComputePipeline.new("qwen35_store_top1_token_id", GEMM_Q56K_SOURCE)
          }
        end

        private def self.read_shared_top1(id_buf : ML::MetalBuffer, value_buf : ML::MetalBuffer) : Array(Float32)
          id = id_buf.contents.as(Pointer(UInt32)).value
          value = value_buf.contents.as(Pointer(Float32)).value
          [id.to_f32, value]
        end

        def self.read_head_top1_buffers(id_buf : ML::MetalBuffer, value_buf : ML::MetalBuffer) : Array(Float32)
          read_shared_top1(id_buf, value_buf)
        end

        def self.read_head_top1_rows_buffers(id_buf : ML::MetalBuffer,
                                             value_buf : ML::MetalBuffer,
                                             rows : Int32) : Array({Int32, Float32})
          raise ArgumentError.new("top-1 row count must be positive") unless rows > 0
          required_ids = rows.to_i64 * sizeof(UInt32)
          required_values = rows.to_i64 * sizeof(Float32)
          raise ArgumentError.new("top-1 row id buffer is too small") if id_buf.size < required_ids
          raise ArgumentError.new("top-1 row value buffer is too small") if value_buf.size < required_values
          read_shared_top1_rows(id_buf, value_buf, rows)
        end

        def self.read_head_top2_buffers(id_buf : ML::MetalBuffer,
                                        value_buf : ML::MetalBuffer,
                                        second_id_buf : ML::MetalBuffer,
                                        second_value_buf : ML::MetalBuffer) : Array(Float32)
          read_shared_top2(id_buf, value_buf, second_id_buf, second_value_buf)
        end

        def self.head_top1_tile_count(out_qw : QuantWeight) : Int32
          (out_qw.out_dim + HEAD_TOP1_ROWS_PER_TG - 1) // HEAD_TOP1_ROWS_PER_TG
        end

        private def self.read_shared_top2(id_buf : ML::MetalBuffer,
                                          value_buf : ML::MetalBuffer,
                                          second_id_buf : ML::MetalBuffer,
                                          second_value_buf : ML::MetalBuffer) : Array(Float32)
          id = id_buf.contents.as(Pointer(UInt32)).value
          value = value_buf.contents.as(Pointer(Float32)).value
          second_id = second_id_buf.contents.as(Pointer(UInt32)).value
          second_value = second_value_buf.contents.as(Pointer(Float32)).value
          [id.to_f32, value, second_id.to_f32, second_value]
        end

        private def self.read_shared_top1_rows(id_buf : ML::MetalBuffer, value_buf : ML::MetalBuffer, rows : Int32) : Array({Int32, Float32})
          ids = id_buf.contents.as(Pointer(UInt32))
          values = value_buf.contents.as(Pointer(Float32))
          Array({Int32, Float32}).new(rows) do |i|
            {ids[i].to_i32, values[i]}
          end
        end

        private def self.read_shared_top2_rows(top_id_buf : ML::MetalBuffer,
                                               top_value_buf : ML::MetalBuffer,
                                               second_id_buf : ML::MetalBuffer,
                                               second_value_buf : ML::MetalBuffer,
                                               rows : Int32) : Array({Int32, Float32, Int32, Float32})
          top_ids = top_id_buf.contents.as(Pointer(UInt32))
          top_values = top_value_buf.contents.as(Pointer(Float32))
          second_ids = second_id_buf.contents.as(Pointer(UInt32))
          second_values = second_value_buf.contents.as(Pointer(Float32))
          Array({Int32, Float32, Int32, Float32}).new(rows) do |i|
            {top_ids[i].to_i32, top_values[i], second_ids[i].to_i32, second_values[i]}
          end
        end

        private def self.head_top1_fused_enabled? : Bool
          ENV["QWEN35_HEAD_TOP1_FUSED"]? != "0"
        end

        private def self.can_use_head_top1_fused?(output_qw : QuantWeight) : Bool
          head_top1_fused_enabled? &&
            ((output_qw.type.q6_k? && output_qw.in_dim % QK_K == 0) ||
              (output_qw.type.q8_0? && output_qw.in_dim % Q8_0_QK == 0))
        end

        private def self.can_use_head_top1_allowed?(output_qw : QuantWeight) : Bool
          head_top1_fused_enabled? && output_qw.type.q6_k? && output_qw.in_dim % QK_K == 0
        end

        private def self.profile_bump_head_top1_shape(label : String,
                                                      output_qw : QuantWeight,
                                                      rows : Int32? = nil) : Nil
          return unless Profile.enabled?

          weight_bytes = if rows
                           row_bytes = output_qw.raw.size.to_i64 // output_qw.out_dim
                           row_bytes * rows.not_nil!
                         else
                           output_qw.raw.size.to_i64
                         end
          Profile.bump_matmul_shape("#{label} #{output_qw.type.name} #{output_qw.in_dim}x#{output_qw.out_dim} b1", weight_bytes)
        end

        # Final output path:
        #   output RMSNorm -> lm_head projection
        # in one command buffer, with only logits read back.
        def self.rmsnorm_project(x : Array(Float32),
                                 norm_weight : Array(Float32),
                                 out_qw : QuantWeight,
                                 eps : Float32) : Array(Float32)?
          out_pipe = gemv_pipeline_for(out_qw)
          return nil if out_pipe.nil?

          ML::Metal::Device.init!

          hidden_dim = x.size
          x_buf = Scratch.get(:head_x, hidden_dim.to_i64 * sizeof(Float32))
          norm_w_buf = Scratch.get(:head_norm_w, norm_weight.size.to_i64 * sizeof(Float32))
          normed_buf = Scratch.get(:head_normed, hidden_dim.to_i64 * sizeof(Float32))
          out_buf = Scratch.get(:head_out, out_qw.out_dim.to_i64 * sizeof(Float32))
          x_buf.write(x)
          norm_w_buf.write(norm_weight)

          out_w_buf, out_w_off = if slot = mmap_slot_for(out_qw.raw)
                                   slot
                                 else
                                   {out_qw.fallback_metal_buffer, 0_i64}
                                 end

          t0 = Time.instant if Profile.enabled?
          cmd = ML::Metal::CommandBuffer.new

          norm_enc = ML::Metal::ComputeEncoder.new(cmd)
          norm_enc.set_pipeline(rmsnorm_vec_pipeline)
          norm_enc.set_buffer(x_buf, 0)
          norm_enc.set_buffer(norm_w_buf, 1)
          norm_enc.set_buffer(normed_buf, 2, ML::Metal::BufferAccess::Write)
          norm_enc.set_value(hidden_dim.to_u32, 3)
          norm_enc.set_value(eps, 4)
          norm_enc.dispatch_threadgroups({1, 1, 1}, {256, 1, 1})
          norm_enc.end_encoding

          out_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_gemv(out_enc, out_pipe.not_nil!, normed_buf, out_buf, out_w_buf, out_w_off, out_qw.in_dim, out_qw.out_dim)
          out_enc.end_encoding

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_f32(out_buf, out_qw.out_dim)
          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_gemv(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          result
        end

        def self.rmsnorm_project_top1(x : Array(Float32),
                                      norm_weight : Array(Float32),
                                      out_qw : QuantWeight,
                                      eps : Float32) : Array(Float32)?
          return nil unless can_use_head_top1_fused?(out_qw)
          return nil unless x.size == out_qw.in_dim

          ML::Metal::Device.init!

          x_buf = Scratch.get(:head_top1_x, out_qw.in_dim.to_i64 * sizeof(Float32))
          x_buf.write(x)
          rmsnorm_project_top1_buffer_impl(
            x_buf, 0_i64, norm_weight, out_qw, eps, "head_top1",
          )
        end

        def self.rmsnorm_project_top1_supported?(out_qw : QuantWeight) : Bool
          can_use_head_top1_fused?(out_qw)
        end

        # The allowed-token kernel is intentionally Q6_K-only. Keep this
        # preflight predicate aligned with the encoder's exact policy so CPU
        # callers can reject before mutating a command buffer.
        def self.rmsnorm_project_top1_allowed_ids_supported?(out_qw : QuantWeight) : Bool
          can_use_head_top1_allowed?(out_qw)
        end

        # Project one hidden row directly from a resident multi-row buffer.
        # `element_offset` is expressed in Float32 elements, not bytes.
        def self.rmsnorm_project_top1_buffer(x_buf : ML::MetalBuffer,
                                             element_offset : Int64,
                                             norm_weight : Array(Float32),
                                             out_qw : QuantWeight,
                                             eps : Float32) : Array(Float32)?
          rmsnorm_project_top1_buffer_impl(
            x_buf, element_offset, norm_weight, out_qw, eps, "head_top1_resident",
          )
        end

        # Encode one resident hidden row through output RMSNorm and the fused
        # top-1 head into a caller-owned command buffer. The caller owns commit,
        # completion, result-buffer lifetime, and result visibility.
        def self.encode_rmsnorm_project_top1_buffer(command : ML::Metal::CommandBuffer,
                                                    x_buf : ML::MetalBuffer,
                                                    element_offset : Int64,
                                                    norm_weight : Array(Float32),
                                                    out_qw : QuantWeight,
                                                    eps : Float32,
                                                    top1_id_buf : ML::MetalBuffer,
                                                    top1_value_buf : ML::MetalBuffer,
                                                    profile_label : String = "head_top1_resident_append") : Bool
          return false unless can_use_head_top1_fused?(out_qw)
          return false if command.committed?
          return false if element_offset < 0
          hidden_dim = out_qw.in_dim
          return false unless norm_weight.size == hidden_dim
          hidden_dim_i64 = hidden_dim.to_i64
          return false if element_offset > Int64::MAX - hidden_dim_i64
          required_elements = element_offset + hidden_dim_i64
          return false if required_elements > Int64::MAX // sizeof(Float32)
          return false if x_buf.size < required_elements * sizeof(Float32)
          return false if top1_id_buf.size < sizeof(UInt32)
          return false if top1_value_buf.size < sizeof(Float32)

          ML::Metal::Device.init!

          tile_count = head_top1_tile_count(out_qw)
          norm_w_buf = Scratch.get(:head_top1_norm_w, norm_weight.size.to_i64 * sizeof(Float32))
          normed_buf = Scratch.get(:head_top1_normed, hidden_dim.to_i64 * sizeof(Float32))
          tile_values_buf = Scratch.get(:head_top1_tile_values, tile_count.to_i64 * sizeof(Float32))
          tile_ids_buf = Scratch.get(:head_top1_tile_ids, tile_count.to_i64 * sizeof(UInt32))
          norm_w_buf.write(norm_weight)

          norm_enc = ML::Metal::ComputeEncoder.new(command)
          norm_enc.set_pipeline(rmsnorm_vec_pipeline)
          norm_enc.set_buffer(x_buf, 0, ML::Metal::BufferAccess::Read,
            offset: element_offset * sizeof(Float32))
          norm_enc.set_buffer(norm_w_buf, 1)
          norm_enc.set_buffer(normed_buf, 2, ML::Metal::BufferAccess::Write)
          norm_enc.set_value(hidden_dim.to_u32, 3)
          norm_enc.set_value(eps, 4)
          norm_enc.dispatch_threadgroups({1, 1, 1}, {256, 1, 1})
          norm_enc.end_encoding

          head_top1_enc = ML::Metal::ComputeEncoder.new(command)
          encoded = encode_head_top1_no_norm_to_buffers(
            head_top1_enc, out_qw, normed_buf,
            tile_values_buf, tile_ids_buf, top1_id_buf, top1_value_buf,
          )
          head_top1_enc.end_encoding
          Profile.bump_route_marker(profile_label) if encoded
          encoded
        end

        # Encode every resident hidden row through output RMSNorm and the
        # batched Q6_K top-1 head into a caller-owned command buffer. The
        # caller must keep the input, output, command, and scratch arena alive
        # through completion. `true` means encoded, not completed; the caller
        # owns commit, wait, result visibility, and arena release.
        def self.encode_rmsnorm_project_top1_rows_buffer(command : ML::Metal::CommandBuffer,
                                                         scratch_arena : Scratch::Arena,
                                                         x_buf : ML::MetalBuffer,
                                                         rows : Int32,
                                                         norm_weight : Array(Float32),
                                                         out_qw : QuantWeight,
                                                         eps : Float32,
                                                         top1_id_buf : ML::MetalBuffer,
                                                         top1_value_buf : ML::MetalBuffer,
                                                         profile_label : String = "head_top1_rows_resident_append") : Bool
          return false if command.committed?
          return false unless head_top1_fused_enabled?
          return false unless out_qw.type.q6_k?
          return false unless out_qw.in_dim > 0
          return false unless out_qw.in_dim % QK_K == 0
          return false unless out_qw.out_dim > 0
          return false unless rows > 0

          hidden_dim = out_qw.in_dim
          return false unless norm_weight.size == hidden_dim
          expected_weight_bytes = out_qw.out_dim.to_i64 * (hidden_dim // QK_K).to_i64 * out_qw.type.block_bytes.to_i64
          return false unless out_qw.raw.size.to_i64 == expected_weight_bytes
          hidden_elements = rows.to_i64 * hidden_dim.to_i64
          return false if hidden_elements > Int64::MAX // sizeof(Float32)
          hidden_bytes = hidden_elements * sizeof(Float32)
          row_bytes = rows.to_i64 * sizeof(Float32)
          return false if x_buf.size < hidden_bytes
          return false if top1_id_buf.size < row_bytes
          return false if top1_value_buf.size < row_bytes

          ML::Metal::Device.init!
          scratch_arena.with do
            tile_count_i64 = (out_qw.out_dim.to_i64 + HEAD_TOP1_ROWS_PER_TG - 1) // HEAD_TOP1_ROWS_PER_TG
            return false if tile_count_i64 > Int32::MAX
            tile_count = tile_count_i64.to_i32
            tile_slots = rows.to_i64 * tile_count.to_i64
            return false if tile_slots > Int64::MAX // sizeof(Float32)
            norm_w_buf = Scratch.get(:head_top1_rows_append_norm_w, norm_weight.size.to_i64 * sizeof(Float32))
            normed_buf = Scratch.get(:head_top1_rows_append_normed, hidden_bytes)
            tile_values_buf = Scratch.get(:head_top1_rows_append_tile_values, tile_slots * sizeof(Float32))
            tile_ids_buf = Scratch.get(:head_top1_rows_append_tile_ids, tile_slots * sizeof(UInt32))
            norm_w_buf.write(norm_weight)

            out_w_buf, out_w_off = weight_slot(out_qw)

            norm_enc = ML::Metal::ComputeEncoder.new(command)
            encode_rmsnorm_rows(norm_enc, x_buf, norm_w_buf, normed_buf, hidden_dim, rows, eps)
            norm_enc.end_encoding

            head_top1_enc = ML::Metal::ComputeEncoder.new(command)
            head_top1_enc.set_pipeline(mv6_top1_tiles_batch_pipeline)
            head_top1_enc.set_buffer(out_w_buf, 0, ML::Metal::BufferAccess::Read, offset: out_w_off)
            head_top1_enc.set_buffer(normed_buf, 1)
            head_top1_enc.set_buffer(tile_values_buf, 2, ML::Metal::BufferAccess::Write)
            head_top1_enc.set_buffer(tile_ids_buf, 3, ML::Metal::BufferAccess::Write)
            head_top1_enc.set_value(hidden_dim.to_u32, 4)
            head_top1_enc.set_value(out_qw.out_dim.to_u32, 5)
            head_top1_enc.set_value(tile_count.to_u32, 6)
            head_top1_enc.dispatch_threadgroups({tile_count, rows, 1}, {64, 1, 1})
            head_top1_enc.end_encoding

            reduce_top1_enc = ML::Metal::ComputeEncoder.new(command)
            reduce_top1_enc.set_pipeline(top1_reduce_tiles_batch_pipeline)
            reduce_top1_enc.set_buffer(tile_values_buf, 0)
            reduce_top1_enc.set_buffer(tile_ids_buf, 1)
            reduce_top1_enc.set_buffer(top1_id_buf, 2, ML::Metal::BufferAccess::Write)
            reduce_top1_enc.set_buffer(top1_value_buf, 3, ML::Metal::BufferAccess::Write)
            reduce_top1_enc.set_value(tile_count.to_u32, 4)
            reduce_top1_enc.dispatch_threadgroups({rows, 1, 1}, {256, 1, 1})
            reduce_top1_enc.end_encoding

            Profile.bump_route_marker(profile_label)
            true
          end
        end

        # Encode one resident hidden row through output RMSNorm and the
        # Q6_K-only allowed-token top-1 head into a caller-owned command
        # buffer. The caller owns commit, completion, result-buffer lifetime,
        # and result visibility.
        def self.encode_rmsnorm_project_top1_allowed_ids_buffer(command : ML::Metal::CommandBuffer,
                                                                x_buf : ML::MetalBuffer,
                                                                element_offset : Int64,
                                                                norm_weight : Array(Float32),
                                                                out_qw : QuantWeight,
                                                                eps : Float32,
                                                                allowed_ids : Array(Int32),
                                                                top1_id_buf : ML::MetalBuffer,
                                                                top1_value_buf : ML::MetalBuffer,
                                                                profile_label : String = "head_top1_allowed_resident_append") : Bool
          return false unless can_use_head_top1_allowed?(out_qw)
          return false if command.committed?
          return false if element_offset < 0
          return false if allowed_ids.empty?

          hidden_dim = out_qw.in_dim
          return false unless norm_weight.size == hidden_dim
          allowed_ids.each do |id|
            return false if id < 0 || id >= out_qw.out_dim
          end

          hidden_dim_i64 = hidden_dim.to_i64
          return false if element_offset > Int64::MAX - hidden_dim_i64
          required_elements = element_offset + hidden_dim_i64
          return false if required_elements > Int64::MAX // sizeof(Float32)
          return false if x_buf.size < required_elements * sizeof(Float32)
          return false if top1_id_buf.size < sizeof(UInt32)
          return false if top1_value_buf.size < sizeof(Float32)

          ML::Metal::Device.init!

          allowed_n = allowed_ids.size
          tile_count = (allowed_n + HEAD_TOP1_ROWS_PER_TG - 1) // HEAD_TOP1_ROWS_PER_TG
          norm_w_buf = Scratch.get(:head_top1_allowed_resident_norm_w,
            norm_weight.size.to_i64 * sizeof(Float32))
          normed_buf = Scratch.get(:head_top1_allowed_resident_normed,
            hidden_dim_i64 * sizeof(Float32))
          allowed_ids_buf = Scratch.get(:head_top1_allowed_resident_ids,
            allowed_n.to_i64 * sizeof(UInt32))
          tile_values_buf = Scratch.get(:head_top1_allowed_resident_tile_values,
            tile_count.to_i64 * sizeof(Float32))
          tile_ids_buf = Scratch.get(:head_top1_allowed_resident_tile_ids,
            tile_count.to_i64 * sizeof(UInt32))
          norm_w_buf.write(norm_weight)
          allowed_ptr = allowed_ids_buf.contents.as(Pointer(UInt32))
          allowed_ids.each_with_index { |id, i| allowed_ptr[i] = id.to_u32 }

          norm_enc = ML::Metal::ComputeEncoder.new(command)
          norm_enc.set_pipeline(rmsnorm_vec_pipeline)
          norm_enc.set_buffer(x_buf, 0, ML::Metal::BufferAccess::Read,
            offset: element_offset * sizeof(Float32))
          norm_enc.set_buffer(norm_w_buf, 1)
          norm_enc.set_buffer(normed_buf, 2, ML::Metal::BufferAccess::Write)
          norm_enc.set_value(hidden_dim.to_u32, 3)
          norm_enc.set_value(eps, 4)
          norm_enc.dispatch_threadgroups({1, 1, 1}, {256, 1, 1})
          norm_enc.end_encoding

          head_top1_enc = ML::Metal::ComputeEncoder.new(command)
          encoded = encode_head_top1_allowed_no_norm_to_buffers(
            head_top1_enc, out_qw, normed_buf, allowed_ids_buf, allowed_n,
            tile_values_buf, tile_ids_buf, top1_id_buf, top1_value_buf,
          )
          head_top1_enc.end_encoding
          Profile.bump_route_marker(profile_label) if encoded
          encoded
        end

        private def self.rmsnorm_project_top1_buffer_impl(x_buf : ML::MetalBuffer,
                                                          element_offset : Int64,
                                                          norm_weight : Array(Float32),
                                                          out_qw : QuantWeight,
                                                          eps : Float32,
                                                          profile_label : String) : Array(Float32)?
          return nil unless can_use_head_top1_fused?(out_qw)
          return nil if element_offset < 0
          hidden_dim = out_qw.in_dim
          return nil unless norm_weight.size == hidden_dim
          hidden_dim_i64 = hidden_dim.to_i64
          return nil if element_offset > Int64::MAX - hidden_dim_i64
          required_elements = element_offset + hidden_dim_i64
          return nil if required_elements > Int64::MAX // sizeof(Float32)
          return nil if x_buf.size < required_elements * sizeof(Float32)

          ML::Metal::Device.init!

          tile_count = (out_qw.out_dim + HEAD_TOP1_ROWS_PER_TG - 1) // HEAD_TOP1_ROWS_PER_TG
          norm_w_buf = Scratch.get(:head_top1_norm_w, norm_weight.size.to_i64 * sizeof(Float32))
          normed_buf = Scratch.get(:head_top1_normed, hidden_dim.to_i64 * sizeof(Float32))
          tile_values_buf = Scratch.get(:head_top1_tile_values, tile_count.to_i64 * sizeof(Float32))
          tile_ids_buf = Scratch.get(:head_top1_tile_ids, tile_count.to_i64 * sizeof(UInt32))
          top1_id_buf = Scratch.get(:head_top1_id, sizeof(UInt32).to_i64)
          top1_value_buf = Scratch.get(:head_top1_value, sizeof(Float32).to_i64)
          norm_w_buf.write(norm_weight)

          out_w_buf, out_w_off = weight_slot(out_qw)

          t0 = Time.instant if Profile.enabled?
          cmd = ML::Metal::CommandBuffer.new

          norm_enc = ML::Metal::ComputeEncoder.new(cmd)
          norm_enc.set_pipeline(rmsnorm_vec_pipeline)
          norm_enc.set_buffer(x_buf, 0, ML::Metal::BufferAccess::Read,
            offset: element_offset * sizeof(Float32))
          norm_enc.set_buffer(norm_w_buf, 1)
          norm_enc.set_buffer(normed_buf, 2, ML::Metal::BufferAccess::Write)
          norm_enc.set_value(hidden_dim.to_u32, 3)
          norm_enc.set_value(eps, 4)
          norm_enc.dispatch_threadgroups({1, 1, 1}, {256, 1, 1})
          norm_enc.end_encoding

          head_top1_enc = ML::Metal::ComputeEncoder.new(cmd)
          head_top1_enc.set_pipeline(out_qw.type.q8_0? ? mv8_top1_tiles_pipeline : mv6_top1_tiles_pipeline)
          profile_bump_head_top1_shape(profile_label, out_qw)
          head_top1_enc.set_buffer(out_w_buf, 0, ML::Metal::BufferAccess::Read, offset: out_w_off)
          head_top1_enc.set_buffer(normed_buf, 1)
          head_top1_enc.set_buffer(tile_values_buf, 2, ML::Metal::BufferAccess::Write)
          head_top1_enc.set_buffer(tile_ids_buf, 3, ML::Metal::BufferAccess::Write)
          head_top1_enc.set_value(out_qw.in_dim.to_u32, 4)
          head_top1_enc.set_value(out_qw.out_dim.to_u32, 5)
          head_top1_enc.dispatch_threadgroups({tile_count, 1, 1}, {out_qw.type.q8_0? ? MV_Q8_NSG * 32 : 64, 1, 1})
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
            Profile.bump_gemv(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          result
        end

        def self.rmsnorm_project_top2(x : Array(Float32),
                                      norm_weight : Array(Float32),
                                      out_qw : QuantWeight,
                                      eps : Float32) : Array(Float32)?
          return nil unless can_use_head_top1_fused?(out_qw)

          ML::Metal::Device.init!

          hidden_dim = x.size
          tile_count = (out_qw.out_dim + HEAD_TOP1_ROWS_PER_TG - 1) // HEAD_TOP1_ROWS_PER_TG
          x_buf = Scratch.get(:head_top2_x, hidden_dim.to_i64 * sizeof(Float32))
          norm_w_buf = Scratch.get(:head_top2_norm_w, norm_weight.size.to_i64 * sizeof(Float32))
          normed_buf = Scratch.get(:head_top2_normed, hidden_dim.to_i64 * sizeof(Float32))
          tile_values_buf = Scratch.get(:head_top2_tile_values, tile_count.to_i64 * sizeof(Float32))
          tile_ids_buf = Scratch.get(:head_top2_tile_ids, tile_count.to_i64 * sizeof(UInt32))
          second_tile_values_buf = Scratch.get(:head_top2_second_tile_values, tile_count.to_i64 * sizeof(Float32))
          second_tile_ids_buf = Scratch.get(:head_top2_second_tile_ids, tile_count.to_i64 * sizeof(UInt32))
          top1_id_buf = Scratch.get(:head_top2_id, sizeof(UInt32).to_i64)
          top1_value_buf = Scratch.get(:head_top2_value, sizeof(Float32).to_i64)
          second_id_buf = Scratch.get(:head_top2_second_id, sizeof(UInt32).to_i64)
          second_value_buf = Scratch.get(:head_top2_second_value, sizeof(Float32).to_i64)
          x_buf.write(x)
          norm_w_buf.write(norm_weight)

          out_w_buf, out_w_off = weight_slot(out_qw)

          t0 = Time.instant if Profile.enabled?
          cmd = ML::Metal::CommandBuffer.new

          norm_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_rmsnorm_vec(norm_enc, x_buf, norm_w_buf, normed_buf, hidden_dim, eps)
          norm_enc.end_encoding

          head_top2_enc = ML::Metal::ComputeEncoder.new(cmd)
          head_top2_enc.set_pipeline(out_qw.type.q8_0? ? mv8_top2_tiles_pipeline : mv6_top2_tiles_pipeline)
          profile_bump_head_top1_shape("head_top2", out_qw)
          head_top2_enc.set_buffer(out_w_buf, 0, ML::Metal::BufferAccess::Read, offset: out_w_off)
          head_top2_enc.set_buffer(normed_buf, 1)
          head_top2_enc.set_buffer(tile_values_buf, 2, ML::Metal::BufferAccess::Write)
          head_top2_enc.set_buffer(tile_ids_buf, 3, ML::Metal::BufferAccess::Write)
          head_top2_enc.set_buffer(second_tile_values_buf, 4, ML::Metal::BufferAccess::Write)
          head_top2_enc.set_buffer(second_tile_ids_buf, 5, ML::Metal::BufferAccess::Write)
          head_top2_enc.set_value(out_qw.in_dim.to_u32, 6)
          head_top2_enc.set_value(out_qw.out_dim.to_u32, 7)
          head_top2_enc.dispatch_threadgroups({tile_count, 1, 1}, {out_qw.type.q8_0? ? MV_Q8_NSG * 32 : 64, 1, 1})
          head_top2_enc.end_encoding

          reduce_top2_enc = ML::Metal::ComputeEncoder.new(cmd)
          reduce_top2_enc.set_pipeline(top2_reduce_tiles_pipeline)
          reduce_top2_enc.set_buffer(tile_values_buf, 0)
          reduce_top2_enc.set_buffer(tile_ids_buf, 1)
          reduce_top2_enc.set_buffer(second_tile_values_buf, 2)
          reduce_top2_enc.set_buffer(second_tile_ids_buf, 3)
          reduce_top2_enc.set_buffer(top1_id_buf, 4, ML::Metal::BufferAccess::Write)
          reduce_top2_enc.set_buffer(top1_value_buf, 5, ML::Metal::BufferAccess::Write)
          reduce_top2_enc.set_buffer(second_id_buf, 6, ML::Metal::BufferAccess::Write)
          reduce_top2_enc.set_buffer(second_value_buf, 7, ML::Metal::BufferAccess::Write)
          reduce_top2_enc.set_value(tile_count.to_u32, 8)
          reduce_top2_enc.dispatch_threadgroups({1, 1, 1}, {256, 1, 1})
          reduce_top2_enc.end_encoding

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_top2(top1_id_buf, top1_value_buf, second_id_buf, second_value_buf)
          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_gemv(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          result
        end

        def self.rmsnorm_project_top1_allowed_ids(x : Array(Float32),
                                                  norm_weight : Array(Float32),
                                                  out_qw : QuantWeight,
                                                  eps : Float32,
                                                  allowed_ids : Array(Int32)) : Array(Float32)?
          return nil unless can_use_head_top1_allowed?(out_qw)
          return nil if allowed_ids.empty?
          raise "rmsnorm_project_top1_allowed_ids input mismatch: expected #{out_qw.in_dim}, got #{x.size}" unless x.size == out_qw.in_dim
          raise "rmsnorm_project_top1_allowed_ids norm mismatch: expected #{out_qw.in_dim}, got #{norm_weight.size}" unless norm_weight.size == out_qw.in_dim
          allowed_ids.each do |id|
            raise "allowed token id #{id} out of range 0...#{out_qw.out_dim}" if id < 0 || id >= out_qw.out_dim
          end

          ML::Metal::Device.init!

          hidden_dim = x.size
          allowed_n = allowed_ids.size
          tile_count = (allowed_n + HEAD_TOP1_ROWS_PER_TG - 1) // HEAD_TOP1_ROWS_PER_TG
          x_buf = Scratch.get(:head_top1_allowed_x, hidden_dim.to_i64 * sizeof(Float32))
          norm_w_buf = Scratch.get(:head_top1_allowed_norm_w, norm_weight.size.to_i64 * sizeof(Float32))
          normed_buf = Scratch.get(:head_top1_allowed_normed, hidden_dim.to_i64 * sizeof(Float32))
          allowed_ids_buf = Scratch.get(:head_top1_allowed_ids, allowed_n.to_i64 * sizeof(UInt32))
          tile_values_buf = Scratch.get(:head_top1_allowed_tile_values, tile_count.to_i64 * sizeof(Float32))
          tile_ids_buf = Scratch.get(:head_top1_allowed_tile_ids, tile_count.to_i64 * sizeof(UInt32))
          top1_id_buf = Scratch.get(:head_top1_allowed_id, sizeof(UInt32).to_i64)
          top1_value_buf = Scratch.get(:head_top1_allowed_value, sizeof(Float32).to_i64)
          x_buf.write(x)
          norm_w_buf.write(norm_weight)
          allowed_ptr = allowed_ids_buf.contents.as(Pointer(UInt32))
          allowed_ids.each_with_index { |id, i| allowed_ptr[i] = id.to_u32 }

          out_w_buf, out_w_off = weight_slot(out_qw)

          t0 = Time.instant if Profile.enabled?
          cmd = ML::Metal::CommandBuffer.new

          norm_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_rmsnorm_vec(norm_enc, x_buf, norm_w_buf, normed_buf, hidden_dim, eps)
          norm_enc.end_encoding

          head_top1_enc = ML::Metal::ComputeEncoder.new(cmd)
          head_top1_enc.set_pipeline(mv6_top1_allowed_tiles_pipeline)
          profile_bump_head_top1_shape("head_top1_allowed#{allowed_n}", out_qw, rows: allowed_n)
          head_top1_enc.set_buffer(out_w_buf, 0, ML::Metal::BufferAccess::Read, offset: out_w_off)
          head_top1_enc.set_buffer(normed_buf, 1)
          head_top1_enc.set_buffer(allowed_ids_buf, 2)
          head_top1_enc.set_buffer(tile_values_buf, 3, ML::Metal::BufferAccess::Write)
          head_top1_enc.set_buffer(tile_ids_buf, 4, ML::Metal::BufferAccess::Write)
          head_top1_enc.set_value(out_qw.in_dim.to_u32, 5)
          head_top1_enc.set_value(out_qw.out_dim.to_u32, 6)
          head_top1_enc.set_value(allowed_n.to_u32, 7)
          head_top1_enc.dispatch_threadgroups({tile_count, 1, 1}, {64, 1, 1})
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
            Profile.bump_gemv(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          result
        end

        def self.project_top1_no_norm(out_qw : QuantWeight,
                                      x : Array(Float32)) : Array(Float32)?
          return nil unless can_use_head_top1_fused?(out_qw)
          raise "project_top1_no_norm input mismatch: expected #{out_qw.in_dim}, got #{x.size}" unless x.size == out_qw.in_dim

          ML::Metal::Device.init!

          tile_count = (out_qw.out_dim + HEAD_TOP1_ROWS_PER_TG - 1) // HEAD_TOP1_ROWS_PER_TG
          x_buf = Scratch.get(:head_top1_nonorm_x, out_qw.in_dim.to_i64 * sizeof(Float32))
          tile_values_buf = Scratch.get(:head_top1_nonorm_tile_values, tile_count.to_i64 * sizeof(Float32))
          tile_ids_buf = Scratch.get(:head_top1_nonorm_tile_ids, tile_count.to_i64 * sizeof(UInt32))
          top1_id_buf = Scratch.get(:head_top1_nonorm_id, sizeof(UInt32).to_i64)
          top1_value_buf = Scratch.get(:head_top1_nonorm_value, sizeof(Float32).to_i64)
          x_buf.write(x)

          out_w_buf, out_w_off = weight_slot(out_qw)

          t0 = Time.instant if Profile.enabled?
          cmd = ML::Metal::CommandBuffer.new

          head_top1_enc = ML::Metal::ComputeEncoder.new(cmd)
          head_top1_enc.set_pipeline(out_qw.type.q8_0? ? mv8_top1_tiles_pipeline : mv6_top1_tiles_pipeline)
          profile_bump_head_top1_shape("head_top1_no_norm", out_qw)
          head_top1_enc.set_buffer(out_w_buf, 0, ML::Metal::BufferAccess::Read, offset: out_w_off)
          head_top1_enc.set_buffer(x_buf, 1)
          head_top1_enc.set_buffer(tile_values_buf, 2, ML::Metal::BufferAccess::Write)
          head_top1_enc.set_buffer(tile_ids_buf, 3, ML::Metal::BufferAccess::Write)
          head_top1_enc.set_value(out_qw.in_dim.to_u32, 4)
          head_top1_enc.set_value(out_qw.out_dim.to_u32, 5)
          head_top1_enc.dispatch_threadgroups({tile_count, 1, 1}, {out_qw.type.q8_0? ? MV_Q8_NSG * 32 : 64, 1, 1})
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
            Profile.bump_gemv(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          result
        end

        def self.project_top1_no_norm_graph(out_qw : QuantWeight,
                                            x : Array(Float32)) : Array(Float32)?
          return nil unless can_use_head_top1_fused?(out_qw)
          raise "project_top1_no_norm_graph input mismatch: expected #{out_qw.in_dim}, got #{x.size}" unless x.size == out_qw.in_dim

          ML::Metal::Device.init!

          tile_count = head_top1_tile_count(out_qw)
          x_buf = Scratch.get(:head_top1_nonorm_graph_x, out_qw.in_dim.to_i64 * sizeof(Float32))
          tile_values_buf = Scratch.get(:head_top1_nonorm_graph_tile_values, tile_count.to_i64 * sizeof(Float32))
          tile_ids_buf = Scratch.get(:head_top1_nonorm_graph_tile_ids, tile_count.to_i64 * sizeof(UInt32))
          top1_id_buf = Scratch.get(:head_top1_nonorm_graph_id, sizeof(UInt32).to_i64)
          top1_value_buf = Scratch.get(:head_top1_nonorm_graph_value, sizeof(Float32).to_i64)
          x_buf.write(x)

          t0 = Time.instant if Profile.enabled?
          graph = ML::Metal::ComputeGraph.new
          enc = ML::Metal::GraphEncoder.new(graph)
          return nil unless encode_head_top1_no_norm_to_buffers(enc, out_qw, x_buf, tile_values_buf, tile_ids_buf, top1_id_buf, top1_value_buf)
          graph.compile!
          t_compile = Time.instant if Profile.enabled?

          cmd = ML::Metal::CommandBuffer.new
          graph.encode(cmd)
          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_top1(top1_id_buf, top1_value_buf)
          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_gemv(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
            Profile.bump_group("head_top1_no_norm_graph.compile",
              (t_compile.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              0_i64,
              0_i64)
          end
          result
        end

        def self.encode_head_top1_no_norm_to_buffers(enc : ML::Metal::ComputeEncoder | ML::Metal::GraphEncoder,
                                                     out_qw : QuantWeight,
                                                     x_buf : ML::MetalBuffer,
                                                     tile_values_buf : ML::MetalBuffer,
                                                     tile_ids_buf : ML::MetalBuffer,
                                                     top1_id_buf : ML::MetalBuffer,
                                                     top1_value_buf : ML::MetalBuffer) : Bool
          return false unless can_use_head_top1_fused?(out_qw)
          return false if x_buf.size < out_qw.in_dim.to_i64 * sizeof(Float32)

          tile_count = head_top1_tile_count(out_qw)
          return false if tile_values_buf.size < tile_count.to_i64 * sizeof(Float32)
          return false if tile_ids_buf.size < tile_count.to_i64 * sizeof(UInt32)
          return false if top1_id_buf.size < sizeof(UInt32)
          return false if top1_value_buf.size < sizeof(Float32)

          out_w_buf, out_w_off = weight_slot(out_qw)

          enc.set_pipeline(out_qw.type.q8_0? ? mv8_top1_tiles_pipeline : mv6_top1_tiles_pipeline)
          profile_bump_head_top1_shape("head_top1_no_norm_resident", out_qw)
          enc.set_buffer(out_w_buf, 0, ML::Metal::BufferAccess::Read, offset: out_w_off)
          enc.set_buffer(x_buf, 1)
          enc.set_buffer(tile_values_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_buffer(tile_ids_buf, 3, ML::Metal::BufferAccess::Write)
          enc.set_value(out_qw.in_dim.to_u32, 4)
          enc.set_value(out_qw.out_dim.to_u32, 5)
          enc.dispatch_threadgroups({tile_count, 1, 1}, {out_qw.type.q8_0? ? MV_Q8_NSG * 32 : 64, 1, 1})

          enc.set_pipeline(top1_reduce_tiles_pipeline)
          enc.set_buffer(tile_values_buf, 0)
          enc.set_buffer(tile_ids_buf, 1)
          enc.set_buffer(top1_id_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_buffer(top1_value_buf, 3, ML::Metal::BufferAccess::Write)
          enc.set_value(tile_count.to_u32, 4)
          enc.dispatch_threadgroups({1, 1, 1}, {256, 1, 1})
          true
        end

        def self.encode_head_top2_no_norm_to_buffers(enc : ML::Metal::ComputeEncoder,
                                                     out_qw : QuantWeight,
                                                     x_buf : ML::MetalBuffer,
                                                     tile_values_buf : ML::MetalBuffer,
                                                     tile_ids_buf : ML::MetalBuffer,
                                                     second_tile_values_buf : ML::MetalBuffer,
                                                     second_tile_ids_buf : ML::MetalBuffer,
                                                     top1_id_buf : ML::MetalBuffer,
                                                     top1_value_buf : ML::MetalBuffer,
                                                     second_id_buf : ML::MetalBuffer,
                                                     second_value_buf : ML::MetalBuffer) : Bool
          return false unless can_use_head_top1_fused?(out_qw)
          return false if x_buf.size < out_qw.in_dim.to_i64 * sizeof(Float32)

          tile_count = head_top1_tile_count(out_qw)
          return false if tile_values_buf.size < tile_count.to_i64 * sizeof(Float32)
          return false if tile_ids_buf.size < tile_count.to_i64 * sizeof(UInt32)
          return false if second_tile_values_buf.size < tile_count.to_i64 * sizeof(Float32)
          return false if second_tile_ids_buf.size < tile_count.to_i64 * sizeof(UInt32)
          return false if top1_id_buf.size < sizeof(UInt32)
          return false if top1_value_buf.size < sizeof(Float32)
          return false if second_id_buf.size < sizeof(UInt32)
          return false if second_value_buf.size < sizeof(Float32)

          out_w_buf, out_w_off = weight_slot(out_qw)

          enc.set_pipeline(out_qw.type.q8_0? ? mv8_top2_tiles_pipeline : mv6_top2_tiles_pipeline)
          profile_bump_head_top1_shape("head_top2_no_norm_resident", out_qw)
          enc.set_buffer(out_w_buf, 0, ML::Metal::BufferAccess::Read, offset: out_w_off)
          enc.set_buffer(x_buf, 1)
          enc.set_buffer(tile_values_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_buffer(tile_ids_buf, 3, ML::Metal::BufferAccess::Write)
          enc.set_buffer(second_tile_values_buf, 4, ML::Metal::BufferAccess::Write)
          enc.set_buffer(second_tile_ids_buf, 5, ML::Metal::BufferAccess::Write)
          enc.set_value(out_qw.in_dim.to_u32, 6)
          enc.set_value(out_qw.out_dim.to_u32, 7)
          enc.dispatch_threadgroups({tile_count, 1, 1}, {out_qw.type.q8_0? ? MV_Q8_NSG * 32 : 64, 1, 1})

          enc.set_pipeline(top2_reduce_tiles_pipeline)
          enc.set_buffer(tile_values_buf, 0)
          enc.set_buffer(tile_ids_buf, 1)
          enc.set_buffer(second_tile_values_buf, 2)
          enc.set_buffer(second_tile_ids_buf, 3)
          enc.set_buffer(top1_id_buf, 4, ML::Metal::BufferAccess::Write)
          enc.set_buffer(top1_value_buf, 5, ML::Metal::BufferAccess::Write)
          enc.set_buffer(second_id_buf, 6, ML::Metal::BufferAccess::Write)
          enc.set_buffer(second_value_buf, 7, ML::Metal::BufferAccess::Write)
          enc.set_value(tile_count.to_u32, 8)
          enc.dispatch_threadgroups({1, 1, 1}, {256, 1, 1})
          true
        end

        def self.encode_head_top1_allowed_no_norm_to_buffers(enc : ML::Metal::ComputeEncoder,
                                                             out_qw : QuantWeight,
                                                             x_buf : ML::MetalBuffer,
                                                             allowed_ids_buf : ML::MetalBuffer,
                                                             allowed_n : Int32,
                                                             tile_values_buf : ML::MetalBuffer,
                                                             tile_ids_buf : ML::MetalBuffer,
                                                             top1_id_buf : ML::MetalBuffer,
                                                             top1_value_buf : ML::MetalBuffer) : Bool
          return false unless can_use_head_top1_allowed?(out_qw)
          return false if allowed_n <= 0
          return false if x_buf.size < out_qw.in_dim.to_i64 * sizeof(Float32)
          return false if allowed_ids_buf.size < allowed_n.to_i64 * sizeof(UInt32)

          tile_count = (allowed_n + HEAD_TOP1_ROWS_PER_TG - 1) // HEAD_TOP1_ROWS_PER_TG
          return false if tile_values_buf.size < tile_count.to_i64 * sizeof(Float32)
          return false if tile_ids_buf.size < tile_count.to_i64 * sizeof(UInt32)
          return false if top1_id_buf.size < sizeof(UInt32)
          return false if top1_value_buf.size < sizeof(Float32)

          out_w_buf, out_w_off = weight_slot(out_qw)

          enc.set_pipeline(mv6_top1_allowed_tiles_pipeline)
          profile_bump_head_top1_shape("head_top1_allowed#{allowed_n}_resident", out_qw, rows: allowed_n)
          enc.set_buffer(out_w_buf, 0, ML::Metal::BufferAccess::Read, offset: out_w_off)
          enc.set_buffer(x_buf, 1)
          enc.set_buffer(allowed_ids_buf, 2)
          enc.set_buffer(tile_values_buf, 3, ML::Metal::BufferAccess::Write)
          enc.set_buffer(tile_ids_buf, 4, ML::Metal::BufferAccess::Write)
          enc.set_value(out_qw.in_dim.to_u32, 5)
          enc.set_value(out_qw.out_dim.to_u32, 6)
          enc.set_value(allowed_n.to_u32, 7)
          enc.dispatch_threadgroups({tile_count, 1, 1}, {64, 1, 1})

          enc.set_pipeline(top1_reduce_tiles_pipeline)
          enc.set_buffer(tile_values_buf, 0)
          enc.set_buffer(tile_ids_buf, 1)
          enc.set_buffer(top1_id_buf, 2, ML::Metal::BufferAccess::Write)
          enc.set_buffer(top1_value_buf, 3, ML::Metal::BufferAccess::Write)
          enc.set_value(tile_count.to_u32, 4)
          enc.dispatch_threadgroups({1, 1, 1}, {256, 1, 1})
          true
        end

        def self.project_top2_no_norm(out_qw : QuantWeight,
                                      x : Array(Float32)) : Array(Float32)?
          return nil unless can_use_head_top1_fused?(out_qw)
          raise "project_top2_no_norm input mismatch: expected #{out_qw.in_dim}, got #{x.size}" unless x.size == out_qw.in_dim

          ML::Metal::Device.init!

          tile_count = (out_qw.out_dim + HEAD_TOP1_ROWS_PER_TG - 1) // HEAD_TOP1_ROWS_PER_TG
          x_buf = Scratch.get(:head_top2_nonorm_x, out_qw.in_dim.to_i64 * sizeof(Float32))
          tile_values_buf = Scratch.get(:head_top2_nonorm_tile_values, tile_count.to_i64 * sizeof(Float32))
          tile_ids_buf = Scratch.get(:head_top2_nonorm_tile_ids, tile_count.to_i64 * sizeof(UInt32))
          second_tile_values_buf = Scratch.get(:head_top2_nonorm_second_tile_values, tile_count.to_i64 * sizeof(Float32))
          second_tile_ids_buf = Scratch.get(:head_top2_nonorm_second_tile_ids, tile_count.to_i64 * sizeof(UInt32))
          top1_id_buf = Scratch.get(:head_top2_nonorm_id, sizeof(UInt32).to_i64)
          top1_value_buf = Scratch.get(:head_top2_nonorm_value, sizeof(Float32).to_i64)
          second_id_buf = Scratch.get(:head_top2_nonorm_second_id, sizeof(UInt32).to_i64)
          second_value_buf = Scratch.get(:head_top2_nonorm_second_value, sizeof(Float32).to_i64)
          x_buf.write(x)

          out_w_buf, out_w_off = weight_slot(out_qw)

          t0 = Time.instant if Profile.enabled?
          cmd = ML::Metal::CommandBuffer.new

          head_top2_enc = ML::Metal::ComputeEncoder.new(cmd)
          head_top2_enc.set_pipeline(out_qw.type.q8_0? ? mv8_top2_tiles_pipeline : mv6_top2_tiles_pipeline)
          head_top2_enc.set_buffer(out_w_buf, 0, ML::Metal::BufferAccess::Read, offset: out_w_off)
          head_top2_enc.set_buffer(x_buf, 1)
          head_top2_enc.set_buffer(tile_values_buf, 2, ML::Metal::BufferAccess::Write)
          head_top2_enc.set_buffer(tile_ids_buf, 3, ML::Metal::BufferAccess::Write)
          head_top2_enc.set_buffer(second_tile_values_buf, 4, ML::Metal::BufferAccess::Write)
          head_top2_enc.set_buffer(second_tile_ids_buf, 5, ML::Metal::BufferAccess::Write)
          head_top2_enc.set_value(out_qw.in_dim.to_u32, 6)
          head_top2_enc.set_value(out_qw.out_dim.to_u32, 7)
          head_top2_enc.dispatch_threadgroups({tile_count, 1, 1}, {out_qw.type.q8_0? ? MV_Q8_NSG * 32 : 64, 1, 1})
          head_top2_enc.end_encoding

          reduce_top2_enc = ML::Metal::ComputeEncoder.new(cmd)
          reduce_top2_enc.set_pipeline(top2_reduce_tiles_pipeline)
          reduce_top2_enc.set_buffer(tile_values_buf, 0)
          reduce_top2_enc.set_buffer(tile_ids_buf, 1)
          reduce_top2_enc.set_buffer(second_tile_values_buf, 2)
          reduce_top2_enc.set_buffer(second_tile_ids_buf, 3)
          reduce_top2_enc.set_buffer(top1_id_buf, 4, ML::Metal::BufferAccess::Write)
          reduce_top2_enc.set_buffer(top1_value_buf, 5, ML::Metal::BufferAccess::Write)
          reduce_top2_enc.set_buffer(second_id_buf, 6, ML::Metal::BufferAccess::Write)
          reduce_top2_enc.set_buffer(second_value_buf, 7, ML::Metal::BufferAccess::Write)
          reduce_top2_enc.set_value(tile_count.to_u32, 8)
          reduce_top2_enc.dispatch_threadgroups({1, 1, 1}, {256, 1, 1})
          reduce_top2_enc.end_encoding

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_top2(top1_id_buf, top1_value_buf, second_id_buf, second_value_buf)
          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_gemv(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          result
        end

        def self.rmsnorm_project_top1_rows(x : Array(Float32),
                                           rows : Int32,
                                           norm_weight : Array(Float32),
                                           out_qw : QuantWeight,
                                           eps : Float32) : Array({Int32, Float32})?
          return nil unless head_top1_fused_enabled?
          return nil unless out_qw.type.q6_k?
          return nil unless out_qw.in_dim % QK_K == 0
          return nil unless rows > 0
          hidden_dim = out_qw.in_dim
          return nil unless x.size == rows * hidden_dim

          ML::Metal::Device.init!

          tile_count = (out_qw.out_dim + HEAD_TOP1_ROWS_PER_TG - 1) // HEAD_TOP1_ROWS_PER_TG
          x_buf = Scratch.get(:head_top1_rows_x, x.size.to_i64 * sizeof(Float32))
          norm_w_buf = Scratch.get(:head_top1_rows_norm_w, norm_weight.size.to_i64 * sizeof(Float32))
          normed_buf = Scratch.get(:head_top1_rows_normed, x.size.to_i64 * sizeof(Float32))
          tile_values_buf = Scratch.get(:head_top1_rows_tile_values, (rows * tile_count).to_i64 * sizeof(Float32))
          tile_ids_buf = Scratch.get(:head_top1_rows_tile_ids, (rows * tile_count).to_i64 * sizeof(UInt32))
          top1_id_buf = Scratch.get(:head_top1_rows_id, rows.to_i64 * sizeof(UInt32))
          top1_value_buf = Scratch.get(:head_top1_rows_value, rows.to_i64 * sizeof(Float32))
          x_buf.write(x)
          norm_w_buf.write(norm_weight)

          out_w_buf, out_w_off = weight_slot(out_qw)

          t0 = Time.instant if Profile.enabled?
          cmd = ML::Metal::CommandBuffer.new

          norm_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_rmsnorm_rows(norm_enc, x_buf, norm_w_buf, normed_buf, hidden_dim, rows, eps)
          norm_enc.end_encoding

          head_top1_enc = ML::Metal::ComputeEncoder.new(cmd)
          head_top1_enc.set_pipeline(mv6_top1_tiles_batch_pipeline)
          head_top1_enc.set_buffer(out_w_buf, 0, ML::Metal::BufferAccess::Read, offset: out_w_off)
          head_top1_enc.set_buffer(normed_buf, 1)
          head_top1_enc.set_buffer(tile_values_buf, 2, ML::Metal::BufferAccess::Write)
          head_top1_enc.set_buffer(tile_ids_buf, 3, ML::Metal::BufferAccess::Write)
          head_top1_enc.set_value(out_qw.in_dim.to_u32, 4)
          head_top1_enc.set_value(out_qw.out_dim.to_u32, 5)
          head_top1_enc.set_value(tile_count.to_u32, 6)
          head_top1_enc.dispatch_threadgroups({tile_count, rows, 1}, {64, 1, 1})
          head_top1_enc.end_encoding

          reduce_top1_enc = ML::Metal::ComputeEncoder.new(cmd)
          reduce_top1_enc.set_pipeline(top1_reduce_tiles_batch_pipeline)
          reduce_top1_enc.set_buffer(tile_values_buf, 0)
          reduce_top1_enc.set_buffer(tile_ids_buf, 1)
          reduce_top1_enc.set_buffer(top1_id_buf, 2, ML::Metal::BufferAccess::Write)
          reduce_top1_enc.set_buffer(top1_value_buf, 3, ML::Metal::BufferAccess::Write)
          reduce_top1_enc.set_value(tile_count.to_u32, 4)
          reduce_top1_enc.dispatch_threadgroups({rows, 1, 1}, {256, 1, 1})
          reduce_top1_enc.end_encoding

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_top1_rows(top1_id_buf, top1_value_buf, rows)
          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_gemv(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          result
        end

        def self.rmsnorm_project_full_top1_rows_guarded(x : Array(Float32),
                                                        rows : Int32,
                                                        norm_weight : Array(Float32),
                                                        out_qw : QuantWeight,
                                                        eps : Float32) : Array({Int32, Float32})?
          return nil unless out_qw.type.q6_k?
          return nil unless out_qw.in_dim % QK_K == 0
          return nil unless rows > GEMM_BATCH_THRESHOLD
          hidden_dim = out_qw.in_dim
          return nil unless x.size == rows * hidden_dim

          margin = (ENV["QWEN35_HEAD_FULL_ROWS_MARGIN"]? || "0.25").to_f32
          return nil if margin < 0.0_f32

          ML::Metal::Device.init!

          x_buf = Scratch.get(:head_full_rows_guard_x, x.size.to_i64 * sizeof(Float32))
          norm_w_buf = Scratch.get(:head_full_rows_guard_norm_w, norm_weight.size.to_i64 * sizeof(Float32))
          normed_buf = Scratch.get(:head_full_rows_guard_normed, x.size.to_i64 * sizeof(Float32))
          normed16_buf = Scratch.get(:head_full_rows_guard_normed16, x.size.to_i64 * 2_i64)
          bias_buf = Scratch.get("head_full_rows_guard_bias_#{out_qw.out_dim}", out_qw.out_dim.to_i64 * sizeof(Float32))
          logits16_buf = Scratch.get(:head_full_rows_guard_logits16, (rows * out_qw.out_dim).to_i64 * 2_i64)
          top1_id_buf = Scratch.get(:head_full_rows_guard_id, rows.to_i64 * sizeof(UInt32))
          top1_value_buf = Scratch.get(:head_full_rows_guard_value, rows.to_i64 * sizeof(Float32))
          second_id_buf = Scratch.get(:head_full_rows_guard_second_id, rows.to_i64 * sizeof(UInt32))
          second_value_buf = Scratch.get(:head_full_rows_guard_second_value, rows.to_i64 * sizeof(Float32))
          x_buf.write(x)
          norm_w_buf.write(norm_weight)
          ConstCache.write_zero_f32_once("head_full_rows_guard_bias_#{out_qw.out_dim}", bias_buf, out_qw.out_dim)

          out_w_buf, out_w_off = weight_slot(out_qw)

          t0 = Time.instant if Profile.enabled?
          cmd = ML::Metal::CommandBuffer.new

          norm_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_rmsnorm_rows(norm_enc, x_buf, norm_w_buf, normed_buf, hidden_dim, rows, eps)
          norm_enc.end_encoding

          out_enc = ML::Metal::ComputeEncoder.new(cmd)
          Profile.bump_matmul_shape("q6_gemm_top2_guard #{out_qw.type.name} #{out_qw.in_dim}x#{out_qw.out_dim} b#{rows}", out_qw.raw.size.to_i64)
          Profile.bump_conversion("f32_to_f16 q56_gemm_input_guard #{out_qw.in_dim} b#{rows}", (rows * out_qw.in_dim).to_i64 * 6_i64)
          out_enc.set_pipeline(f32_to_f16_pipeline)
          out_enc.set_buffer(normed_buf, 0)
          out_enc.set_buffer(normed16_buf, 1, ML::Metal::BufferAccess::Write)
          out_enc.set_value((rows * out_qw.in_dim).to_u32, 2)
          out_enc.dispatch_1d(rows * out_qw.in_dim, 256)

          out_enc.set_pipeline(mm6_pipeline)
          out_enc.set_buffer(out_w_buf, 0, ML::Metal::BufferAccess::Read, offset: out_w_off)
          out_enc.set_buffer(normed16_buf, 1)
          out_enc.set_buffer(bias_buf, 2)
          out_enc.set_buffer(logits16_buf, 3, ML::Metal::BufferAccess::Write)
          out_enc.set_value(out_qw.in_dim.to_u32, 4)
          out_enc.set_value(out_qw.out_dim.to_u32, 5)
          out_enc.set_value(rows.to_u32, 6)
          out_enc.set_value(0_u32, 7)
          out_enc.set_threadgroup_memory(MM_SHMEM, 0)
          out_enc.dispatch_threadgroups({
            (rows + MM_NR1 - 1) // MM_NR1,
            (out_qw.out_dim + MM_NR0 - 1) // MM_NR0,
            1,
          }, {MM_TG, 1, 1})
          out_enc.end_encoding

          reduce_enc = ML::Metal::ComputeEncoder.new(cmd)
          reduce_enc.set_pipeline(top2_reduce_f16_rows_pipeline)
          reduce_enc.set_buffer(logits16_buf, 0)
          reduce_enc.set_buffer(top1_id_buf, 1, ML::Metal::BufferAccess::Write)
          reduce_enc.set_buffer(top1_value_buf, 2, ML::Metal::BufferAccess::Write)
          reduce_enc.set_buffer(second_id_buf, 3, ML::Metal::BufferAccess::Write)
          reduce_enc.set_buffer(second_value_buf, 4, ML::Metal::BufferAccess::Write)
          reduce_enc.set_value(out_qw.out_dim.to_u32, 5)
          reduce_enc.dispatch_threadgroups({rows, 1, 1}, {256, 1, 1})
          reduce_enc.end_encoding

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?

          top2s = read_shared_top2_rows(top1_id_buf, top1_value_buf, second_id_buf, second_value_buf, rows)
          result = Array({Int32, Float32}).new(rows) { |i| {top2s[i][0], top2s[i][1]} }

          fallback_rows = [] of Int32
          top2s.each_with_index do |(_, top_value, _, second_value), i|
            fallback_rows << i.to_i32 if top_value - second_value < margin
          end

          if !fallback_rows.empty?
            fallback_x = Array(Float32).new(fallback_rows.size * hidden_dim, 0.0_f32)
            fallback_rows.each_with_index do |row, compact_row|
              src_offset = row * hidden_dim
              dst_offset = compact_row * hidden_dim
              hidden_dim.times do |j|
                fallback_x[dst_offset + j] = x[src_offset + j]
              end
            end
            if exact = rmsnorm_project_top1_rows(fallback_x, fallback_rows.size.to_i32, norm_weight, out_qw, eps)
              fallback_rows.each_with_index do |row, compact_row|
                result[row] = exact[compact_row]
              end
            else
              return nil
            end
          end

          if ENV["QWEN35_HEAD_FULL_ROWS_GUARD_TRACE"]? == "1"
            STDERR.puts "head_full_rows_guard rows=#{rows} fallback=#{fallback_rows.size} margin=#{margin}"
          end

          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_gemv(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end

          result
        end

        def self.rmsnorm_project_full_top1_rows(x : Array(Float32),
                                                rows : Int32,
                                                norm_weight : Array(Float32),
                                                out_qw : QuantWeight,
                                                eps : Float32) : Array({Int32, Float32})?
          return nil unless out_qw.type.q6_k?
          return nil unless out_qw.in_dim % QK_K == 0
          return nil unless rows > GEMM_BATCH_THRESHOLD
          hidden_dim = out_qw.in_dim
          return nil unless x.size == rows * hidden_dim

          ML::Metal::Device.init!

          x_buf = Scratch.get(:head_full_rows_x, x.size.to_i64 * sizeof(Float32))
          norm_w_buf = Scratch.get(:head_full_rows_norm_w, norm_weight.size.to_i64 * sizeof(Float32))
          normed_buf = Scratch.get(:head_full_rows_normed, x.size.to_i64 * sizeof(Float32))
          normed16_buf = Scratch.get(:head_full_rows_normed16, x.size.to_i64 * 2_i64)
          bias_buf = Scratch.get("head_full_rows_bias_#{out_qw.out_dim}", out_qw.out_dim.to_i64 * sizeof(Float32))
          logits16_buf = Scratch.get(:head_full_rows_logits16, (rows * out_qw.out_dim).to_i64 * 2_i64)
          top1_id_buf = Scratch.get(:head_full_rows_id, rows.to_i64 * sizeof(UInt32))
          top1_value_buf = Scratch.get(:head_full_rows_value, rows.to_i64 * sizeof(Float32))
          x_buf.write(x)
          norm_w_buf.write(norm_weight)
          ConstCache.write_zero_f32_once("head_full_rows_bias_#{out_qw.out_dim}", bias_buf, out_qw.out_dim)

          out_w_buf, out_w_off = weight_slot(out_qw)

          t0 = Time.instant if Profile.enabled?
          cmd = ML::Metal::CommandBuffer.new

          norm_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_rmsnorm_rows(norm_enc, x_buf, norm_w_buf, normed_buf, hidden_dim, rows, eps)
          norm_enc.end_encoding

          out_enc = ML::Metal::ComputeEncoder.new(cmd)
          Profile.bump_matmul_shape("q6_gemm_top1 #{out_qw.type.name} #{out_qw.in_dim}x#{out_qw.out_dim} b#{rows}", out_qw.raw.size.to_i64)
          Profile.bump_conversion("f32_to_f16 q56_gemm_input #{out_qw.in_dim} b#{rows}", (rows * out_qw.in_dim).to_i64 * 6_i64)
          out_enc.set_pipeline(f32_to_f16_pipeline)
          out_enc.set_buffer(normed_buf, 0)
          out_enc.set_buffer(normed16_buf, 1, ML::Metal::BufferAccess::Write)
          out_enc.set_value((rows * out_qw.in_dim).to_u32, 2)
          out_enc.dispatch_1d(rows * out_qw.in_dim, 256)

          out_enc.set_pipeline(mm6_pipeline)
          out_enc.set_buffer(out_w_buf, 0, ML::Metal::BufferAccess::Read, offset: out_w_off)
          out_enc.set_buffer(normed16_buf, 1)
          out_enc.set_buffer(bias_buf, 2)
          out_enc.set_buffer(logits16_buf, 3, ML::Metal::BufferAccess::Write)
          out_enc.set_value(out_qw.in_dim.to_u32, 4)
          out_enc.set_value(out_qw.out_dim.to_u32, 5)
          out_enc.set_value(rows.to_u32, 6)
          out_enc.set_value(0_u32, 7)
          out_enc.set_threadgroup_memory(MM_SHMEM, 0)
          out_enc.dispatch_threadgroups({
            (rows + MM_NR1 - 1) // MM_NR1,
            (out_qw.out_dim + MM_NR0 - 1) // MM_NR0,
            1,
          }, {MM_TG, 1, 1})
          out_enc.end_encoding

          reduce_enc = ML::Metal::ComputeEncoder.new(cmd)
          reduce_enc.set_pipeline(top1_reduce_f16_rows_pipeline)
          reduce_enc.set_buffer(logits16_buf, 0)
          reduce_enc.set_buffer(top1_id_buf, 1, ML::Metal::BufferAccess::Write)
          reduce_enc.set_buffer(top1_value_buf, 2, ML::Metal::BufferAccess::Write)
          reduce_enc.set_value(out_qw.out_dim.to_u32, 3)
          reduce_enc.dispatch_threadgroups({rows, 1, 1}, {256, 1, 1})
          reduce_enc.end_encoding

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_top1_rows(top1_id_buf, top1_value_buf, rows)

          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_gemv(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end

          result
        end

        def self.rmsnorm_project_top1_rows_buffer(x_buf : ML::MetalBuffer,
                                                  rows : Int32,
                                                  norm_weight : Array(Float32),
                                                  out_qw : QuantWeight,
                                                  eps : Float32) : Array({Int32, Float32})?
          return nil unless head_top1_fused_enabled?
          return nil unless out_qw.type.q6_k?
          return nil unless out_qw.in_dim % QK_K == 0
          return nil unless rows > 0
          hidden_dim = out_qw.in_dim
          hidden_bytes = (rows * hidden_dim).to_i64 * sizeof(Float32)
          return nil if x_buf.size < hidden_bytes

          ML::Metal::Device.init!

          tile_count = (out_qw.out_dim + HEAD_TOP1_ROWS_PER_TG - 1) // HEAD_TOP1_ROWS_PER_TG
          norm_w_buf = Scratch.get(:head_top1_rows_buf_norm_w, norm_weight.size.to_i64 * sizeof(Float32))
          normed_buf = Scratch.get(:head_top1_rows_buf_normed, hidden_bytes)
          tile_values_buf = Scratch.get(:head_top1_rows_buf_tile_values, (rows * tile_count).to_i64 * sizeof(Float32))
          tile_ids_buf = Scratch.get(:head_top1_rows_buf_tile_ids, (rows * tile_count).to_i64 * sizeof(UInt32))
          top1_id_buf = Scratch.get(:head_top1_rows_buf_id, rows.to_i64 * sizeof(UInt32))
          top1_value_buf = Scratch.get(:head_top1_rows_buf_value, rows.to_i64 * sizeof(Float32))
          norm_w_buf.write(norm_weight)

          out_w_buf, out_w_off = weight_slot(out_qw)

          t0 = Time.instant if Profile.enabled?
          cmd = ML::Metal::CommandBuffer.new

          norm_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_rmsnorm_rows(norm_enc, x_buf, norm_w_buf, normed_buf, hidden_dim, rows, eps)
          norm_enc.end_encoding

          head_top1_enc = ML::Metal::ComputeEncoder.new(cmd)
          head_top1_enc.set_pipeline(mv6_top1_tiles_batch_pipeline)
          head_top1_enc.set_buffer(out_w_buf, 0, ML::Metal::BufferAccess::Read, offset: out_w_off)
          head_top1_enc.set_buffer(normed_buf, 1)
          head_top1_enc.set_buffer(tile_values_buf, 2, ML::Metal::BufferAccess::Write)
          head_top1_enc.set_buffer(tile_ids_buf, 3, ML::Metal::BufferAccess::Write)
          head_top1_enc.set_value(out_qw.in_dim.to_u32, 4)
          head_top1_enc.set_value(out_qw.out_dim.to_u32, 5)
          head_top1_enc.set_value(tile_count.to_u32, 6)
          head_top1_enc.dispatch_threadgroups({tile_count, rows, 1}, {64, 1, 1})
          head_top1_enc.end_encoding

          reduce_top1_enc = ML::Metal::ComputeEncoder.new(cmd)
          reduce_top1_enc.set_pipeline(top1_reduce_tiles_batch_pipeline)
          reduce_top1_enc.set_buffer(tile_values_buf, 0)
          reduce_top1_enc.set_buffer(tile_ids_buf, 1)
          reduce_top1_enc.set_buffer(top1_id_buf, 2, ML::Metal::BufferAccess::Write)
          reduce_top1_enc.set_buffer(top1_value_buf, 3, ML::Metal::BufferAccess::Write)
          reduce_top1_enc.set_value(tile_count.to_u32, 4)
          reduce_top1_enc.dispatch_threadgroups({rows, 1, 1}, {256, 1, 1})
          reduce_top1_enc.end_encoding

          t_enc = Time.instant if Profile.enabled?
          cmd.commit
          cmd.wait
          t_wait = Time.instant if Profile.enabled?
          result = read_shared_top1_rows(top1_id_buf, top1_value_buf, rows)
          if Profile.enabled?
            t_read = Time.instant
            Profile.bump_gemv(
              (t_enc.not_nil! - t0.not_nil!).total_nanoseconds.to_i64,
              (t_wait.not_nil! - t_enc.not_nil!).total_nanoseconds.to_i64,
              (t_read - t_wait.not_nil!).total_nanoseconds.to_i64,
            )
          end
          result
        end

        def self.bench_head_top1_rows_wait_ms(out_qw : QuantWeight,
                                              x : Array(Float32),
                                              norm_weight : Array(Float32),
                                              eps : Float32,
                                              rows_per_tg : Int32,
                                              validate : Bool = false) : Float64
          raise ArgumentError.new("head top1 rows bench requires Q6_K output weight") unless out_qw.type.q6_k?
          raise ArgumentError.new("head top1 rows bench input mismatch") unless x.size == out_qw.in_dim
          raise ArgumentError.new("head top1 rows bench norm mismatch") unless norm_weight.size == out_qw.in_dim

          ML::Metal::Device.init!
          hidden_dim = x.size
          tile_count = (out_qw.out_dim + rows_per_tg - 1) // rows_per_tg
          x_buf = Scratch.get(:bench_head_top1_x, hidden_dim.to_i64 * sizeof(Float32))
          norm_w_buf = Scratch.get(:bench_head_top1_norm_w, norm_weight.size.to_i64 * sizeof(Float32))
          normed_buf = Scratch.get(:bench_head_top1_normed, hidden_dim.to_i64 * sizeof(Float32))
          tile_values_buf = Scratch.get(:bench_head_top1_tile_values, tile_count.to_i64 * sizeof(Float32))
          tile_ids_buf = Scratch.get(:bench_head_top1_tile_ids, tile_count.to_i64 * sizeof(UInt32))
          top1_id_buf = Scratch.get(:bench_head_top1_id, sizeof(UInt32).to_i64)
          top1_value_buf = Scratch.get(:bench_head_top1_value, sizeof(Float32).to_i64)
          x_buf.write(x)
          norm_w_buf.write(norm_weight)
          out_w_buf, out_w_off = weight_slot(out_qw)

          cmd = ML::Metal::CommandBuffer.new
          norm_enc = ML::Metal::ComputeEncoder.new(cmd)
          encode_rmsnorm_vec(norm_enc, x_buf, norm_w_buf, normed_buf, hidden_dim, eps)
          norm_enc.end_encoding

          head_top1_enc = ML::Metal::ComputeEncoder.new(cmd)
          head_top1_enc.set_pipeline(mv6_top1_tiles_rows_pipeline(rows_per_tg))
          head_top1_enc.set_buffer(out_w_buf, 0, ML::Metal::BufferAccess::Read, offset: out_w_off)
          head_top1_enc.set_buffer(normed_buf, 1)
          head_top1_enc.set_buffer(tile_values_buf, 2, ML::Metal::BufferAccess::Write)
          head_top1_enc.set_buffer(tile_ids_buf, 3, ML::Metal::BufferAccess::Write)
          head_top1_enc.set_value(out_qw.in_dim.to_u32, 4)
          head_top1_enc.set_value(out_qw.out_dim.to_u32, 5)
          head_top1_enc.dispatch_threadgroups({tile_count, 1, 1}, {MV_Q6_NSG * 32, 1, 1})
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

          t0 = Time.instant
          cmd.commit
          cmd.wait
          elapsed = (Time.instant - t0).total_milliseconds

          if validate
            candidate = read_shared_top1(top1_id_buf, top1_value_buf)
            baseline = rmsnorm_project_top1(x, norm_weight, out_qw, eps)
            raise "default head top1 baseline returned nil" if baseline.nil?
            id_match = candidate[0].to_i == baseline.not_nil![0].to_i
            value_diff = (candidate[1] - baseline.not_nil![1]).abs
            raise "head top1 rows=#{rows_per_tg} mismatch id=#{candidate[0].to_i}/#{baseline.not_nil![0].to_i} diff=#{value_diff}" unless id_match && value_diff <= 1.0e-3_f32
          end

          elapsed
        end
      {% end %}
    end
  end
end
