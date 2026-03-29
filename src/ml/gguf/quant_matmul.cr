# Fused quantized matrix-vector multiply.
# Dequantizes on-the-fly during dot product — preserves full F32 precision
# in the accumulator while reading quantized weights directly from memory.
#
# This matches llama.cpp's approach and gives higher precision than
# bulk dequant→F32→matmul because intermediate values stay in registers.
#
# Supports: Q5_K, Q6_K, F32, F16

require "./reader"  # for TensorType

module ML::GGUF
  module QuantMatmul
    QK_K = 256

    # Fused matmul: result[o] = Σ_j x[j] * dequant(W_raw[o, j]) + bias[o]
    # W_raw is quantized weight data as raw bytes, row-major [out_dim rows, in_dim cols].
    # Each row is a sequence of quantized blocks covering in_dim elements.
    def self.matmul_add(
      x : Array(Float32), rows : Int32, in_dim : Int32,
      w_raw : Bytes, w_type : TensorType, out_dim : Int32,
      bias : Array(Float32),
    ) : Array(Float32)
      case w_type
      when .q5_k? then matmul_add_q5k(x, rows, in_dim, w_raw, out_dim, bias)
      when .q6_k? then matmul_add_q6k(x, rows, in_dim, w_raw, out_dim, bias)
      when .f32?  then matmul_add_f32(x, rows, in_dim, w_raw, out_dim, bias)
      when .f16?  then matmul_add_f16(x, rows, in_dim, w_raw, out_dim, bias)
      else
        raise "Unsupported quant type for fused matmul: #{w_type.name}"
      end
    end

    # Q5_K fused matmul: for each output neuron, walk through Q5_K blocks
    # and accumulate dot product with dequantized values.
    private def self.matmul_add_q5k(
      x : Array(Float32), rows : Int32, in_dim : Int32,
      w_raw : Bytes, out_dim : Int32, bias : Array(Float32),
    ) : Array(Float32)
      block_size = 176  # bytes per Q5_K block
      blocks_per_row = in_dim // QK_K
      row_bytes = blocks_per_row * block_size
      result = Array(Float32).new(rows * out_dim, 0.0_f32)
      w_ptr = w_raw.to_unsafe

      rows.times do |r|
        x_off = r * in_dim
        r_off = r * out_dim

        out_dim.times do |o|
          sum = bias[o].to_f64
          w_row = w_ptr + o * row_bytes

          blocks_per_row.times do |blk|
            blk_ptr = w_row + blk * block_size
            d = Dequant.fp16_to_f32(Bytes.new(blk_ptr, 2)).to_f64
            dmin = Dequant.fp16_to_f32(Bytes.new(blk_ptr + 2, 2)).to_f64
            scales_ptr = blk_ptr + 4
            qh_ptr = blk_ptr + 16
            ql_ptr = blk_ptr + 48

            base_j = blk * QK_K
            u1 = 1_u8
            u2 = 2_u8
            is = 0
            ql_off = 0

            4.times do
              sc, m = Dequant.get_scale_min_k4(is, scales_ptr)
              d1 = d * sc; m1 = dmin * m
              sc2, m2 = Dequant.get_scale_min_k4(is + 1, scales_ptr)
              d2 = d * sc2; m2_val = dmin * m2

              32.times do |l|
                j = base_j + (is // 2) * 64 + l
                q_low = ql_ptr[ql_off + l] & 0x0F
                q_high = (qh_ptr[l] & u1) != 0 ? 16 : 0
                val = d1 * (q_low + q_high) - m1
                sum += x[x_off + j] * val
              end
              32.times do |l|
                j = base_j + (is // 2) * 64 + 32 + l
                q_low = (ql_ptr[ql_off + l] >> 4) & 0x0F
                q_high = (qh_ptr[l] & u2) != 0 ? 16 : 0
                val = d2 * (q_low + q_high) - m2_val
                sum += x[x_off + j] * val
              end

              ql_off += 32
              is += 2
              u1 = u1 << 2
              u2 = u2 << 2
            end
          end

          result[r_off + o] = sum.to_f32
        end
      end

      result
    end

    # Q6_K fused matmul
    private def self.matmul_add_q6k(
      x : Array(Float32), rows : Int32, in_dim : Int32,
      w_raw : Bytes, out_dim : Int32, bias : Array(Float32),
    ) : Array(Float32)
      block_size = 210
      blocks_per_row = in_dim // QK_K
      row_bytes = blocks_per_row * block_size
      result = Array(Float32).new(rows * out_dim, 0.0_f32)
      w_ptr = w_raw.to_unsafe

      rows.times do |r|
        x_off = r * in_dim
        r_off = r * out_dim

        out_dim.times do |o|
          sum = bias[o].to_f64
          w_row = w_ptr + o * row_bytes

          blocks_per_row.times do |blk|
            blk_ptr = w_row + blk * block_size
            ql_ptr = blk_ptr
            qh_ptr = blk_ptr + 128
            sc_ptr = blk_ptr + 192
            d = Dequant.fp16_to_f32(Bytes.new(blk_ptr + 208, 2)).to_f64

            base_j = blk * QK_K
            ql_off = 0
            qh_off = 0
            sc_off = 0

            2.times do |n_iter|
              32.times do |l|
                is = l // 16
                q1 = ((ql_ptr[ql_off + l].to_i32 & 0xF) | (((qh_ptr[qh_off + l].to_i32 >> 0) & 3) << 4)) - 32
                q2 = ((ql_ptr[ql_off + l + 32].to_i32 & 0xF) | (((qh_ptr[qh_off + l].to_i32 >> 2) & 3) << 4)) - 32
                q3 = ((ql_ptr[ql_off + l].to_i32 >> 4) | (((qh_ptr[qh_off + l].to_i32 >> 4) & 3) << 4)) - 32
                q4 = ((ql_ptr[ql_off + l + 32].to_i32 >> 4) | (((qh_ptr[qh_off + l].to_i32 >> 6) & 3) << 4)) - 32

                s0 = sc_ptr[sc_off + is].unsafe_as(Int8).to_f64
                s2 = sc_ptr[sc_off + is + 2].unsafe_as(Int8).to_f64
                s4 = sc_ptr[sc_off + is + 4].unsafe_as(Int8).to_f64
                s6 = sc_ptr[sc_off + is + 6].unsafe_as(Int8).to_f64

                j_base = base_j + n_iter * 128
                sum += x[x_off + j_base + l] * (d * s0 * q1)
                sum += x[x_off + j_base + l + 32] * (d * s2 * q2)
                sum += x[x_off + j_base + l + 64] * (d * s4 * q3)
                sum += x[x_off + j_base + l + 96] * (d * s6 * q4)
              end
              ql_off += 64
              qh_off += 32
              sc_off += 8
            end
          end

          result[r_off + o] = sum.to_f32
        end
      end

      result
    end

    # F32 fused matmul (reference — same as regular matmul)
    private def self.matmul_add_f32(
      x : Array(Float32), rows : Int32, in_dim : Int32,
      w_raw : Bytes, out_dim : Int32, bias : Array(Float32),
    ) : Array(Float32)
      result = Array(Float32).new(rows * out_dim, 0.0_f32)
      w_f32 = w_raw.to_unsafe.as(Pointer(Float32))

      rows.times do |r|
        x_off = r * in_dim
        r_off = r * out_dim
        out_dim.times do |o|
          sum = bias[o].to_f64
          w_off = o * in_dim
          in_dim.times { |j| sum += x[x_off + j] * w_f32[w_off + j] }
          result[r_off + o] = sum.to_f32
        end
      end
      result
    end

    # F16 fused matmul
    private def self.matmul_add_f16(
      x : Array(Float32), rows : Int32, in_dim : Int32,
      w_raw : Bytes, out_dim : Int32, bias : Array(Float32),
    ) : Array(Float32)
      result = Array(Float32).new(rows * out_dim, 0.0_f32)
      w_ptr = w_raw.to_unsafe

      rows.times do |r|
        x_off = r * in_dim
        r_off = r * out_dim
        out_dim.times do |o|
          sum = bias[o].to_f64
          w_off = (o * in_dim) * 2  # 2 bytes per f16
          in_dim.times do |j|
            val = Dequant.fp16_to_f32(Bytes.new(w_ptr + w_off + j * 2, 2))
            sum += x[x_off + j] * val
          end
          result[r_off + o] = sum.to_f32
        end
      end
      result
    end
  end
end
