# Compute backends for BERT inference — parameterize precision and device.
#
# F32Backend:     CPU Float32 accumulation (current default)
# F16SimBackend:  CPU with FP16 truncation at key points (matches GPU precision)
# MetalBackend:   Metal GPU (future)

require "./quant_matmul"
require "./dequant"
require "./reader"  # for TensorType

module ML::GGUF
  # Quantized weight: raw bytes + type for on-the-fly dequant during matmul
  struct QuantWeight
    getter raw : Bytes
    getter type : TensorType
    getter out_dim : Int32
    getter in_dim : Int32

    def initialize(@raw, @type, @out_dim, @in_dim)
    end
  end
  # Abstract compute backend. All math ops go through this interface.
  module ComputeBackend
    # Matrix multiply: x[rows, in_dim] × W_quant + bias → [rows, out_dim]
    abstract def matmul(x : Array(Float32), rows : Int32, qw : QuantWeight, bias : Array(Float32)) : Array(Float32)

    # In-place layer norm
    abstract def layer_norm!(x : Array(Float32), n_pos : Int32, dim : Int32, w : Array(Float32), b : Array(Float32)) : Nil

    # In-place softmax over a row
    abstract def softmax_row!(scores : Array(Float32), offset : Int32, len : Int32) : Nil

    # GELU activation (returns new value)
    abstract def gelu(x : Float32) : Float32

    # Dot product for attention scores
    abstract def dot(a : Array(Float32), a_off : Int32, b : Array(Float32), b_off : Int32, len : Int32) : Float32
  end

  # CPU Float32 backend — full precision, current default
  struct F32Backend
    include ComputeBackend

    def matmul(x : Array(Float32), rows : Int32, qw : QuantWeight, bias : Array(Float32)) : Array(Float32)
      QuantMatmul.matmul_add(x, rows, qw.in_dim, qw.raw, qw.type, qw.out_dim, bias)
    end

    def layer_norm!(x : Array(Float32), n_pos : Int32, dim : Int32, w : Array(Float32), b : Array(Float32)) : Nil
      eps = 1e-5_f32
      n_pos.times do |pos|
        off = pos * dim
        mean = 0.0_f32
        dim.times { |j| mean += x[off + j] }
        mean /= dim
        var = 0.0_f32
        dim.times { |j| d = x[off + j] - mean; var += d * d }
        var /= dim
        inv_std = 1.0_f32 / Math.sqrt(var + eps)
        dim.times { |j| x[off + j] = (x[off + j] - mean) * inv_std * w[j] + b[j] }
      end
    end

    def softmax_row!(scores : Array(Float32), offset : Int32, len : Int32) : Nil
      max_val = -Float32::MAX
      len.times { |i| max_val = Math.max(max_val, scores[offset + i]) }
      sum = 0.0_f32
      len.times do |i|
        scores[offset + i] = Math.exp(scores[offset + i] - max_val)
        sum += scores[offset + i]
      end
      inv_sum = 1.0_f32 / sum
      len.times { |i| scores[offset + i] *= inv_sum }
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

  # FP16 simulation backend — truncates intermediates to FP16 precision.
  # Matches GPU Metal behavior where computations happen in half precision.
  struct F16SimBackend
    include ComputeBackend

    # FP16 round-trip: F32 → F16 → F32 (truncates mantissa to 10 bits)
    @[AlwaysInline]
    private def fp16(v : Float32) : Float32
      # Pack to IEEE 754 half, unpack back
      bits = v.unsafe_as(UInt32)
      sign = (bits >> 16) & 0x8000_u32
      exp = ((bits >> 23) & 0xFF).to_i32 - 127 + 15
      mant = (bits >> 13) & 0x03FF_u32

      h = if exp <= 0
            0_u16  # Flush subnormals to zero
          elsif exp >= 31
            (sign | 0x7C00).to_u16  # Inf
          else
            (sign | (exp.to_u32 << 10) | mant).to_u16
          end
      Dequant.fp16_to_f32(h)
    end

    def matmul(x : Array(Float32), rows : Int32, qw : QuantWeight, bias : Array(Float32)) : Array(Float32)
      # Use fused quant matmul but truncate the final result per-element
      result = QuantMatmul.matmul_add(x, rows, qw.in_dim, qw.raw, qw.type, qw.out_dim, bias)
      result.map! { |v| fp16(v) }
      result
    end

    def layer_norm!(x : Array(Float32), n_pos : Int32, dim : Int32, w : Array(Float32), b : Array(Float32)) : Nil
      eps = 1e-5_f32
      n_pos.times do |pos|
        off = pos * dim
        # Accumulate in FP16
        mean = 0.0_f32
        dim.times { |j| mean = fp16(mean + fp16(x[off + j])) }
        mean = fp16(mean / dim)
        var = 0.0_f32
        dim.times { |j| d = fp16(x[off + j] - mean); var = fp16(var + fp16(d * d)) }
        var = fp16(var / dim)
        inv_std = fp16(1.0_f32 / Math.sqrt(fp16(var + eps)))
        dim.times { |j| x[off + j] = fp16(fp16((fp16(x[off + j]) - mean) * inv_std) * fp16(w[j]) + fp16(b[j])) }
      end
    end

    def softmax_row!(scores : Array(Float32), offset : Int32, len : Int32) : Nil
      max_val = -Float32::MAX
      len.times { |i| max_val = Math.max(max_val, scores[offset + i]) }
      sum = 0.0_f32
      len.times do |i|
        scores[offset + i] = fp16(Math.exp(fp16(scores[offset + i] - max_val)))
        sum = fp16(sum + scores[offset + i])
      end
      inv_sum = fp16(1.0_f32 / sum)
      len.times { |i| scores[offset + i] = fp16(scores[offset + i] * inv_sum) }
    end

    def gelu(x : Float32) : Float32
      xf = fp16(x)
      fp16(0.5_f32 * xf * fp16(1.0_f32 + fp16(Math.tanh(fp16(0.7978845608_f32 * fp16(xf + fp16(0.044715_f32 * fp16(xf * fp16(xf * xf)))))))))
    end

    def dot(a : Array(Float32), a_off : Int32, b : Array(Float32), b_off : Int32, len : Int32) : Float32
      sum = 0.0_f32
      len.times { |i| sum = fp16(sum + fp16(fp16(a[a_off + i]) * fp16(b[b_off + i]))) }
      sum
    end
  end
end
