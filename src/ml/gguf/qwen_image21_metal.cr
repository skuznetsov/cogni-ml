require "./qwen_image21_block"
require "./qwen35_metal"

# Hybrid reference backend for Qwen-Image 2.1 block admission.
#
# Projection matmuls use the established Qwen 3.5 Metal K-quant kernels while
# normalization, RoPE, attention, and elementwise math remain on the CPU. This
# deliberately narrow boundary supports exact one-block CPU/Metal parity before
# the full block is fused or made resident on-device.
module ML::GGUF
  class QwenImage21MetalProjectionBackend
    include ComputeBackend

    getter metal_projection_count = 0

    def initialize(@strict : Bool = true)
      @cpu = F32Backend.new
    end

    def self.available? : Bool
      Qwen35Metal.available?
    end

    def matmul(x : Array(Float32), rows : Int32, qw : QuantWeight,
               bias : Array(Float32)) : Array(Float32)
      unless bias.size == qw.out_dim
        raise ArgumentError.new("bias size #{bias.size} does not match projection output #{qw.out_dim}")
      end

      if result = Qwen35Metal.matmul(qw, x, rows)
        @metal_projection_count += 1
        add_bias!(result, rows, qw.out_dim, bias)
        result
      elsif @strict
        raise ArgumentError.new(
          "no Qwen-Image Metal projection route for #{qw.type.name} #{qw.in_dim}x#{qw.out_dim} batch=#{rows}"
        )
      else
        @cpu.matmul(x, rows, qw, bias)
      end
    end

    def layer_norm!(x : Array(Float32), n_pos : Int32, dim : Int32,
                    w : Array(Float32), b : Array(Float32)) : Nil
      @cpu.layer_norm!(x, n_pos, dim, w, b)
    end

    def softmax_row!(scores : Array(Float32), offset : Int32, len : Int32) : Nil
      @cpu.softmax_row!(scores, offset, len)
    end

    def gelu(x : Float32) : Float32
      @cpu.gelu(x)
    end

    def dot(a : Array(Float32), a_off : Int32, b : Array(Float32),
            b_off : Int32, len : Int32) : Float32
      @cpu.dot(a, a_off, b, b_off, len)
    end

    private def add_bias!(output : Array(Float32), rows : Int32, out_dim : Int32,
                          bias : Array(Float32)) : Nil
      return if bias.all?(&.zero?)
      rows.times do |row|
        offset = row * out_dim
        out_dim.times { |column| output[offset + column] += bias[column] }
      end
    end
  end
end
