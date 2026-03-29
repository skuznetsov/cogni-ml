# Metal GPU compute backend for NomicBertMoE inference.
# Fused dequant-matmul kernels for Q5_K and Q6_K.

{% unless flag?(:cpu_only) %}

require "./compute"

module ML::GGUF
  BERT_METAL_SOURCE = {{ read_file("#{__DIR__}/kernels/bert_embed.metal") }}

  struct MetalBackend
    include ComputeBackend

    def initialize
      raise "Metal not available" unless ML::Metal::Device.available?
      %w[matmul_dequant_q5k matmul_dequant_q6k].each do |name|
        ML::Metal::PipelineCache.get(name) { ML::Metal::ComputePipeline.new(name, BERT_METAL_SOURCE) }
      end
    end

    def matmul(x : Array(Float32), rows : Int32, qw : QuantWeight, bias : Array(Float32)) : Array(Float32)
      kernel_name = case qw.type
                    when .q5_k? then "matmul_dequant_q5k"
                    when .q6_k? then "matmul_dequant_q6k"
                    else return F32Backend.new.matmul(x, rows, qw, bias)
                    end

      out_dim = qw.out_dim
      in_dim = qw.in_dim
      result = Array(Float32).new(rows * out_dim, 0.0_f32)

      # GPU buffers — weight + bias are reusable across rows
      w_buf = ML::MetalBuffer.new(qw.raw.size.to_i64)
      w_buf.write_bytes(qw.raw.to_unsafe, qw.raw.size)

      b_buf = ML::MetalBuffer.new(bias.size.to_i64 * 4)
      b_buf.write(bias)

      pipeline = ML::Metal::PipelineCache.get(kernel_name) { raise "not compiled" }

      rows.times do |r|
        x_buf = ML::MetalBuffer.new(in_dim.to_i64 * 4)
        x_buf.write_slice(Slice(Float32).new(x.to_unsafe + r * in_dim, in_dim))

        o_buf = ML::MetalBuffer.new(out_dim.to_i64 * 4)

        cmd = ML::Metal::CommandBuffer.new
        enc = ML::Metal::ComputeEncoder.new(cmd)
        enc.set_pipeline(pipeline)
        enc.set_buffer(w_buf, 0)
        enc.set_buffer(x_buf, 1)
        enc.set_buffer(b_buf, 2)
        enc.set_buffer(o_buf, 3)
        enc.set_value(in_dim.to_u32, 4)
        enc.set_value(out_dim.to_u32, 5)
        enc.dispatch_1d(out_dim, 256)
        enc.end_encoding
        cmd.commit_and_wait

        row_out = o_buf.read(out_dim)
        out_dim.times { |j| result[r * out_dim + j] = row_out[j] }
      end

      result
    end

    def layer_norm!(x : Array(Float32), n_pos : Int32, dim : Int32, w : Array(Float32), b : Array(Float32)) : Nil
      F32Backend.new.layer_norm!(x, n_pos, dim, w, b)
    end

    def softmax_row!(scores : Array(Float32), offset : Int32, len : Int32) : Nil
      F32Backend.new.softmax_row!(scores, offset, len)
    end

    def gelu(x : Float32) : Float32
      F32Backend.new.gelu(x)
    end

    def dot(a : Array(Float32), a_off : Int32, b : Array(Float32), b_off : Int32, len : Int32) : Float32
      F32Backend.new.dot(a, a_off, b, b_off, len)
    end
  end
end

{% end %}
