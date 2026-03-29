# Metal GPU compute backend for NomicBertMoE — zero-copy unified memory.
#
# All buffers use StorageMode::Shared (unified memory on Apple Silicon).
# Weight buffers pre-uploaded at model load. Scratch buffers reused.
# No allocations during inference.

{% unless flag?(:cpu_only) %}

require "./compute"

module ML::GGUF
  BERT_METAL_SOURCE = {{ read_file("#{__DIR__}/kernels/bert_embed.metal") }}

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

  class MetalBackend
    include ComputeBackend

    @gpu_weights : Hash(UInt64, GPUWeight)
    # Preallocated scratch buffers (reused every matmul, no allocation during inference)
    @scratch_x : ML::MetalBuffer    # Input scratch
    @scratch_o : ML::MetalBuffer    # Output scratch
    @scratch_x_cap : Int32           # Current capacity in floats
    @scratch_o_cap : Int32

    # Max dims for nomic-bert-moe: in=3072 (FFN), out=3072 (FFN), seq=512
    INITIAL_SCRATCH = 512 * 3072  # ~6MB — fits all matmul sizes

    def initialize
      raise "Metal not available" unless ML::Metal::Device.available?
      %w[matmul_dequant_q5k matmul_dequant_q6k].each do |name|
        ML::Metal::PipelineCache.get(name) { ML::Metal::ComputePipeline.new(name, BERT_METAL_SOURCE) }
      end
      @gpu_weights = {} of UInt64 => GPUWeight
      @scratch_x_cap = INITIAL_SCRATCH
      @scratch_o_cap = INITIAL_SCRATCH
      @scratch_x = ML::MetalBuffer.new(@scratch_x_cap.to_i64 * 4)
      @scratch_o = ML::MetalBuffer.new(@scratch_o_cap.to_i64 * 4)
    end

    def upload_weight(qw : QuantWeight, bias : Array(Float32)) : Nil
      key = qw.raw.to_unsafe.address
      @gpu_weights[key] ||= GPUWeight.new(qw, bias)
    end

    private def get_or_upload(qw : QuantWeight, bias : Array(Float32)) : GPUWeight
      key = qw.raw.to_unsafe.address
      @gpu_weights[key] ||= GPUWeight.new(qw, bias)
    end

    private def ensure_scratch(need_x : Int32, need_o : Int32)
      if need_x > @scratch_x_cap
        @scratch_x_cap = need_x
        @scratch_x = ML::MetalBuffer.new(need_x.to_i64 * 4)
      end
      if need_o > @scratch_o_cap
        @scratch_o_cap = need_o
        @scratch_o = ML::MetalBuffer.new(need_o.to_i64 * 4)
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
      gw = get_or_upload(qw, bias)
      pipeline = ML::Metal::PipelineCache.get(kernel_name) { raise "not compiled" }

      ensure_scratch(in_dim, rows * out_dim)
      result = Array(Float32).new(rows * out_dim, 0.0_f32)

      rows.times do |r|
        # Write input row to scratch via unified memory pointer (zero copy)
        x_ptr = @scratch_x.contents.as(Pointer(Float32))
        in_dim.times { |j| x_ptr[j] = x[r * in_dim + j] }

        # Dispatch
        cmd = ML::Metal::CommandBuffer.new
        enc = ML::Metal::ComputeEncoder.new(cmd)
        enc.set_pipeline(pipeline)
        enc.set_buffer(gw.buffer, 0)
        enc.set_buffer(@scratch_x, 1)
        enc.set_buffer(gw.bias_buffer, 2)
        enc.set_buffer(@scratch_o, 3)
        enc.set_value(in_dim.to_u32, 4)
        enc.set_value(out_dim.to_u32, 5)
        enc.dispatch_1d(out_dim, 256)
        enc.end_encoding
        cmd.commit_and_wait

        # Read output via unified memory pointer (zero copy)
        o_ptr = @scratch_o.contents.as(Pointer(Float32))
        out_dim.times { |j| result[r * out_dim + j] = o_ptr[j] }
      end

      result
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
end

{% end %}
