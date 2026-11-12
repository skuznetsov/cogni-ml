# Small FP32 LoRA projection primitive for the bounded Qwen3.5 final-block
# experiment. The GGUF weight remains quantized and is never modified.

require "./qwen35_cpu"

module ML::GGUF::Qwen35CPU
  # Public adapter boundary around the native routed projection helpers. Keep
  # single-token decode on qmatvec and batches on the existing routed GEMM.
  def self.qlora_project(qw : QuantWeight, x : Array(Float32), rows : Int32) : Array(Float32)
    raise ArgumentError.new("qlora projection rows must be positive") unless rows > 0
    raise ArgumentError.new("qlora projection dimensions must be positive") unless qw.in_dim > 0 && qw.out_dim > 0
    expected_input = rows.to_i64 * qw.in_dim.to_i64
    raise ArgumentError.new("qlora projection input size mismatch: expected #{expected_input}, got #{x.size}") unless x.size.to_i64 == expected_input

    if rows == 1
      qmatvec_nobias(qw, x)
    else
      {% if flag?(:cpu_only) %}
        zero = Array(Float32).new(qw.out_dim, 0.0_f32)
        QuantMatmul.matmul_add(x, rows, qw.in_dim, qw.raw, qw.type, qw.out_dim, zero)
      {% else %}
        if Qwen35Metal.available? && qw.type.q5_k?
          # The generic Q5_K/Q6_K batch route selects H16-output GEMM kernels.
          # Use the existing F32-output entry points for this trainable block.
          Qwen35Metal.matmul_q5k(x, qw.raw, qw.in_dim, qw.out_dim, rows)
        elsif Qwen35Metal.available? && qw.type.q6_k?
          Qwen35Metal.matmul_q6k(x, qw.raw, qw.in_dim, qw.out_dim, rows)
        else
          qmatmul_nobias(qw, x, rows)
        end
      {% end %}
    end
  end
end

module ML::GGUF::Qwen35QLoRA
  module MathOps
    {% if flag?(:darwin) %}
      @[Link(framework: "Accelerate")]
      lib LibQwen35QLoRAAccelerate
        fun cblas_sgemm(layout : Int32, trans_a : Int32, trans_b : Int32,
                        m : Int32, n : Int32, k : Int32,
                        alpha : Float32, a : Float32*, lda : Int32,
                        b : Float32*, ldb : Int32,
                        beta : Float32, c : Float32*, ldc : Int32) : Void
      end
    {% end %}

    # Row-major matrix product. Physical A and B arrays are contiguous; the
    # transpose flags change their logical shapes without copying.
    def self.gemm(a : Array(Float32), b : Array(Float32),
                  m : Int32, n : Int32, k : Int32,
                  transpose_a : Bool = false, transpose_b : Bool = false) : Array(Float32)
      raise ArgumentError.new("gemm dimensions must be positive") unless m > 0 && n > 0 && k > 0

      expected_a = (m.to_i64 * k.to_i64)
      expected_b = (k.to_i64 * n.to_i64)
      output_size = (m.to_i64 * n.to_i64)
      raise ArgumentError.new("gemm A size mismatch: expected #{expected_a}, got #{a.size}") unless a.size.to_i64 == expected_a
      raise ArgumentError.new("gemm B size mismatch: expected #{expected_b}, got #{b.size}") unless b.size.to_i64 == expected_b
      raise ArgumentError.new("gemm output is too large") if output_size > Int32::MAX

      output = Array(Float32).new(output_size.to_i32, 0.0_f32)

      {% if flag?(:darwin) %}
        # CBLAS row-major enums: NoTrans=111, Trans=112. lda/ldb describe the
        # physical row stride before applying each transpose.
        LibQwen35QLoRAAccelerate.cblas_sgemm(
          101_i32,
          transpose_a ? 112_i32 : 111_i32,
          transpose_b ? 112_i32 : 111_i32,
          m, n, k,
          1.0_f32, a.to_unsafe, transpose_a ? m : k,
          b.to_unsafe, transpose_b ? k : n,
          0.0_f32, output.to_unsafe, n,
        )
      {% else %}
        # Portable reference fallback for non-macOS builds. The experiment's
        # macOS path uses Accelerate; this fallback preserves correctness.
        m.times do |row|
          n.times do |column|
            sum = 0.0_f32
            k.times do |inner|
              a_index = transpose_a ? inner * m + row : row * k + inner
              b_index = transpose_b ? column * k + inner : inner * n + column
              sum += a[a_index] * b[b_index]
            end
            output[row * n + column] = sum
          end
        end
      {% end %}

      output
    end
  end

  class Projection
    getter a : Array(Float32)
    getter b : Array(Float32)
    getter grad_a : Array(Float32)
    getter grad_b : Array(Float32)
    getter rank : Int32
    getter alpha : Float32
    getter input_dim : Int32
    getter output_dim : Int32

    @weight : QuantWeight
    @scale : Float32
    @dequantized_weight : Array(Float32)? = nil

    def initialize(@weight : QuantWeight, random : Random,
                   @rank : Int32 = 8, @alpha : Float32 = 16_f32)
      raise ArgumentError.new("projection dimensions must be positive") unless @weight.in_dim > 0 && @weight.out_dim > 0
      raise ArgumentError.new("rank must be positive") unless @rank > 0
      raise ArgumentError.new("projection weight is too large") if @weight.in_dim.to_i64 * @weight.out_dim.to_i64 > Int32::MAX
      raise ArgumentError.new("LoRA A is too large") if @weight.in_dim.to_i64 * @rank.to_i64 > Int32::MAX
      raise ArgumentError.new("LoRA B is too large") if @weight.out_dim.to_i64 * @rank.to_i64 > Int32::MAX

      @input_dim = @weight.in_dim
      @output_dim = @weight.out_dim
      @scale = @alpha / @rank.to_f32
      bound = Math.sqrt(1.0_f64 / @input_dim.to_f64).to_f32

      # Kaiming-uniform A and zero B preserve the frozen model output at
      # initialization while allowing B to receive a first-step gradient.
      @a = Array(Float32).new(@rank * @input_dim) do
        (random.rand.to_f32 * 2.0_f32 - 1.0_f32) * bound
      end
      @b = Array(Float32).new(@output_dim * @rank, 0.0_f32)
      @grad_a = Array(Float32).new(@a.size, 0.0_f32)
      @grad_b = Array(Float32).new(@b.size, 0.0_f32)
    end

    def zero_grad : Nil
      @grad_a.fill(0.0_f32)
      @grad_b.fill(0.0_f32)
    end

    # x is row-major [rows,input_dim]; output is [rows,output_dim].
    def forward(x : Array(Float32), rows : Int32) : Array(Float32)
      validate_rows_and_input(x, rows)

      output = Qwen35CPU.qlora_project(@weight, x, rows)
      hidden = MathOps.gemm(x, @a, rows, @rank, @input_dim, transpose_b: true)
      adapter = MathOps.gemm(hidden, @b, rows, @output_dim, @rank, transpose_b: true)
      output.size.times { |index| output[index] += @scale * adapter[index] }
      output
    end

    # Accumulates LoRA gradients across calls. The frozen quantized weight has
    # no parameter gradient; it is dequantized only when input-gradient flow is
    # requested, then used by the native GEMM transpose path on macOS.
    def backward(x : Array(Float32), grad : Array(Float32), rows : Int32,
                 need_input : Bool = true) : Array(Float32)
      validate_rows_and_input(x, rows)
      expected_grad = rows.to_i64 * @output_dim.to_i64
      raise ArgumentError.new("projection grad size mismatch: expected #{expected_grad}, got #{grad.size}") unless grad.size.to_i64 == expected_grad

      hidden = MathOps.gemm(x, @a, rows, @rank, @input_dim, transpose_b: true)
      grad_b = MathOps.gemm(grad, hidden, @output_dim, @rank, rows, transpose_a: true)
      grad_hidden = MathOps.gemm(grad, @b, rows, @rank, @output_dim)
      grad_a = MathOps.gemm(grad_hidden, x, @rank, @input_dim, rows, transpose_a: true)
      grad_b.size.times { |index| grad_b[index] *= @scale }
      grad_a.size.times { |index| grad_a[index] *= @scale }

      grad_input = if need_input
                     frozen = dequantized_weight_for_backward
                     frozen_input = MathOps.gemm(grad, frozen, rows, @input_dim, @output_dim)
                     adapter_input = MathOps.gemm(grad_hidden, @a, rows, @input_dim, @rank)
                     frozen_input.size.times { |index| frozen_input[index] += @scale * adapter_input[index] }
                     frozen_input
                   else
                     [] of Float32
                   end

      grad_a.size.times { |index| @grad_a[index] += grad_a[index] }
      grad_b.size.times { |index| @grad_b[index] += grad_b[index] }
      grad_input
    end

    private def validate_rows_and_input(x : Array(Float32), rows : Int32) : Nil
      raise ArgumentError.new("projection rows must be positive") unless rows > 0
      expected_input = rows.to_i64 * @input_dim.to_i64
      raise ArgumentError.new("projection input size mismatch: expected #{expected_input}, got #{x.size}") unless x.size.to_i64 == expected_input
    end

    # QuantWeight raw bytes are immutable for the lifetime of this experiment.
    # Retain one FP32 copy only if input-gradient flow is requested, avoiding
    # repeated large dequantization work on every training step.
    private def dequantized_weight_for_backward : Array(Float32)
      if cached = @dequantized_weight
        cached
      else
        decoded = Dequant.dequantize(@weight.raw, @weight.type, @output_dim * @input_dim)
        @dequantized_weight = decoded
        decoded
      end
    end
  end
end
