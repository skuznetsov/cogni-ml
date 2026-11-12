require "./spec_helper"
require "../src/ml/gguf/qwen35_qlora_projection"

private def qwen35_qlora_projection_f32_weight(values : Array(Float32), out_dim : Int32, in_dim : Int32) : ML::GGUF::QuantWeight
  raw = Bytes.new(values.size * 4)
  values.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, raw[index * 4, 4])
  end
  ML::GGUF::QuantWeight.new(raw, ML::GGUF::TensorType::F32, out_dim, in_dim)
end

private def qwen35_qlora_projection_q8_weight(out_dim : Int32, in_dim : Int32) : ML::GGUF::QuantWeight
  raise "Q8_0 fixture requires rows of 32 elements" unless in_dim == 32
  raw = Bytes.new(out_dim * 34)
  out_dim.times do |row|
    offset = row * 34
    IO::ByteFormat::LittleEndian.encode(0x3c00_u16, raw[offset, 2])
    in_dim.times do |column|
      quantized = ((row * 5 + column * 3) % 15 - 7).to_i8
      raw[offset + 2 + column] = quantized.unsafe_as(UInt8)
    end
  end
  ML::GGUF::QuantWeight.new(raw, ML::GGUF::TensorType::Q8_0, out_dim, in_dim)
end

private def qwen35_qlora_projection_loss(
  projection : ML::GGUF::Qwen35QLoRA::Projection,
  x : Array(Float32), grad : Array(Float32), rows : Int32,
) : Float32
  output = projection.forward(x, rows)
  output.zip(grad).sum { |value, derivative| value * derivative }
end

describe ML::GGUF::Qwen35QLoRA::MathOps do
  it "computes rectangular row-major products with transpose flags" do
    a = [1.0_f32, 2.0_f32, 3.0_f32,
         4.0_f32, 5.0_f32, 6.0_f32]
    b = [1.0_f32, 2.0_f32,
         3.0_f32, 4.0_f32,
         5.0_f32, 6.0_f32]

    ML::GGUF::Qwen35QLoRA::MathOps.gemm(a, b, 2, 2, 3).should eq([
      22.0_f32, 28.0_f32,
      49.0_f32, 64.0_f32,
    ])
    ML::GGUF::Qwen35QLoRA::MathOps.gemm(b, a, 2, 2, 3, transpose_a: true, transpose_b: true).should eq([
      22.0_f32, 49.0_f32,
      28.0_f32, 64.0_f32,
    ])
  end
end

describe ML::GGUF::Qwen35QLoRA::Projection do
  it "matches the native frozen projection and exposes the zero-initialized adapter" do
    weight = qwen35_qlora_projection_f32_weight([
      1.0_f32, -2.0_f32, 0.5_f32,
      -0.25_f32, 4.0_f32, 2.0_f32,
    ], 2, 3)
    projection = ML::GGUF::Qwen35QLoRA::Projection.new(weight, Random.new(42), rank: 2, alpha: 4.0_f32)
    input = [2.0_f32, -1.0_f32, 0.5_f32]

    projection.forward(input, 1).should eq(ML::GGUF::Qwen35CPU.qmatvec_nobias(weight, input))
    projection.a.size.should eq(2 * 3)
    projection.b.size.should eq(2 * 2)
    projection.grad_a.size.should eq(projection.a.size)
    projection.grad_b.size.should eq(projection.b.size)
    projection.zero_grad
    projection.grad_a.should eq(Array(Float32).new(projection.a.size, 0.0_f32))
    projection.grad_b.should eq(Array(Float32).new(projection.b.size, 0.0_f32))
  end

  it "matches rowwise native matvec for a quantized multi-row base projection" do
    weight = qwen35_qlora_projection_q8_weight(2, 32)
    projection = ML::GGUF::Qwen35QLoRA::Projection.new(weight, Random.new(21), rank: 1, alpha: 1.0_f32)
    input = Array(Float32).new(2 * 32) { |index| ((index * 5 % 17) - 8).to_f32 / 7.0_f32 }
    raw_before = weight.raw.dup

    expected = input.each_slice(32).flat_map do |row|
      ML::GGUF::Qwen35CPU.qmatvec_nobias(weight, row.to_a)
    end.to_a
    actual = projection.forward(input, 2)
    actual.size.should eq(expected.size)
    actual.zip(expected).each { |value, reference| value.should be_close(reference, 1e-5_f32) }
    weight.raw.should eq(raw_before)
  end

  {% unless flag?(:cpu_only) %}
    it "returns FP32 results for batched Q5_K and Q6_K Metal projections" do
      pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?
      gemm_switch = "QWEN35_Q56K_BATCH_GEMM_OFF"
      previous = ENV[gemm_switch]?
      ENV.delete(gemm_switch)
      begin
        rows = Math.max(16, ML::GGUF::Qwen35Metal::GEMM_BATCH_THRESHOLD + 1)
        in_dim = 256
        out_dim = 64
        q5_block = Bytes.new(176, 0x33_u8)
        IO::ByteFormat::LittleEndian.encode(0x2800_u16, q5_block[0, 2])
        IO::ByteFormat::LittleEndian.encode(0x2000_u16, q5_block[2, 2])
        q6_block = Bytes.new(210, 0x23_u8)
        q6_block[192, 16].fill(1_u8)
        IO::ByteFormat::LittleEndian.encode(0x2800_u16, q6_block[208, 2])
        input = Array(Float32).new(rows * in_dim) { |i| (i % 7 - 2).to_f32 / 16.0_f32 }

        { {ML::GGUF::TensorType::Q5_K, q5_block}, {ML::GGUF::TensorType::Q6_K, q6_block} }.each do |type, block|
          raw = Bytes.new(out_dim * block.size)
          out_dim.times { |row| raw[row * block.size, block.size].copy_from(block) }
          weight = ML::GGUF::QuantWeight.new(raw, type, out_dim, in_dim)
          dense = ML::GGUF::Dequant.dequantize(raw, type, in_dim * out_dim)
          expected = ML::GGUF::Qwen35QLoRA::MathOps.gemm(input, dense, rows, out_dim, in_dim, transpose_b: true)
          expected.any? { |value| value.abs > 0.1_f32 }.should be_true
          actual = ML::GGUF::Qwen35CPU.qlora_project(weight, input, rows)
          actual.size.should eq(expected.size)
          actual.zip(expected).each do |value, reference|
            value.should be_close(reference, Math.max(1e-3_f32, reference.abs * 5e-4_f32))
          end
        end
      ensure
        if previous
          ENV[gemm_switch] = previous
        else
          ENV.delete(gemm_switch)
        end
      end
    end
  {% end %}

  it "matches finite differences for the frozen input path and both LoRA factors" do
    weight = qwen35_qlora_projection_f32_weight([
      0.25_f32, -0.5_f32, 0.75_f32,
      1.25_f32, 0.5_f32, -1.0_f32,
    ], 2, 3)
    projection = ML::GGUF::Qwen35QLoRA::Projection.new(weight, Random.new(7), rank: 2, alpha: 3.0_f32)
    projection.a.each_index { |index| projection.a[index] = (index + 1).to_f32 / 7.0_f32 - 0.25_f32 }
    projection.b.each_index { |index| projection.b[index] = (index + 2).to_f32 / 5.0_f32 - 0.5_f32 }

    rows = 2
    input = [
      0.5_f32, -1.0_f32, 0.25_f32,
      1.5_f32, 0.75_f32, -0.5_f32,
    ]
    grad = [
      0.25_f32, -0.5_f32,
      1.0_f32, 0.75_f32,
    ]

    projection.zero_grad
    grad_input = projection.backward(input, grad, rows)
    step = 1e-3_f32
    tolerance = 2e-3_f32

    (projection.a.size).times do |index|
      original = projection.a[index]
      projection.a[index] = original + step
      plus = qwen35_qlora_projection_loss(projection, input, grad, rows)
      projection.a[index] = original - step
      minus = qwen35_qlora_projection_loss(projection, input, grad, rows)
      projection.a[index] = original
      numerical = (plus - minus) / (2.0_f32 * step)
      projection.grad_a[index].should be_close(numerical, tolerance)
    end

    (projection.b.size).times do |index|
      original = projection.b[index]
      projection.b[index] = original + step
      plus = qwen35_qlora_projection_loss(projection, input, grad, rows)
      projection.b[index] = original - step
      minus = qwen35_qlora_projection_loss(projection, input, grad, rows)
      projection.b[index] = original
      numerical = (plus - minus) / (2.0_f32 * step)
      projection.grad_b[index].should be_close(numerical, tolerance)
    end

    input.each_index do |index|
      original = input[index]
      input[index] = original + step
      plus = qwen35_qlora_projection_loss(projection, input, grad, rows)
      input[index] = original - step
      minus = qwen35_qlora_projection_loss(projection, input, grad, rows)
      input[index] = original
      numerical = (plus - minus) / (2.0_f32 * step)
      grad_input[index].should be_close(numerical, tolerance)
    end
  end

  it "accumulates factor gradients and can omit the input gradient" do
    weight = qwen35_qlora_projection_f32_weight([
      1.0_f32, 0.0_f32,
      0.0_f32, 1.0_f32,
    ], 2, 2)
    projection = ML::GGUF::Qwen35QLoRA::Projection.new(weight, Random.new(5), rank: 1, alpha: 1.0_f32)
    projection.a[0] = 0.5_f32
    projection.a[1] = -0.25_f32
    projection.b[0] = 0.75_f32
    projection.b[1] = -0.5_f32
    input = [1.0_f32, 2.0_f32]
    grad = [0.5_f32, -1.0_f32]

    projection.zero_grad
    first = projection.backward(input, grad, 1)
    first_a = projection.grad_a.dup
    first_b = projection.grad_b.dup
    second = projection.backward(input, grad, 1, need_input: false)

    second.should be_empty
    projection.grad_a.each_with_index { |value, i| value.should be_close(first_a[i] * 2.0_f32, 1e-6_f32) }
    projection.grad_b.each_with_index { |value, i| value.should be_close(first_b[i] * 2.0_f32, 1e-6_f32) }
    first.size.should eq(input.size)
  end
end
