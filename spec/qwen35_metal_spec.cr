require "./spec_helper"
require "../src/ml/gguf/qwen35_weights"
require "../src/ml/gguf/qwen35_metal"
require "../src/ml/gguf/quant_matmul"
require "../src/ml/gguf/reader"

QWEN_9B_METAL  = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
QWEN_08B_METAL = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"

# Borrow a real quantized weight from the 9B model and return raw bytes
# + dimensions. Caller asserts the expected TensorType.
def quant_tensor_bytes(model_path : String, name : String,
                       expected : ML::GGUF::TensorType) : {Bytes, Int32, Int32}
  g = ML::GGUF::GGUFFile.new(model_path)
  info = g.tensor(name).not_nil!
  raise "expected #{expected} tensor, got #{info.type}" unless info.type == expected
  raw = g.read_tensor_raw(info)
  # GGUF dims are [in_dim, out_dim] for matmul weights (dims[0] innermost).
  in_dim = info.dims[0].to_i32
  out_dim = info.dims[1].to_i32
  {raw, in_dim, out_dim}
end

def q4k_tensor_bytes(model_path : String, name : String) : {Bytes, Int32, Int32}
  quant_tensor_bytes(model_path, name, ML::GGUF::TensorType::Q4_K)
end

def cosine(a : Array(Float32), b : Array(Float32)) : Float64
  dot = 0.0
  na = 0.0
  nb = 0.0
  a.each_with_index do |av, i|
    bv = b[i]
    dot += av.to_f64 * bv.to_f64
    na += av.to_f64 * av.to_f64
    nb += bv.to_f64 * bv.to_f64
  end
  dot / (Math.sqrt(na) * Math.sqrt(nb))
end

def max_abs_diff(a : Array(Float32), b : Array(Float32)) : Float32
  m = 0.0_f32
  a.each_with_index do |av, i|
    d = (av - b[i]).abs
    m = d if d > m
  end
  m
end

describe ML::GGUF::Qwen35Metal do
  pending!("9B model not present") unless File.exists?(QWEN_9B_METAL)
  pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

  it "matmul_q4k GEMV (batch=1) matches CPU reference" do
    w_raw, in_dim, out_dim = q4k_tensor_bytes(QWEN_9B_METAL, "blk.0.ffn_up.weight")
    in_dim.should eq(4096)
    out_dim.should eq(12288)

    rng = Random.new(42)
    x = Array(Float32).new(in_dim) { rng.rand(-1.0_f32..1.0_f32) }
    zero_bias = Array(Float32).new(out_dim, 0.0_f32)

    t0 = Time.instant
    gpu = ML::GGUF::Qwen35Metal.matmul_q4k(x, w_raw, in_dim, out_dim, 1)
    dt_gpu = Time.instant - t0
    puts "  [metal_q4k_gemv] GPU: #{dt_gpu.total_milliseconds.round(1)} ms"

    t0 = Time.instant
    cpu = ML::GGUF::QuantMatmul.matmul_add(x, 1, in_dim, w_raw, ML::GGUF::TensorType::Q4_K, out_dim, zero_bias)
    dt_cpu = Time.instant - t0
    puts "  [metal_q4k_gemv] CPU: #{dt_cpu.total_milliseconds.round(1)} ms"

    gpu.size.should eq(out_dim)
    cos = cosine(gpu, cpu)
    diff = max_abs_diff(gpu, cpu)
    puts "  [metal_q4k_gemv] cos=#{cos.round(6)}, max|Δ|=#{diff}"
    cos.should be >= 0.9999
    diff.should be < 0.01_f32 * cpu.map(&.abs).max
  end

  it "matmul_q4k GEMM (batch=16) matches CPU reference" do
    w_raw, in_dim, out_dim = q4k_tensor_bytes(QWEN_9B_METAL, "blk.0.ffn_up.weight")
    batch = 16

    rng = Random.new(123)
    x = Array(Float32).new(batch * in_dim) { rng.rand(-1.0_f32..1.0_f32) }
    zero_bias = Array(Float32).new(out_dim, 0.0_f32)

    t0 = Time.instant
    gpu = ML::GGUF::Qwen35Metal.matmul_q4k(x, w_raw, in_dim, out_dim, batch)
    dt_gpu = Time.instant - t0
    puts "  [metal_q4k_gemm] GPU: #{dt_gpu.total_milliseconds.round(1)} ms"

    t0 = Time.instant
    cpu = ML::GGUF::QuantMatmul.matmul_add(x, batch, in_dim, w_raw, ML::GGUF::TensorType::Q4_K, out_dim, zero_bias)
    dt_cpu = Time.instant - t0
    puts "  [metal_q4k_gemm] CPU: #{dt_cpu.total_milliseconds.round(1)} ms"

    gpu.size.should eq(batch * out_dim)
    cos = cosine(gpu, cpu)
    diff = max_abs_diff(gpu, cpu)
    puts "  [metal_q4k_gemm] cos=#{cos.round(6)}, max|Δ|=#{diff}"
    cos.should be >= 0.9999
    diff.should be < 0.05_f32 * cpu.map(&.abs).max
  end

  it "matmul_q4k handles the ffn_down shape (12288→4096)" do
    w_raw, in_dim, out_dim = q4k_tensor_bytes(QWEN_9B_METAL, "blk.4.ffn_down.weight")
    in_dim.should eq(12288)
    out_dim.should eq(4096)

    rng = Random.new(7)
    x = Array(Float32).new(in_dim) { rng.rand(-1.0_f32..1.0_f32) }
    zero_bias = Array(Float32).new(out_dim, 0.0_f32)

    gpu = ML::GGUF::Qwen35Metal.matmul_q4k(x, w_raw, in_dim, out_dim, 1)
    cpu = ML::GGUF::QuantMatmul.matmul_add(x, 1, in_dim, w_raw, ML::GGUF::TensorType::Q4_K, out_dim, zero_bias)

    cos = cosine(gpu, cpu)
    diff = max_abs_diff(gpu, cpu)
    puts "  [metal_q4k_ffndown] cos=#{cos.round(6)}, max|Δ|=#{diff}"
    cos.should be >= 0.9999
  end

  it "matmul_q5k GEMV matches CPU reference" do
    w_raw, in_dim, out_dim = quant_tensor_bytes(
      QWEN_9B_METAL, "blk.0.attn_qkv.weight", ML::GGUF::TensorType::Q5_K)
    rng = Random.new(11)
    x = Array(Float32).new(in_dim) { rng.rand(-1.0_f32..1.0_f32) }
    zero_bias = Array(Float32).new(out_dim, 0.0_f32)

    t0 = Time.instant
    gpu = ML::GGUF::Qwen35Metal.matmul_q5k(x, w_raw, in_dim, out_dim, 1)
    dt_gpu = Time.instant - t0
    t0 = Time.instant
    cpu = ML::GGUF::QuantMatmul.matmul_add(x, 1, in_dim, w_raw, ML::GGUF::TensorType::Q5_K, out_dim, zero_bias)
    dt_cpu = Time.instant - t0

    cos = cosine(gpu, cpu)
    diff = max_abs_diff(gpu, cpu)
    puts "  [metal_q5k_gemv] GPU: #{dt_gpu.total_milliseconds.round(1)} ms, CPU: #{dt_cpu.total_milliseconds.round(1)} ms"
    puts "  [metal_q5k_gemv] cos=#{cos.round(6)}, max|Δ|=#{diff}  (#{in_dim}→#{out_dim})"
    cos.should be >= 0.9999
  end

  it "matmul_q5k GEMM (batch=16) matches CPU reference" do
    w_raw, in_dim, out_dim = quant_tensor_bytes(
      QWEN_9B_METAL, "blk.0.attn_qkv.weight", ML::GGUF::TensorType::Q5_K)
    batch = 16
    rng = Random.new(111)
    x = Array(Float32).new(batch * in_dim) { rng.rand(-1.0_f32..1.0_f32) }
    zero_bias = Array(Float32).new(out_dim, 0.0_f32)

    old = ENV["QWEN35_Q56K_BATCH_GEMM_OFF"]?
    ENV.delete("QWEN35_Q56K_BATCH_GEMM_OFF")
    t0 = Time.instant
    begin
      gpu = ML::GGUF::Qwen35Metal.matmul_q5k(x, w_raw, in_dim, out_dim, batch)
    ensure
      if old
        ENV["QWEN35_Q56K_BATCH_GEMM_OFF"] = old
      else
        ENV.delete("QWEN35_Q56K_BATCH_GEMM_OFF")
      end
    end
    dt_gpu = Time.instant - t0
    t0 = Time.instant
    cpu = ML::GGUF::QuantMatmul.matmul_add(x, batch, in_dim, w_raw, ML::GGUF::TensorType::Q5_K, out_dim, zero_bias)
    dt_cpu = Time.instant - t0

    cos = cosine(gpu, cpu)
    diff = max_abs_diff(gpu, cpu)
    puts "  [metal_q5k_gemm] GPU: #{dt_gpu.total_milliseconds.round(1)} ms, CPU: #{dt_cpu.total_milliseconds.round(1)} ms"
    puts "  [metal_q5k_gemm] cos=#{cos.round(6)}, max|Δ|=#{diff}  (batch=#{batch}, #{in_dim}→#{out_dim})"
    cos.should be >= 0.999
  end

  it "matmul_q6k GEMV matches CPU reference (ffn_down)" do
    w_raw, in_dim, out_dim = quant_tensor_bytes(
      QWEN_9B_METAL, "blk.0.ffn_down.weight", ML::GGUF::TensorType::Q6_K)
    rng = Random.new(13)
    x = Array(Float32).new(in_dim) { rng.rand(-1.0_f32..1.0_f32) }
    zero_bias = Array(Float32).new(out_dim, 0.0_f32)

    t0 = Time.instant
    gpu = ML::GGUF::Qwen35Metal.matmul_q6k(x, w_raw, in_dim, out_dim, 1)
    dt_gpu = Time.instant - t0
    t0 = Time.instant
    cpu = ML::GGUF::QuantMatmul.matmul_add(x, 1, in_dim, w_raw, ML::GGUF::TensorType::Q6_K, out_dim, zero_bias)
    dt_cpu = Time.instant - t0

    cos = cosine(gpu, cpu)
    diff = max_abs_diff(gpu, cpu)
    puts "  [metal_q6k_gemv] GPU: #{dt_gpu.total_milliseconds.round(1)} ms, CPU: #{dt_cpu.total_milliseconds.round(1)} ms"
    puts "  [metal_q6k_gemv] cos=#{cos.round(6)}, max|Δ|=#{diff}  (#{in_dim}→#{out_dim})"
    cos.should be >= 0.9999
  end

  it "matmul_q6k GEMM (batch=16) matches CPU reference" do
    w_raw, in_dim, out_dim = quant_tensor_bytes(
      QWEN_9B_METAL, "blk.0.ffn_down.weight", ML::GGUF::TensorType::Q6_K)
    batch = 16
    rng = Random.new(113)
    x = Array(Float32).new(batch * in_dim) { rng.rand(-1.0_f32..1.0_f32) }
    zero_bias = Array(Float32).new(out_dim, 0.0_f32)

    old = ENV["QWEN35_Q56K_BATCH_GEMM_OFF"]?
    ENV.delete("QWEN35_Q56K_BATCH_GEMM_OFF")
    t0 = Time.instant
    begin
      gpu = ML::GGUF::Qwen35Metal.matmul_q6k(x, w_raw, in_dim, out_dim, batch)
    ensure
      if old
        ENV["QWEN35_Q56K_BATCH_GEMM_OFF"] = old
      else
        ENV.delete("QWEN35_Q56K_BATCH_GEMM_OFF")
      end
    end
    dt_gpu = Time.instant - t0
    t0 = Time.instant
    cpu = ML::GGUF::QuantMatmul.matmul_add(x, batch, in_dim, w_raw, ML::GGUF::TensorType::Q6_K, out_dim, zero_bias)
    dt_cpu = Time.instant - t0

    cos = cosine(gpu, cpu)
    diff = max_abs_diff(gpu, cpu)
    puts "  [metal_q6k_gemm] GPU: #{dt_gpu.total_milliseconds.round(1)} ms, CPU: #{dt_cpu.total_milliseconds.round(1)} ms"
    puts "  [metal_q6k_gemm] cos=#{cos.round(6)}, max|Δ|=#{diff}  (batch=#{batch}, #{in_dim}→#{out_dim})"
    cos.should be >= 0.999
  end

  it "matmul_q6k GEMV matches CPU on lm_head shape (4096→248320)" do
    w_raw, in_dim, out_dim = quant_tensor_bytes(
      QWEN_9B_METAL, "output.weight", ML::GGUF::TensorType::Q6_K)
    in_dim.should eq(4096)
    out_dim.should eq(248320)

    rng = Random.new(17)
    x = Array(Float32).new(in_dim) { rng.rand(-1.0_f32..1.0_f32) }
    zero_bias = Array(Float32).new(out_dim, 0.0_f32)

    t0 = Time.instant
    gpu = ML::GGUF::Qwen35Metal.matmul_q6k(x, w_raw, in_dim, out_dim, 1)
    dt_gpu = Time.instant - t0
    t0 = Time.instant
    cpu = ML::GGUF::QuantMatmul.matmul_add(x, 1, in_dim, w_raw, ML::GGUF::TensorType::Q6_K, out_dim, zero_bias)
    dt_cpu = Time.instant - t0

    cos = cosine(gpu, cpu)
    diff = max_abs_diff(gpu, cpu)
    puts "  [metal_q6k_lmhead] GPU: #{dt_gpu.total_milliseconds.round(1)} ms, CPU: #{dt_cpu.total_milliseconds.round(1)} ms"
    puts "  [metal_q6k_lmhead] cos=#{cos.round(6)}, max|Δ|=#{diff}"
    cos.should be >= 0.9999
  end

  it "matmul_q8_0 GEMV matches CPU reference when 0.8B draft is present" do
    pending!("0.8B Q8_0 model not present") unless File.exists?(QWEN_08B_METAL)
    w_raw, in_dim, out_dim = quant_tensor_bytes(
      QWEN_08B_METAL, "blk.0.ffn_up.weight", ML::GGUF::TensorType::Q8_0)
    in_dim.should eq(1024)
    out_dim.should eq(3584)

    rng = Random.new(23)
    x = Array(Float32).new(in_dim) { rng.rand(-1.0_f32..1.0_f32) }
    zero_bias = Array(Float32).new(out_dim, 0.0_f32)

    t0 = Time.instant
    gpu = ML::GGUF::Qwen35Metal.matmul_q8_0(x, w_raw, in_dim, out_dim, 1)
    dt_gpu = Time.instant - t0
    t0 = Time.instant
    cpu = ML::GGUF::QuantMatmul.matmul_add(x, 1, in_dim, w_raw, ML::GGUF::TensorType::Q8_0, out_dim, zero_bias)
    dt_cpu = Time.instant - t0

    cos = cosine(gpu, cpu)
    diff = max_abs_diff(gpu, cpu)
    puts "  [metal_q8_0_gemv] GPU: #{dt_gpu.total_milliseconds.round(1)} ms, CPU: #{dt_cpu.total_milliseconds.round(1)} ms"
    puts "  [metal_q8_0_gemv] cos=#{cos.round(6)}, max|Δ|=#{diff}  (#{in_dim}→#{out_dim})"
    cos.should be >= 0.9999
  end
end
