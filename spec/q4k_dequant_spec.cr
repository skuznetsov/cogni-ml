require "./spec_helper"
require "../src/ml/gguf/reader"
require "../src/ml/gguf/dequant"

# Correctness test: Crystal Q4_K dequant vs llama.cpp's dequantize_row_q4_K.
# The helper binary at spec/support/q4k_ref is built from spec/support/q4k_ref.c
# and links libggml-base (see spec/support/q4k_ref.c header for build command).

QWEN_9B_Q4K_PATH = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
Q4K_REF_BIN      = File.expand_path("support/q4k_ref", __DIR__)

private def cosine(a : Array(Float32), b : Array(Float32)) : Float64
  raise "size mismatch" unless a.size == b.size
  dot = 0.0
  na = 0.0
  nb = 0.0
  a.size.times do |i|
    dot += a[i].to_f64 * b[i].to_f64
    na += a[i].to_f64 * a[i].to_f64
    nb += b[i].to_f64 * b[i].to_f64
  end
  dot / (Math.sqrt(na) * Math.sqrt(nb))
end

private def max_abs_diff(a : Array(Float32), b : Array(Float32)) : Float32
  m = 0.0_f32
  a.size.times do |i|
    d = (a[i] - b[i]).abs
    m = d if d > m
  end
  m
end

private def run_q4k_ref(raw : Bytes, n : Int32) : Array(Float32)
  raw_path = "/tmp/q4k_raw.bin"
  ref_path = "/tmp/q4k_ref.bin"
  File.write(raw_path, raw)
  status = Process.run(Q4K_REF_BIN, [raw_path, n.to_s, ref_path])
  raise "q4k_ref failed: status=#{status.exit_code}" unless status.success?
  bytes = File.read(ref_path).to_slice
  result = Array(Float32).new(n, 0.0_f32)
  n.times do |i|
    result[i] = IO::ByteFormat::LittleEndian.decode(Float32, bytes[i * 4, 4])
  end
  result
end

describe ML::GGUF::Dequant do
  describe "dequantize_q4_k" do
    it "matches libggml dequantize_row_q4_K on blk.0.ssm_alpha.weight (n=131072)" do
      pending!("9B model not present") unless File.exists?(QWEN_9B_Q4K_PATH)
      pending!("q4k_ref helper not built (run: clang -O2 -o spec/support/q4k_ref ... -lggml-base)") unless File.exists?(Q4K_REF_BIN)

      g = ML::GGUF::GGUFFile.new(QWEN_9B_Q4K_PATH)
      info = g.tensor("blk.0.ssm_alpha.weight").not_nil!
      info.type.q4_k?.should be_true
      raw = g.read_tensor_raw(info).dup
      n = info.n_elements.to_i32

      mine = ML::GGUF::Dequant.dequantize_q4_k(raw, n)
      ref  = run_q4k_ref(raw, n)
      g.close

      mine.size.should eq(ref.size)
      mine.size.should eq(n)

      cos = cosine(mine, ref)
      maxd = max_abs_diff(mine, ref)
      puts "[Q4_K/ssm_alpha]  cos=#{cos}  max_abs_diff=#{maxd}"

      # Direct port of same algorithm — should be bit-identical in fp32
      cos.should be >= 0.99999999
      maxd.should be <= 1.0e-6_f32
    end

    it "matches libggml on a larger tensor blk.0.ffn_up.weight" do
      pending!("9B model not present") unless File.exists?(QWEN_9B_Q4K_PATH)
      pending!("q4k_ref helper not built") unless File.exists?(Q4K_REF_BIN)

      g = ML::GGUF::GGUFFile.new(QWEN_9B_Q4K_PATH)
      info = g.tensor("blk.0.ffn_up.weight").not_nil!
      info.type.q4_k?.should be_true
      raw = g.read_tensor_raw(info).dup
      n = info.n_elements.to_i32

      mine = ML::GGUF::Dequant.dequantize_q4_k(raw, n)
      ref  = run_q4k_ref(raw, n)
      g.close

      cos = cosine(mine, ref)
      maxd = max_abs_diff(mine, ref)
      puts "[Q4_K/ffn_up]  cos=#{cos}  max_abs_diff=#{maxd}"

      cos.should be >= 0.99999999
      maxd.should be <= 1.0e-6_f32
    end
  end
end
