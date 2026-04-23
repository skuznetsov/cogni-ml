require "./spec_helper"
require "../src/ml/gguf/reader"
require "../src/ml/gguf/dequant"
require "../src/ml/gguf/quant_matmul"

# Correctness test for ML::GGUF::QuantMatmul.matmul_add_q4k.
#
# Reference: bulk dequantize Q4_K → dense F32 matmul.
# Target:    fused matmul_add_q4k walks blocks and accumulates in one pass.
# They must agree to ~fp32 precision.

QWEN_9B_Q4K_MM = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"

private def ref_matmul_via_dequant(
  x : Array(Float32), rows : Int32, in_dim : Int32,
  w_raw : Bytes, out_dim : Int32, bias : Array(Float32),
) : Array(Float32)
  # Dequantize entire weight matrix, then do a plain F32 matmul.
  n_w = out_dim * in_dim
  w_f32 = ML::GGUF::Dequant.dequantize_q4_k(w_raw, n_w)
  result = Array(Float32).new(rows * out_dim, 0.0_f32)
  rows.times do |r|
    x_off = r * in_dim
    r_off = r * out_dim
    out_dim.times do |o|
      sum = bias[o].to_f64
      w_off = o * in_dim
      in_dim.times { |j| sum += x[x_off + j].to_f64 * w_f32[w_off + j].to_f64 }
      result[r_off + o] = sum.to_f32
    end
  end
  result
end

private def cosine(a : Array(Float32), b : Array(Float32)) : Float64
  dot = 0.0; na = 0.0; nb = 0.0
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

describe ML::GGUF::QuantMatmul do
  describe "matmul_add_q4k" do
    it "matches bulk-dequant + dense matmul on blk.0.attn_gate (4096x4096)" do
      pending!("9B model not present") unless File.exists?(QWEN_9B_Q4K_MM)

      g = ML::GGUF::GGUFFile.new(QWEN_9B_Q4K_MM)
      info = g.tensor("blk.0.attn_gate.weight").not_nil!
      info.type.q4_k?.should be_true
      in_dim  = info.dims[0].to_i32
      out_dim = info.dims[1].to_i32
      in_dim.should eq(4096)
      out_dim.should eq(4096)
      w_raw = g.read_tensor_raw(info).dup
      g.close

      # Deterministic pseudo-random input, no RNG seed dependency
      rows = 2
      x = Array(Float32).new(rows * in_dim) do |i|
        v = ((i.to_u32! &* 2654435761_u32) % 2000_u32).to_i32 - 1000 # -1000..999
        v / 1000.0_f32
      end
      bias = Array(Float32).new(out_dim) { |i| (i % 10) / 100.0_f32 }

      mine = ML::GGUF::QuantMatmul.matmul_add(x, rows, in_dim, w_raw, ML::GGUF::TensorType::Q4_K, out_dim, bias)
      ref  = ref_matmul_via_dequant(x, rows, in_dim, w_raw, out_dim, bias)

      cos = cosine(mine, ref)
      maxd = max_abs_diff(mine, ref)
      puts "[Q4_K mm/attn_gate] rows=#{rows} in=#{in_dim} out=#{out_dim} cos=#{cos} max_abs_diff=#{maxd}"

      cos.should be >= 0.9999999
      # fp32 accumulation order differs slightly; tolerance relative to magnitudes
      mag = 0.0_f32
      ref.each { |v| mag = v.abs if v.abs > mag }
      (maxd / mag).should be <= 1.0e-5_f32
    end
  end
end
