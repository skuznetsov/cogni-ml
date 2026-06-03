require "./spec_helper"
require "../src/ml/gguf/gemma4_cpu"
require "../src/ml/gguf/qwen35_metal"

GEMMA4_METAL_BUFFER_12B_Q4KM = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf"

def gemma4_buffer_max_abs_diff(a : Array(Float32), b : Array(Float32)) : Float32
  max = 0.0_f32
  a.each_with_index do |av, i|
    diff = (av - b[i]).abs
    max = diff if diff > max
  end
  max
end

def gemma4_expect_resident_matmul_matches(label : String,
                                          qw : ML::GGUF::QuantWeight,
                                          x : Array(Float32)) : Nil
  expected = ML::GGUF::Qwen35Metal.matmul(qw, x, 1).not_nil!
  x_buf = ML::MetalBuffer.from_array(x)
  out_buf = ML::MetalBuffer.new(qw.out_dim.to_i64 * sizeof(Float32))

  ML::GGUF::Qwen35Metal.matmul_to_buffer(qw, x_buf, out_buf, 1).should be_true
  actual = out_buf.read(qw.out_dim)
  diff = gemma4_buffer_max_abs_diff(expected, actual)
  puts "  [#{label}] max|d|=#{diff}"
  diff.should be <= 1.0e-6_f32
end

describe "Gemma4 resident Metal matmul buffers" do
  pending!("Gemma4 12B GGUF not found") unless File.exists?(GEMMA4_METAL_BUFFER_12B_Q4KM)
  pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

  it "projects Q4 and Q6 Gemma4 weights from resident input buffers like the array path" do
    w = ML::GGUF::Gemma4Weights.from_gguf(GEMMA4_METAL_BUFFER_12B_Q4KM)
    x = ML::GGUF::Gemma4CPU.embedding_lookup(w.token_embd, 42)
    lw = w.layers[0]
    x_norm = ML::GGUF::Gemma4CPU.rms_norm(x, lw.attn_norm, w.hparams.rms_eps)

    ML::GGUF::Qwen35Metal.matmul_to_buffer(lw.attn_q_qw, ML::MetalBuffer.from_array(x_norm), ML::MetalBuffer.new(lw.attn_q_qw.out_dim.to_i64 * sizeof(Float32)), 0).should be_false
    gemma4_expect_resident_matmul_matches("gemma4_resident_q4_attn_q", lw.attn_q_qw, x_norm)

    ctx = Array(Float32).new(lw.ffn_down_qw.in_dim) { |i| Math.sin(i.to_f32 * 0.013_f32).to_f32 * 0.25_f32 }
    gemma4_expect_resident_matmul_matches("gemma4_resident_q6_ffn_down", lw.ffn_down_qw, ctx)
  end
end
