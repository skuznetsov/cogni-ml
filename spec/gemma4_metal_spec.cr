require "./spec_helper"
require "../src/ml/gguf/gemma4_cpu"
require "../src/ml/gguf/qwen35_metal"

GEMMA4_METAL_12B_Q4KM = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf"

def gemma4_metal_cosine(a : Array(Float32), b : Array(Float32)) : Float64
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

def gemma4_metal_max_abs_diff(a : Array(Float32), b : Array(Float32)) : Float32
  max = 0.0_f32
  a.each_with_index do |av, i|
    diff = (av - b[i]).abs
    max = diff if diff > max
  end
  max
end

describe "Gemma4 Metal primitives" do
  pending!("Gemma4 12B GGUF not found") unless File.exists?(GEMMA4_METAL_12B_Q4KM)
  pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

  it "dequantizes a Q6_K token embedding row on Metal like the CPU reference" do
    w = ML::GGUF::Gemma4Weights.from_gguf(GEMMA4_METAL_12B_Q4KM)
    w.token_embd.type.q6_k?.should be_true

    token_id = 42
    cpu = ML::GGUF::Gemma4CPU.embedding_lookup(w.token_embd, token_id)
    gpu = ML::GGUF::Qwen35Metal.embedding_q6k_from_token_id(w.token_embd, token_id).not_nil!

    gpu.size.should eq(w.hparams.n_embd)
    cos = gemma4_metal_cosine(cpu, gpu)
    diff = gemma4_metal_max_abs_diff(cpu, gpu)
    puts "  [gemma4_q6k_embed] cos=#{cos.round(8)}, max|d|=#{diff}"
    cos.should be >= 0.999999
    diff.should be < 1.0e-6_f32
  end

  it "zeros an out-of-range Q6_K token embedding request" do
    w = ML::GGUF::Gemma4Weights.from_gguf(GEMMA4_METAL_12B_Q4KM)
    gpu = ML::GGUF::Qwen35Metal.embedding_q6k_from_token_id(w.token_embd, w.token_embd.out_dim).not_nil!

    gpu.size.should eq(w.hparams.n_embd)
    gpu.all? { |v| v == 0.0_f32 }.should be_true
  end
end
