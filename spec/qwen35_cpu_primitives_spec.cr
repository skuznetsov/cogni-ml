require "./spec_helper"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_weights"

QWEN_9B_CPU = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"

describe ML::GGUF::Qwen35CPU do
  describe "rms_norm" do
    it "matches reference formula on fixed input" do
      x = [1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32]
      w = [0.5_f32, 1.0_f32, 1.5_f32, 2.0_f32]
      y = ML::GGUF::Qwen35CPU.rms_norm(x, w, 1.0e-6_f32)
      # mean(x^2) = (1+4+9+16)/4 = 7.5
      # inv_rms = 1/sqrt(7.5 + 1e-6) ≈ 0.36514837
      # y = x * inv_rms * w
      # y[0] = 1 * 0.36514837 * 0.5 ≈ 0.18257
      # y[3] = 4 * 0.36514837 * 2 ≈ 2.92119
      y[0].should be_close(0.18257_f32, 1.0e-4)
      y[1].should be_close(0.73030_f32, 1.0e-4)  # 2 * 0.36514837 * 1
      y[2].should be_close(1.64317_f32, 1.0e-4)  # 3 * 0.36514837 * 1.5
      y[3].should be_close(2.92119_f32, 1.0e-4)
    end
  end

  describe "silu" do
    it "computes x * sigmoid(x) correctly" do
      ML::GGUF::Qwen35CPU.silu(0.0_f32).should be_close(0.0_f32, 1.0e-6)
      ML::GGUF::Qwen35CPU.silu(1.0_f32).should be_close(0.7310586_f32, 1.0e-5) # 1/(1+e^-1)
      ML::GGUF::Qwen35CPU.silu(-1.0_f32).should be_close(-0.2689414_f32, 1.0e-5)
    end
  end

  describe "sigmoid" do
    it "matches logistic function" do
      ML::GGUF::Qwen35CPU.sigmoid(0.0_f32).should be_close(0.5_f32, 1.0e-6)
      ML::GGUF::Qwen35CPU.sigmoid(2.0_f32).should be_close(0.88079708_f32, 1.0e-5)
    end
  end

  describe "rope_partial" do
    it "rotates first n_rot dims and passes through rest" do
      # head_dim=8, n_rot=4, pos=1, freq_base=10.0
      x = [1.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 7.0_f32, 8.0_f32, 9.0_f32, 10.0_f32] # head at offset 0
      ML::GGUF::Qwen35CPU.rope_partial!(x, 0, 4, 8, 1, 10.0_f32)
      # half = 2
      # i=0: freq = 1 / 10^0 = 1; theta=1; cos≈0.5403; sin≈0.8415
      #   i0=0, i1=2 (0 + half)
      #   x0=1, x1=0  →  dst[0]=0.5403, dst[2]=0.8415
      # i=1: freq = 1/10^(2/4) = 1/sqrt(10); theta=1/sqrt(10)
      #   i0=1, i1=3
      #   x0=0, x1=0  →  dst[1]=0, dst[3]=0
      x[0].should be_close(0.5403_f32, 1.0e-3)
      x[2].should be_close(0.8415_f32, 1.0e-3)
      x[1].should be_close(0.0_f32, 1.0e-5)
      x[3].should be_close(0.0_f32, 1.0e-5)
      # Passthrough
      x[4].should eq(7.0_f32)
      x[5].should eq(8.0_f32)
      x[6].should eq(9.0_f32)
      x[7].should eq(10.0_f32)
    end
  end

  describe "softmax_slice" do
    it "produces distribution that sums to 1" do
      x = [0.0_f32, 1.0_f32, 2.0_f32, 3.0_f32, 99.0_f32, 100.0_f32]  # last two in slice [3..6)
      ML::GGUF::Qwen35CPU.softmax_slice!(x, 3, 3)
      s = 0.0_f32
      3.times { |i| s += x[3 + i] }
      s.should be_close(1.0_f32, 1.0e-5)
      # 100 should dominate
      x[5].should be > x[4]
      x[4].should be > x[3]
    end
  end

  describe "l2_norm_slice" do
    it "normalizes slice to unit norm" do
      x = [3.0_f32, 4.0_f32, 0.0_f32, 0.0_f32]
      ML::GGUF::Qwen35CPU.l2_norm_slice!(x, 0, 2, 1.0e-6_f32)
      x[0].should be_close(0.6_f32, 1.0e-5)  # 3/5
      x[1].should be_close(0.8_f32, 1.0e-5)  # 4/5
      x[2].should eq(0.0_f32)
      x[3].should eq(0.0_f32)
    end
  end

  describe "embedding_lookup" do
    it "returns [n_embd] vector for a valid token id on Qwen 3.5 9B" do
      pending!("9B model not present") unless File.exists?(QWEN_9B_CPU)
      w = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_CPU)

      emb = ML::GGUF::Qwen35CPU.embedding_lookup(w.token_embd, 0)
      emb.size.should eq(w.hparams.n_embd)
      # embedding should not be all-zero in general
      nonzero = emb.count { |v| v != 0.0_f32 }
      nonzero.should be > 100

      # Different tokens → different embeddings
      emb2 = ML::GGUF::Qwen35CPU.embedding_lookup(w.token_embd, 42)
      (emb.zip(emb2).any? { |a, b| a != b }).should be_true
    end
  end
end
