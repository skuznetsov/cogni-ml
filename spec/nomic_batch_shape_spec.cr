require "./spec_helper"
require "../src/ml/gguf/nomic_batch_shape"

describe ML::GGUF::NomicBatchShape do
  it "rounds sequence lengths to reusable power-of-two buckets" do
    ML::GGUF::NomicBatchShape.bucket_seq_len(1, 512, enabled: true).should eq(1)
    ML::GGUF::NomicBatchShape.bucket_seq_len(17, 512, enabled: true).should eq(32)
    ML::GGUF::NomicBatchShape.bucket_seq_len(187, 512, enabled: true).should eq(256)
    ML::GGUF::NomicBatchShape.bucket_seq_len(201, 512, enabled: true).should eq(256)
    ML::GGUF::NomicBatchShape.bucket_seq_len(257, 512, enabled: true).should eq(512)
  end

  it "caps the physical bucket at the model context length" do
    ML::GGUF::NomicBatchShape.bucket_seq_len(300, 384, enabled: true).should eq(384)
    ML::GGUF::NomicBatchShape.bucket_seq_len(384, 384, enabled: true).should eq(384)
  end

  it "can leave shapes exact for A/B comparison" do
    ML::GGUF::NomicBatchShape.bucket_seq_len(187, 512, enabled: false).should eq(187)
  end

  it "reports padded token overhead for a batch" do
    lengths = [187, 201, 64]
    ML::GGUF::NomicBatchShape.padded_token_overhead(lengths, 256).should eq(316)
  end
end
