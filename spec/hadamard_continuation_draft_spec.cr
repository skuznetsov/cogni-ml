require "./spec_helper"
require "../src/ml/gguf/hadamard_continuation_draft"

describe ML::GGUF::HadamardContinuationDraft do
  it "proposes a continuation from a near-repeat structural window" do
    history = [
      100, 1, 200, 2, 300, 3, 400, 401, 402,
      100, 9, 200, 8, 300, 7,
    ]

    span = ML::GGUF::HadamardContinuationDraft::IndexedHistory.new(history, window_size: 6)
      .candidate_span(gamma: 3, max_hamming: 32).not_nil!

    span.ids.should eq([400, 401, 402])
    span.source_start.should eq(0)
    span.window_size.should eq(6)
    span.hamming.should be <= 32
  end

  it "returns no candidate when the nearest window exceeds the hamming gate" do
    history = [
      10, 11, 12, 13,
      20, 21, 22, 23,
      90, 91, 92, 93,
    ]

    ML::GGUF::HadamardContinuationDraft.candidates(history,
      gamma: 2,
      window_size: 4,
      max_hamming: 0).should eq([] of Int32)
  end

  it "respects the economic minimum candidate count" do
    history = [
      1, 2, 3, 4, 99,
      1, 2, 3, 4,
    ]

    index = ML::GGUF::HadamardContinuationDraft::IndexedHistory.new(history, window_size: 4)
    index.candidates(gamma: 4, min_candidates: 5, max_hamming: 0).should eq([] of Int32)
    index.candidates(gamma: 4, min_candidates: 4, max_hamming: 0).should eq([99, 1, 2, 3])
  end

  it "rejects invalid parameters" do
    expect_raises(ArgumentError, "window_size must be positive") do
      ML::GGUF::HadamardContinuationDraft::IndexedHistory.new([] of Int32, window_size: 0)
    end

    expect_raises(ArgumentError, "sketch_bits must be <= 64") do
      ML::GGUF::HadamardContinuationDraft::IndexedHistory.new([] of Int32, sketch_bits: 65)
    end

    expect_raises(ArgumentError, "vector_dim must be a positive power of two") do
      ML::GGUF::HadamardContinuationDraft::IndexedHistory.new([] of Int32, vector_dim: 48)
    end

    index = ML::GGUF::HadamardContinuationDraft::IndexedHistory.new([1, 2, 3, 1, 2, 3], window_size: 3)
    expect_raises(ArgumentError, "gamma must be positive") do
      index.candidates(gamma: 0)
    end
  end
end
