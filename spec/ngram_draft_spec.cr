require "./spec_helper"
require "../src/ml/gguf/ngram_draft"

describe ML::GGUF::NgramDraft do
  it "returns no candidates when no suffix match reaches the minimum length" do
    history = [1, 2, 3, 4, 5, 6]
    ML::GGUF::NgramDraft.candidates(history, gamma: 4, max_ngram: 4, min_ngram: 3).should eq([] of Int32)
  end

  it "uses the longest repeated suffix and caps the proposed continuation" do
    history = [10, 11, 12, 13, 14, 10, 11, 12]
    ML::GGUF::NgramDraft.candidates(history, gamma: 2, max_ngram: 4, min_ngram: 3).should eq([13, 14])
  end

  it "can recursively extend candidates through its own scratch history" do
    history = [1, 2, 3, 4, 1, 2]
    ML::GGUF::NgramDraft.candidates(history, gamma: 4, max_ngram: 2, min_ngram: 2, recursive: true).should eq([3, 4, 1, 2])
  end

  it "ignores weak short repeats below the minimum length" do
    history = [1, 2, 3, 4, 2, 3]
    ML::GGUF::NgramDraft.candidates(history, gamma: 4, max_ngram: 3, min_ngram: 3).should eq([] of Int32)
  end

  it "rejects invalid parameters" do
    expect_raises(ArgumentError, "gamma must be positive") do
      ML::GGUF::NgramDraft.candidates([1, 2, 1], gamma: 0, max_ngram: 2, min_ngram: 1)
    end

    expect_raises(ArgumentError, "max_ngram must be >= min_ngram") do
      ML::GGUF::NgramDraft.candidates([1, 2, 1], gamma: 1, max_ngram: 1, min_ngram: 2)
    end
  end
end
