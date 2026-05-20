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

  it "keeps indexed candidates equivalent to the stateless scanner" do
    history = [10, 11, 12, 13, 14, 10, 11, 12]
    index = ML::GGUF::NgramDraft::IndexedHistory.new(history, max_ngram: 4, min_ngram: 2)

    index.candidates(gamma: 4).should eq(ML::GGUF::NgramDraft.candidates(history, gamma: 4, max_ngram: 4, min_ngram: 2))
    index.match_len.should eq(ML::GGUF::NgramDraft.match_len(history, max_ngram: 4, min_ngram: 2))
  end

  it "updates indexed candidates incrementally" do
    index = ML::GGUF::NgramDraft::IndexedHistory.new([1, 2, 3, 4], max_ngram: 2, min_ngram: 2)
    index.candidates(gamma: 4).should eq([] of Int32)

    index.append(1)
    index.append(2)
    index.candidates(gamma: 4).should eq([3, 4, 1, 2])
    index.match_len.should eq(2)
  end

  it "keeps indexed recursive expansion equivalent to the stateless scanner" do
    history = [1, 2, 3, 4, 1, 2]
    index = ML::GGUF::NgramDraft::IndexedHistory.new(history, max_ngram: 2, min_ngram: 2)

    index.candidates(gamma: 4, recursive: true).should eq(ML::GGUF::NgramDraft.candidates(history, gamma: 4, max_ngram: 2, min_ngram: 2, recursive: true))
  end

  it "reports the source span used for indexed candidates" do
    history = [10, 11, 12, 13, 14, 10, 11, 12]
    index = ML::GGUF::NgramDraft::IndexedHistory.new(history, max_ngram: 4, min_ngram: 3)

    span = index.candidate_span(gamma: 2).not_nil!
    span.ids.should eq([13, 14])
    span.match_len.should eq(3)
    span.source_start.should eq(0)
  end

  it "reports the suffix match length used by the n-gram draft" do
    history = [1, 2, 3, 4, 1, 2]

    ML::GGUF::NgramDraft.match_len(history, max_ngram: 4, min_ngram: 2).should eq(2)
  end

  it "ignores weak short repeats below the minimum length" do
    history = [1, 2, 3, 4, 2, 3]
    ML::GGUF::NgramDraft.candidates(history, gamma: 4, max_ngram: 3, min_ngram: 3).should eq([] of Int32)
  end

  it "can suppress tiny candidate chunks below an economic minimum" do
    history = [1, 2, 1]

    ML::GGUF::NgramDraft.candidates(history, gamma: 4, max_ngram: 1, min_ngram: 1, min_candidates: 3).should eq([] of Int32)
    ML::GGUF::NgramDraft.candidates(history, gamma: 4, max_ngram: 1, min_ngram: 1, min_candidates: 2).should eq([2, 1])
  end

  it "accounts full-accept-only fixed split replay economics" do
    expected = [10, 11, 12, 13, 14, 15, 16, 17]
    actual = [10, 11, 99, 13, 14, 15, 16, 17]

    result = ML::GGUF::NgramDraft.fixed_split_acceptance(expected, actual, split_size: 4)
    result.chunks.should eq([4])
    result.full_accept_chunks.should eq(0)
    result.verified_tokens.should eq(4)
    result.committed_tokens.should eq(0)
    result.discarded_accept_prefix.should eq(2)
    result.reject_index.should eq(2)
    result.full_accept.should be_false
  end

  it "accounts progressive replay schedules that grow after full accepts" do
    expected = [10, 11, 12, 13, 14, 15, 16, 17]
    actual = [10, 11, 12, 13, 14, 99, 16, 17]

    result = ML::GGUF::NgramDraft.schedule_acceptance(expected, actual, [1, 1, 2, 2, 4])
    result.chunks.should eq([1, 1, 2, 2])
    result.full_accept_chunks.should eq(3)
    result.verified_tokens.should eq(6)
    result.committed_tokens.should eq(4)
    result.discarded_accept_prefix.should eq(1)
    result.reject_index.should eq(5)
    result.full_accept.should be_false
  end

  it "accounts clean progressive replay schedules" do
    expected = [10, 11, 12, 13, 14, 15, 16, 17]

    result = ML::GGUF::NgramDraft.schedule_acceptance(expected, expected, [1, 1, 2, 2, 4])
    result.chunks.should eq([1, 1, 2, 2, 2])
    result.full_accept_chunks.should eq(5)
    result.verified_tokens.should eq(8)
    result.committed_tokens.should eq(8)
    result.discarded_accept_prefix.should eq(0)
    result.reject_index.should eq(-1)
    result.full_accept.should be_true
  end

  it "rejects invalid parameters" do
    expect_raises(ArgumentError, "gamma must be positive") do
      ML::GGUF::NgramDraft.candidates([1, 2, 1], gamma: 0, max_ngram: 2, min_ngram: 1)
    end

    expect_raises(ArgumentError, "max_ngram must be >= min_ngram") do
      ML::GGUF::NgramDraft.candidates([1, 2, 1], gamma: 1, max_ngram: 1, min_ngram: 2)
    end

    expect_raises(ArgumentError, "min_candidates must be non-negative") do
      ML::GGUF::NgramDraft.candidates([1, 2, 1], gamma: 1, max_ngram: 1, min_ngram: 1, min_candidates: -1)
    end

    expect_raises(ArgumentError, "schedule must not be empty") do
      ML::GGUF::NgramDraft.schedule_acceptance([1], [1], [] of Int32)
    end

    expect_raises(ArgumentError, "schedule chunk sizes must be positive") do
      ML::GGUF::NgramDraft.schedule_acceptance([1], [1], [1, 0])
    end
  end

  it "recognizes exact candidate periods for conservative risk gating" do
    ids = [1, 2, 3, 4, 5, 6, 7, 8] * 2

    ML::GGUF::NgramDraft.exact_period(ids, 8).should eq(8)
    ML::GGUF::NgramDraft.risky_candidate_shape?(ids, min_size: 16).should be_true
  end

  it "does not risk-gate short or compact low-period candidates" do
    short_ids = [1, 2, 3, 4, 5, 6, 7, 8]
    compact_ids = [1, 2, 3, 4] * 4

    ML::GGUF::NgramDraft.risky_candidate_shape?(short_ids, min_size: 16).should be_false
    ML::GGUF::NgramDraft.risky_candidate_shape?(compact_ids, min_size: 16).should be_false
  end

  it "risk-gates short period-eight candidate tails when they repeat tokens internally" do
    ids = [1, 2, 3, 4, 5, 6, 1, 7]

    ML::GGUF::NgramDraft.exact_period(ids, 8).should eq(8)
    ML::GGUF::NgramDraft.unique_ratio(ids).should be < 0.95
    ML::GGUF::NgramDraft.risky_candidate_shape?(ids, min_size: 16, match_len: 5).should be_true
  end

  it "does not risk-gate short table-like repeats with only a short suffix match" do
    ids = [735, 7993, 735, 220, 18, 735, 735, 5388]

    ML::GGUF::NgramDraft.exact_period(ids, 8).should eq(8)
    ML::GGUF::NgramDraft.risky_candidate_shape?(ids, min_size: 16, match_len: 4).should be_false
    ML::GGUF::NgramDraft.risky_candidate_shape?(ids, min_size: 16, match_len: 5).should be_true
  end

  it "keeps medium chunks with strong lag-eight continuation" do
    ids = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3]

    ML::GGUF::NgramDraft.lag_ratio(ids, 8).should eq(1.0)
    ML::GGUF::NgramDraft.risky_candidate_shape?(ids, min_size: 16, match_len: 2).should be_false
  end

  it "risk-gates medium chunks without strong lag-eight continuation" do
    code_like = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4]
    math_like = [1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 10, 11]

    ML::GGUF::NgramDraft.lag_ratio(code_like, 8).should be < 0.75
    ML::GGUF::NgramDraft.risky_candidate_shape?(code_like, min_size: 16, match_len: 2).should be_true
    ML::GGUF::NgramDraft.lag_ratio(math_like, 8).should be < 0.75
    ML::GGUF::NgramDraft.risky_candidate_shape?(math_like, min_size: 16, match_len: 2).should be_true
  end

  it "risk-gates small-period prefixes that overrun into a different tail" do
    ids = [15, 13, 15, 13, 15, 13, 16, 198, 220, 471, 803, 25, 13053, 198]

    ML::GGUF::NgramDraft.prefix_period_run(ids, 4).should eq(6)
    ML::GGUF::NgramDraft.exact_period(ids, 4).should eq(0)
    ML::GGUF::NgramDraft.risky_candidate_shape?(ids, min_size: 16, match_len: 6).should be_true
  end

  it "does not risk-gate fully periodic compact candidate chunks" do
    ids = [8029, 13053, 20956] * 4

    ML::GGUF::NgramDraft.prefix_period_run(ids, 4).should eq(ids.size)
    ML::GGUF::NgramDraft.exact_period(ids, 4).should eq(3)
    ML::GGUF::NgramDraft.risky_candidate_shape?(ids, min_size: 16, match_len: 6).should be_false
  end

  it "detects non-repeating high-diversity candidate tails" do
    ids = (1..16).to_a

    ML::GGUF::NgramDraft.pair_unique_ratio(ids).should eq(1.0)
    ML::GGUF::NgramDraft.lag_ratio(ids, 4).should eq(0.0)
    ML::GGUF::NgramDraft.entropy_norm(ids).should eq(1.0)
    ML::GGUF::NgramDraft.risky_candidate_shape?(ids, min_size: 16).should be_true
    ML::GGUF::NgramDraft.corridor_candidate_shape?(ids).should be_false
  end

  it "corridor-gates only candidate shapes with enough repeating transport evidence" do
    one_token = [42]
    lag_four = [1, 2, 3, 4, 1, 9, 10, 11]
    low_entropy = [7, 7, 8, 7, 7, 8, 7, 7]
    weak_nearmiss = [198, 220, 471, 850, 25, 220, 17, 198, 262, 803, 25, 8029, 198, 262, 869, 25]

    ML::GGUF::NgramDraft.corridor_candidate_shape?(one_token).should be_false
    ML::GGUF::NgramDraft.corridor_candidate_shape?(lag_four).should be_true
    ML::GGUF::NgramDraft.corridor_candidate_shape?(weak_nearmiss, match_len: 8, match_len_min: 8).should be_true
    ML::GGUF::NgramDraft.entropy_norm(low_entropy).should be <= 0.6
    ML::GGUF::NgramDraft.corridor_candidate_shape?(low_entropy).should be_true
    ML::GGUF::NgramDraft.lag_ratio(weak_nearmiss, 4).should be < 0.25
    ML::GGUF::NgramDraft.lag_ratio(weak_nearmiss, 8).should be < 0.5
    ML::GGUF::NgramDraft.entropy_norm(weak_nearmiss).should be > 0.6
    ML::GGUF::NgramDraft.corridor_candidate_shape?(weak_nearmiss).should be_false
  end

  it "risk-gates structured YAML-like tails with weak lag-four reuse" do
    ids = [198, 220, 471, 850, 25, 220, 17, 198, 262, 803, 25, 8029, 198, 262, 869, 25]

    ML::GGUF::NgramDraft.pair_unique_ratio(ids).should be > 0.90
    ML::GGUF::NgramDraft.lag_ratio(ids, 4).should be < 0.10
    ML::GGUF::NgramDraft.lag_ratio(ids, 8).should be < 0.20
    ML::GGUF::NgramDraft.risky_candidate_shape?(ids, min_size: 16, match_len: 8).should be_true
  end

  it "risk-gates CSV-like diverse tails with only weak lag-four reuse" do
    ids = [1, 2, 3, 4, 1, 6, 7, 8, 1, 10, 11, 12, 13, 14, 15, 16]

    ML::GGUF::NgramDraft.pair_unique_ratio(ids).should eq(1.0)
    ML::GGUF::NgramDraft.lag_ratio(ids, 4).should be > 0.10
    ML::GGUF::NgramDraft.lag_ratio(ids, 4).should be < 0.20
    ML::GGUF::NgramDraft.lag_ratio(ids, 8).should be < 0.20
    ML::GGUF::NgramDraft.risky_candidate_shape?(ids, min_size: 16, match_len: 3).should be_true
  end
end
