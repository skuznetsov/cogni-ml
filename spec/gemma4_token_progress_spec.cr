require "spec"
require "../src/ml/gguf/gemma4_token_progress"

private def gemma4_partial(tokens : Array(Int32), predicted : Int32,
                           content : String = "", stop : Bool = false) : String
  {content: content, tokens: tokens, tokens_predicted: predicted, stop: stop}.to_json
end

private def gemma4_final(tokens : Array(Int32), predicted : Int32,
                         timing_predicted : Int32 = predicted) : String
  {content: "", tokens: tokens, tokens_predicted: predicted, stop: true,
   stop_type: "limit", tokens_evaluated: 12, truncated: false,
   timings: {predicted_n: timing_predicted}}.to_json
end

describe ML::GGUF::Gemma4TokenProgress::Collector do
  it "counts token IDs from chunks even when content is empty and a chunk has multiple IDs" do
    collector = ML::GGUF::Gemma4TokenProgress::Collector.new(1_000_i64, 3)
    collector.consume(gemma4_partial([7, 8], 2, content: ""), 1_015_i64, 15_i64)
    collector.consume(gemma4_partial([9], 3, content: "text is not the counter"), 1_030_i64, 30_i64)
    collector.generated_tokens.should eq(3)
    collector.token_ids.should eq([7_i64, 8_i64, 9_i64])
  end

  it "does not count a matching cumulative final token snapshot twice" do
    collector = ML::GGUF::Gemma4TokenProgress::Collector.new(2_000_i64, 3)
    collector.consume(gemma4_partial([4], 1), 2_010_i64, 10_i64)
    collector.consume(gemma4_partial([4, 4], 3), 2_020_i64, 20_i64)
    collector.consume(gemma4_final([4, 4, 4], 3), 2_030_i64, 30_i64)
    collector.finish!(2_045_i64, 45_i64)
    collector.generated_tokens.should eq(3)
    collector.token_ids.should eq([4_i64, 4_i64, 4_i64])
    collector.to_json["status"].as_s.should eq("available")
    collector.to_json["points"].as_a.last["elapsed_ms"].as_i64.should eq(45)
    collector.to_json["points"].as_a.last["generated_tokens"].as_i64.should eq(3)
  end

  it "coalesces chunks received within the same millisecond" do
    collector = ML::GGUF::Gemma4TokenProgress::Collector.new(3_000_i64, 2)
    collector.consume(gemma4_partial([11], 1), 3_010_i64, 10_i64)
    collector.consume(gemma4_partial([12], 2), 3_010_i64, 11_i64)
    collector.points.map { |point| {point.elapsed_ms, point.generated_tokens} }
      .should eq([{0_i64, 0_i64}, {10_i64, 2_i64}])
  end

  it "rejects a counter gap, an inconsistent final timing, and a truncated stream" do
    gap = ML::GGUF::Gemma4TokenProgress::Collector.new(4_000_i64, 2)
    expect_raises(ArgumentError, /token IDs/) do
      gap.consume(gemma4_partial([1], 2), 4_010_i64, 10_i64)
    end

    timing = ML::GGUF::Gemma4TokenProgress::Collector.new(4_000_i64, 1)
    timing.consume(gemma4_partial([1], 1), 4_010_i64, 10_i64)
    expect_raises(ArgumentError, /timings/) do
      timing.consume(gemma4_final([] of Int32, 1, timing_predicted: 0), 4_020_i64, 20_i64)
    end

    truncated = ML::GGUF::Gemma4TokenProgress::Collector.new(4_000_i64, 2)
    truncated.consume(gemma4_partial([1], 1), 4_010_i64, 10_i64)
    expect_raises(ArgumentError, /truncated/) { truncated.finish!(4_020_i64, 20_i64) }
  end

  it "rejects a mismatched final snapshot and backwards receipt time" do
    mismatch = ML::GGUF::Gemma4TokenProgress::Collector.new(5_000_i64, 2)
    mismatch.consume(gemma4_partial([1], 1), 5_010_i64, 10_i64)
    expect_raises(ArgumentError, /token IDs/) do
      mismatch.consume(gemma4_final([2, 2], 2), 5_020_i64, 20_i64)
    end

    backwards = ML::GGUF::Gemma4TokenProgress::Collector.new(5_000_i64, 2)
    backwards.consume(gemma4_partial([1], 1), 5_010_i64, 10_i64)
    expect_raises(ArgumentError, /timestamp|clock/) do
      backwards.consume(gemma4_partial([2], 2), 5_009_i64, 11_i64)
    end
  end

  it "keeps observed partial progress when the stream has no final event" do
    collector = ML::GGUF::Gemma4TokenProgress::Collector.new(6_000_i64, 2)
    collector.consume(gemma4_partial([31], 1), 6_012_i64, 12_i64)
    expect_raises(ArgumentError, /truncated/) { collector.finish!(6_020_i64, 20_i64) }
    partial = collector.to_json(status: "partial")
    partial["observed_tokens"].as_i64.should eq(1)
    partial["points"].as_a.last["generated_tokens"].as_i64.should eq(1)
    partial["final_tokens"].raw.should be_nil
  end
end
