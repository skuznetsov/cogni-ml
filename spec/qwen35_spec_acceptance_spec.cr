require "./spec_helper"
require "../src/ml/gguf/qwen35_spec_acceptance"

describe ML::GGUF::Qwen35SpecAcceptance do
  it "accepts a full candidate span and advances expected ids" do
    result = ML::GGUF::Qwen35SpecAcceptance.scan(
      [10_i32, 11_i32, 12_i32],
      10_i32,
      [{11_i32, 0.0_f32}, {12_i32, 0.0_f32}, {13_i32, 0.0_f32}],
      8,
    )

    result.emitted.should eq([10, 11, 12])
    result.expected_ids.should eq([10, 11, 12])
    result.accepted.should eq(3)
    result.rejected.should be_false
    result.reject_index.should be_nil
    result.next_expected.should eq(13)
    result.full_accept?.should be_true
  end

  it "emits the exact correction on first mismatch" do
    result = ML::GGUF::Qwen35SpecAcceptance.scan(
      [10_i32, 99_i32, 12_i32],
      10_i32,
      [{11_i32, 0.0_f32}, {12_i32, 0.0_f32}],
      8,
    )

    result.emitted.should eq([10, 11])
    result.expected_ids.should eq([10, 11])
    result.accepted.should eq(1)
    result.rejected.should be_true
    result.reject_index.should eq(1)
    result.next_expected.should eq(11)
    result.full_accept?.should be_false
  end

  it "stops at output cap before scanning the full candidate" do
    result = ML::GGUF::Qwen35SpecAcceptance.scan(
      [1_i32, 2_i32, 3_i32],
      1_i32,
      [{2_i32, 0.0_f32}, {3_i32, 0.0_f32}],
      2,
    )

    result.emitted.should eq([1, 2])
    result.expected_ids.should eq([1, 2])
    result.accepted.should eq(2)
    result.rejected.should be_false
  end

  it "stops on EOS after accepting it" do
    result = ML::GGUF::Qwen35SpecAcceptance.scan(
      [1_i32, 2_i32, 3_i32],
      1_i32,
      [{2_i32, 0.0_f32}, {3_i32, 0.0_f32}],
      8,
      eos_id: 2_i32,
    )

    result.emitted.should eq([1, 2])
    result.expected_ids.should eq([1, 2])
    result.accepted.should eq(2)
    result.rejected.should be_false
  end
end
