require "./spec_helper"
require "../src/ml/gguf/qwen35_qlora"

private def with_qlora_checkpoint_header(header : String, &)
  file = File.tempfile("qwen35-qlora-checkpoint")
  begin
    file << header
    file.close
    yield file.path
  ensure
    file.close unless file.closed?
    File.delete?(file.path)
  end
end

describe ML::GGUF::Qwen35QLoRA::Model do
  digest = "a" * 64
  version1 = "COGNI-QWEN35-QLORA-1\n#{digest}\n"
  version2 = "COGNI-QWEN35-QLORA-2\n#{digest}\n"

  it "detects the placement count in legacy, four-block, and six-block checkpoints" do
    {
      {version1, 1},
      {version2 + "11,15,19,23\n", 4},
      {version2 + "3,7,11,15,19,23\n", 6},
    }.each do |header, expected|
      with_qlora_checkpoint_header(header) do |path|
        ML::GGUF::Qwen35QLoRA::Model.adapter_full_blocks(path).should eq(expected)
      end
    end
  end

  it "rejects malformed and unsupported placement headers before choosing a generation route" do
    {
      "3,7,11,15,19,23",
      "03,7,11,15,19,23\n",
      "3,7,11,15,15,23\n",
      "3,7,15,11,19,23\n",
      "-3,7,11,15,19,23\n",
      "3,7,11,15,23\n",
      "3,7,11,15,19,23,27\n",
    }.each do |placement|
      with_qlora_checkpoint_header(version2 + placement) do |path|
        expect_raises(ArgumentError, "QLoRA adapter block placement") do
          ML::GGUF::Qwen35QLoRA::Model.adapter_full_blocks(path)
        end
      end
    end
  end
end
