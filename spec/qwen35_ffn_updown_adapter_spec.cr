require "./spec_helper"
require "file_utils"
require "../src/ml/gguf/qwen35_ffn_updown_adapter"

describe ML::GGUF::Qwen35FFNUpDownAdapter do
  it "projects through the same centered low-rank formula used by the probe" do
    adapter = ML::GGUF::Qwen35FFNUpDownAdapter.new(
      x_mean: [1.0, -1.0],
      c_mean: [0.5, -0.25],
      coeff_weights: [
        [2.0, 0.0],
        [0.0, -1.0],
      ],
      down_basis: [
        [1.0_f32, 2.0_f32],
        [3.0_f32, -1.0_f32],
      ],
    )

    out = adapter.project([2.0_f32, 1.0_f32], 2)
    out[0].should be_close(-4.25_f32, 1.0e-6)
    out[1].should be_close(7.25_f32, 1.0e-6)
  end

  it "round-trips the adapter artifact format" do
    root = File.tempname("qwen35-ffn-updown-adapter")
    Dir.mkdir_p(root)
    path = File.join(root, "adapters.json")
    begin
      adapters = ML::GGUF::Qwen35FFNUpDownAdapterMap{
        4 => ML::GGUF::Qwen35FFNUpDownAdapter.new(
          x_mean: [1.0, 2.0],
          c_mean: [0.25],
          coeff_weights: [[0.5, -0.5]],
          down_basis: [[2.0_f32, 3.0_f32]],
        ),
      }

      ML::GGUF::Qwen35FFNUpDownAdapterArtifact.dump(path, adapters, rank: 1, hidden_dim: 2, source: "spec")
      loaded = ML::GGUF::Qwen35FFNUpDownAdapterArtifact.load(path)

      loaded[:source].should eq("spec")
      loaded[:hidden_dim].should eq(2)
      loaded[:rank].should eq(1)
      loaded_adapter = loaded[:adapters][4]
      loaded_adapter.x_mean.should eq([1.0, 2.0])
      loaded_adapter.c_mean.should eq([0.25])
      loaded_adapter.coeff_weights.should eq([[0.5, -0.5]])
      loaded_adapter.down_basis.should eq([[2.0_f32, 3.0_f32]])
    ensure
      FileUtils.rm_rf(root) if Dir.exists?(root)
    end
  end

  it "keeps normalized block Hadamard self-inverse" do
    values = [1.0, 2.0, -3.0, 4.0]
    tmp = values.dup
    ML::GGUF::Qwen35FFNUpDownAdapter.block_hadamard_inplace!(tmp, 4)
    ML::GGUF::Qwen35FFNUpDownAdapter.block_hadamard_inplace!(tmp, 4)

    tmp.zip(values).each do |actual, expected|
      actual.should be_close(expected, 1.0e-9)
    end
  end

  it "validates quantization parameters" do
    expect_raises(Exception, /quant bits/) do
      ML::GGUF::Qwen35FFNUpDownAdapter.symmetric_quant_dequant([1.0], 1)
    end
    expect_raises(Exception, /positive power of two/) do
      ML::GGUF::Qwen35FFNUpDownAdapter.block_hadamard_inplace!([1.0, 2.0], 3)
    end
  end
end
